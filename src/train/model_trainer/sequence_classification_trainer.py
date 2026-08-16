"""序列分类模型与 Transformers Trainer 的通用训练边界。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from src.core.config_entity import ModelConfig, TrainerConfig
from src.core.training_entity import MixedPrecision


class SequenceClassificationTrainer(ABC):
    """封装任意二分类序列分类基座的加载和 Trainer 创建。"""

    def __init__(self, model_config: ModelConfig, trainer_config: TrainerConfig | None = None) -> None:
        """绑定模型身份；仅训练场景额外绑定 Trainer 超参数。"""
        self._model_config = model_config
        self._trainer_config = trainer_config

    @abstractmethod
    def load_tokenizer(self) -> Any:
        """加载与当前模型权重严格匹配的 tokenizer。"""
        raise NotImplementedError

    @abstractmethod
    def load_base_model(self) -> Any:
        """加载并校验风险正类为 label=1 的二分类模型。"""
        raise NotImplementedError

    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        validation_dataset: Any,
        checkpoint_dir: Path,
        device: str,
    ) -> Trainer:
        """按统一训练口径创建带动态 padding 和早停的 Trainer。"""
        if self._trainer_config is None:
            raise RuntimeError("评测加载器不能创建 Transformers Trainer")
        config = self._trainer_config
        self._validate_mixed_precision(config.mixed_precision, device)
        total_steps = self._total_optimization_steps(train_dataset)
        arguments = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=ceil(total_steps * config.warmup_ratio),
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=config.save_total_limit,
            report_to="none",
            fp16=config.mixed_precision is MixedPrecision.FP16,
            bf16=config.mixed_precision is MixedPrecision.BF16,
            optim="adamw_torch",
        )
        return Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        )

    @staticmethod
    def _validate_mixed_precision(mixed_precision: MixedPrecision, device: str) -> None:
        """限制 FP16/BF16 仅在 CUDA 设备上启用，避免 CPU 或 MPS 静默降级。"""
        if mixed_precision is not MixedPrecision.FP32 and device != "cuda":
            raise ValueError(f"{mixed_precision.value} 仅支持 CUDA 训练，当前设备为 {device}")

    def _total_optimization_steps(self, train_dataset: Any) -> int:
        """按批大小和梯度累积计算当前训练配置的总优化器更新次数。"""
        if self._trainer_config is None:
            raise RuntimeError("评测加载器不能计算训练优化步数")
        config = self._trainer_config
        batches_per_epoch = ceil(len(train_dataset) / config.per_device_train_batch_size)
        updates_per_epoch = ceil(batches_per_epoch / config.gradient_accumulation_steps)
        return ceil(updates_per_epoch * config.num_train_epochs)

    @staticmethod
    def select_device() -> str:
        """优先选择 MPS，其次 CUDA，最后回退 CPU。"""
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
