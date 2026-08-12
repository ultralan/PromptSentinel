"""序列分类模型与 Transformers Trainer 的通用训练边界。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from src.core.config_entity import ModelConfig, TrainerConfig


class SequenceClassificationTrainer(ABC):
    """封装任意二分类序列分类基座的加载和 Trainer 创建。"""

    def __init__(self, model_config: ModelConfig, trainer_config: TrainerConfig) -> None:
        """绑定不可变的模型身份与通用训练超参数。"""
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
        config = self._trainer_config
        arguments = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=config.save_total_limit,
            report_to="none",
            use_mps_device=device == "mps",
            fp16=False,
            bf16=False,
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
    def select_device() -> str:
        """优先选择 MPS，其次 CUDA，最后回退 CPU。"""
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
