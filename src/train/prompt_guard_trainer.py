"""Prompt Guard 2 的模型加载与 Transformers Trainer 工厂对象。"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from src.core.config_entity import ModelConfig, TrainerConfig


class PromptGuardTrainer:
    """封装 Prompt Guard 2 的加载、标签校验和 Transformers Trainer 创建。"""

    def __init__(self, model_config: ModelConfig, trainer_config: TrainerConfig) -> None:
        """绑定模型身份配置与训练器超参数。"""
        self._model_config = model_config
        self._trainer_config = trainer_config

    def load_tokenizer(self) -> Any:
        """加载 Prompt Guard 2 tokenizer 并保证动态 padding 可用。"""
        tokenizer = AutoTokenizer.from_pretrained(self._model_config.name_or_path, revision=self._model_config.revision, use_fast=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            raise ValueError("Prompt Guard 2 tokenizer 没有可用 pad token")
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self) -> Any:
        """加载二分类基座，并确认 label=1 表示风险正类。"""
        model = AutoModelForSequenceClassification.from_pretrained(self._model_config.name_or_path, revision=self._model_config.revision)
        if model.config.num_labels != 2:
            raise ValueError(f"Prompt Guard 2 应为二分类模型，实际为 {model.config.num_labels}")
        label = str(model.config.id2label.get(1, "")).lower()
        if not any(term in label for term in ("injection", "malicious", "unsafe", "attack")):
            raise ValueError(f"模型 label=1 不可识别为风险正类: {model.config.id2label.get(1)!r}")
        return model

    def create_trainer(self, model: Any, tokenizer: Any, train_dataset: Any, validation_dataset: Any, checkpoint_dir: Path, device: str) -> Trainer:
        """根据绑定配置创建带早停回调的 Transformers Trainer。"""
        config = self._trainer_config
        arguments = TrainingArguments(
            output_dir=str(checkpoint_dir), num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio, logging_steps=config.logging_steps,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            save_total_limit=config.save_total_limit, report_to="none",
            use_mps_device=device == "mps", fp16=False, bf16=False, optim="adamw_torch",
        )
        return Trainer(
            model=model, args=arguments, train_dataset=train_dataset, eval_dataset=validation_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        )

    @staticmethod
    def select_device() -> str:
        """优先选择 MPS，其次 CUDA，最后回退 CPU。"""
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
