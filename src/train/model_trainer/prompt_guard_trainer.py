"""Prompt Guard 2 的模型加载实现。"""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train.model_trainer.sequence_classification_trainer import SequenceClassificationTrainer


class PromptGuardTrainer(SequenceClassificationTrainer):
    """封装 Prompt Guard 2 的加载、标签校验和 Transformers Trainer 创建。"""

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
