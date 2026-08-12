"""Protect AI DeBERTa v2 基座的模型加载实现。"""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train.sequence_classification_trainer import SequenceClassificationTrainer


class ProtectAiDebertaTrainer(SequenceClassificationTrainer):
    """加载英语 prompt injection 二分类基座并校验其标签语义。"""

    def load_tokenizer(self) -> Any:
        """加载 Protect AI DeBERTa tokenizer 并启用右侧动态 padding。"""
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.name_or_path,
            revision=self._model_config.revision,
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            raise ValueError("Protect AI DeBERTa tokenizer 没有可用 pad token")
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self) -> Any:
        """加载二分类权重，并确认 label=1 表示 prompt injection。"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_config.name_or_path,
            revision=self._model_config.revision,
        )
        if model.config.num_labels != 2:
            raise ValueError(f"Protect AI DeBERTa 应为二分类模型，实际为 {model.config.num_labels}")
        label = str(model.config.id2label.get(1, "")).lower()
        if "injection" not in label:
            raise ValueError(f"模型 label=1 不可识别为 prompt injection: {model.config.id2label.get(1)!r}")
        return model
