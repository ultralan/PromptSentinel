"""基座与训练产物的统一加载和批量分类推理。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from peft import PeftModel

from src.core.config_entity import EvaluationCandidateConfig, EvaluationConfig
from src.core.data_entity import ClassificationRecord
from src.core.training_entity import ModelArtifactType
from src.train.model_trainer.sequence_classification_trainer import SequenceClassificationTrainer


class ModelEvaluator:
    """以完全相同的 tokenization 和推理实现比较每个候选模型。"""

    def __init__(self, config: EvaluationConfig, model_trainer: SequenceClassificationTrainer) -> None:
        """绑定评测配置和已经按模型身份选择的加载器。"""
        self._config = config
        self._model_trainer = model_trainer

    def load(self, candidate: EvaluationCandidateConfig) -> tuple[Any, Any, str]:
        """加载一个候选模型及 tokenizer，并切换到当前可用推理设备。"""
        tokenizer = self._model_trainer.load_tokenizer()
        model = self._load_model(candidate)
        device = self._model_trainer.select_device()
        model.to(device)
        model.eval()
        return tokenizer, model, device

    def predict(
        self,
        tokenizer: Any,
        model: Any,
        device: str,
        records: list[ClassificationRecord],
        stage_name: str,
    ) -> list[float]:
        """用已加载模型计算风险正类概率，并定期记录本地推理进度。"""
        probabilities: list[float] = []
        batch_size = self._config.evaluation.per_device_batch_size
        total_batches = (len(records) + batch_size - 1) // batch_size
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                inputs = tokenizer(
                    [record.text for record in batch],
                    truncation=True,
                    max_length=self._config.model.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = model(**{name: value.to(device) for name, value in inputs.items()}).logits
                probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist())
                batch_index = start // batch_size + 1
                if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
                    logging.getLogger(__name__).info(
                        "%s 推理进度: %s/%s batches (%s/%s samples)",
                        stage_name,
                        batch_index,
                        total_batches,
                        min(start + batch_size, len(records)),
                        len(records),
                    )
        return probabilities

    def _load_model(self, candidate: EvaluationCandidateConfig) -> Any:
        """按候选产物类型加载基座、LoRA adapter 或完整微调权重。"""
        if candidate.artifact_type is ModelArtifactType.BASE:
            return self._model_trainer.load_base_model()
        if candidate.path is None or not candidate.path.is_dir():
            raise FileNotFoundError(f"候选模型产物不存在: {candidate.path}")
        if candidate.artifact_type is ModelArtifactType.LORA_ADAPTER:
            return PeftModel.from_pretrained(self._model_trainer.load_base_model(), candidate.path)
        if candidate.artifact_type is ModelArtifactType.FULL_FINE_TUNED:
            from transformers import AutoModelForSequenceClassification

            return AutoModelForSequenceClassification.from_pretrained(candidate.path)
        raise ValueError(f"不支持的候选模型类型: {candidate.artifact_type}")
