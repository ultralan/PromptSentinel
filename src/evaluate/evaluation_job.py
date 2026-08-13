"""在独立 prepared test split 上完成多候选模型的公平评估。"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config_entity import DataConfig, EvaluationCandidateConfig, EvaluationConfig
from src.core.data_entity import ClassificationRecord, DatasetSplit
from src.preprocess.prepared_dataset import PreparedDataset
from src.evaluate.evaluation_run import EvaluationRun
from src.evaluate.metrics import BinaryMetrics
from src.evaluate.model_evaluator import ModelEvaluator
from src.train.model_trainer.sequence_classification_trainer import SequenceClassificationTrainer


class EvaluationJob:
    """按统一验证阈值和独立测试集，比较基座与所有训练候选。"""

    def __init__(
        self,
        config: EvaluationConfig,
        data_config: DataConfig,
        prepared_dataset: PreparedDataset,
        model_evaluator: ModelEvaluator,
        evaluation_run: EvaluationRun,
        model_trainer: SequenceClassificationTrainer,
    ) -> None:
        """注入配置、标准数据、模型推理器和本次评测运行对象。"""
        self._config = config
        self._data_config = data_config
        self._prepared_dataset = prepared_dataset
        self._model_evaluator = model_evaluator
        self._evaluation_run = evaluation_run
        self._model_trainer = model_trainer

    def run(self) -> None:
        """对全部候选执行 validation 阈值校准和官方 test 评测。"""
        manifest = self._prepared_dataset.load_manifest()
        if manifest.max_length != self._config.model.max_length:
            raise ValueError("prepared manifest 与评测模型的 max_length 不一致")
        validation_records = self._prepared_dataset.load_training_split(DatasetSplit.VALIDATION)
        test_records = self._prepared_dataset.load_test_split()
        self._evaluation_run.initialize(
            self._data_config,
            {"validation": len(validation_records), "test": len(test_records)},
            self._model_trainer.select_device(),
        )
        reports = [self._evaluate_candidate(candidate, validation_records, test_records) for candidate in self._config.candidates]
        self._evaluation_run.write_json(
            {
                "threshold_source": "prepared validation split 的良性样本",
                "target_false_positive_rate": self._config.evaluation.target_false_positive_rate,
                "test_source": "prepared test split",
                "candidates": reports,
            },
            "metrics.json",
        )

    def _evaluate_candidate(
        self,
        candidate: EvaluationCandidateConfig,
        validation_records: list[ClassificationRecord],
        test_records: list[ClassificationRecord],
    ) -> dict:
        """校准一个候选模型，并保存其独立测试预测与汇总指标。"""
        tokenizer, model, device = self._model_evaluator.load(candidate)
        validation_scores = self._model_evaluator.predict(
            tokenizer, model, device, validation_records, f"{candidate.name}/validation",
        )
        threshold = BinaryMetrics.calibrate_threshold(
            [record.label for record in validation_records],
            validation_scores,
            self._config.evaluation.target_false_positive_rate,
        )
        validation_metrics = BinaryMetrics.evaluate(
            [record.label for record in validation_records], validation_scores, threshold,
        )
        test_scores = self._model_evaluator.predict(tokenizer, model, device, test_records, f"{candidate.name}/test")
        test_metrics = BinaryMetrics.evaluate([record.label for record in test_records], test_scores, threshold)
        self._write_predictions(candidate.name, test_records, test_scores, threshold)
        return {
            "name": candidate.name,
            "artifact_type": candidate.artifact_type.value,
            "path": str(candidate.path) if candidate.path is not None else None,
            "validation": validation_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        }

    def _write_predictions(
        self,
        candidate_name: str,
        records: list[ClassificationRecord],
        probabilities: list[float],
        threshold: float,
    ) -> Path:
        """写入可审计的 test 逐样本概率、标签和阈值预测。"""
        path = self._evaluation_run.predictions_path(candidate_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record, probability in zip(records, probabilities, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "id": record.id,
                            "label": record.label,
                            "risk_probability": probability,
                            "prediction": int(probability >= threshold),
                            "threshold": threshold,
                            "token_length": record.token_length,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path
