"""原始能力保持评测业务对象。"""

from __future__ import annotations

from src.core.config_entity import EvaluationConfig
from src.core.data_entity import ClassificationRecord
from src.evaluate.metrics import BinaryMetrics
from src.evaluate.model_evaluator import ModelEvaluator
from src.evaluate.evaluation_run import EvaluationRun


class CapabilityRetentionJob:
    """比较基座正确边界在微调后是否被遗忘。"""

    def __init__(self, config: EvaluationConfig, evaluator: ModelEvaluator, run: EvaluationRun) -> None:
        """注入候选模型、统一推理器和结果运行目录。"""
        self._config = config
        self._evaluator = evaluator
        self._run = run

    def run(self, records: list[ClassificationRecord]) -> None:
        """以固定 0.5 阈值评估三模型，并计算相对基座的遗忘率。"""
        labels = [record.label for record in records]
        scores_by_name: dict[str, list[float]] = {}
        reports = []
        for candidate in self._config.candidates:
            tokenizer, model, device = self._evaluator.load(candidate)
            scores = self._evaluator.predict(tokenizer, model, device, records, f"capability/{candidate.name}")
            scores_by_name[candidate.name] = scores
            metrics = BinaryMetrics.evaluate(labels, scores, 0.5)
            reports.append({"name": candidate.name, "metrics": self._with_classification_metrics(metrics.to_dict())})
        base_scores = scores_by_name["base"]
        base_correct = [int(score >= 0.5) == label for score, label in zip(base_scores, labels, strict=True)]
        for report in reports:
            if report["name"] == "base":
                report["forgetting_rate"] = 0.0
                continue
            scores = scores_by_name[report["name"]]
            forgotten = sum(
                was_correct and int(score >= 0.5) != label
                for was_correct, score, label in zip(base_correct, scores, labels, strict=True)
            )
            report["forgetting_rate"] = forgotten / sum(base_correct)
        self._run.write_json({"dataset": "jackhhao/jailbreak-classification", "threshold": 0.5, "records": len(records), "reports": reports}, "capability_metrics.json")

    @staticmethod
    def _with_classification_metrics(metrics: dict) -> dict:
        """补充 Balanced Accuracy 与 F1，保持原有混淆矩阵可审计。"""
        true_positive, false_positive = metrics["true_positive"], metrics["false_positive"]
        true_negative, false_negative = metrics["true_negative"], metrics["false_negative"]
        recall = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        metrics["balanced_accuracy"] = (recall + specificity) / 2
        metrics["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return metrics
