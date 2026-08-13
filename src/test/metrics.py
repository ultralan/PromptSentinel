"""二分类风险检测的无依赖可复现指标计算。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """单个候选模型在一个数据集上的排序和阈值指标。"""

    auc_roc: float
    average_precision: float
    threshold: float
    false_positive_rate: float
    recall: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def to_dict(self) -> dict:
        """转换为稳定 JSON 字段映射。"""
        return asdict(self)


class BinaryMetrics:
    """计算概率分数的 AUC、AP 与 validation 校准后的二分类结果。"""

    @staticmethod
    def calibrate_threshold(labels: list[int], probabilities: list[float], target_fpr: float) -> float:
        """在良性 validation 上选择满足目标 FPR 的最低可用阈值。"""
        BinaryMetrics._validate(labels, probabilities)
        clean_scores = sorted((score for label, score in zip(labels, probabilities, strict=True) if label == 0), reverse=True)
        if not clean_scores:
            raise ValueError("阈值校准需要至少一条良性 validation 样本")
        allowed_false_positives = int(np.floor(len(clean_scores) * target_fpr))
        threshold = float(np.nextafter(clean_scores[0], np.inf))
        accepted = 0
        index = 0
        while index < len(clean_scores):
            end = index + 1
            while end < len(clean_scores) and clean_scores[end] == clean_scores[index]:
                end += 1
            if accepted + (end - index) > allowed_false_positives:
                break
            threshold = clean_scores[index]
            accepted = end
            index = end
        return float(threshold)

    @staticmethod
    def evaluate(labels: list[int], probabilities: list[float], threshold: float) -> BinaryClassificationMetrics:
        """以固定阈值统计 ROC-AUC、平均精度、FPR、recall 和混淆矩阵。"""
        BinaryMetrics._validate(labels, probabilities)
        labels_array = np.asarray(labels, dtype=int)
        scores = np.asarray(probabilities, dtype=float)
        predictions = scores >= threshold
        positive = labels_array == 1
        negative = ~positive
        true_positive = int(np.sum(predictions & positive))
        false_positive = int(np.sum(predictions & negative))
        true_negative = int(np.sum(~predictions & negative))
        false_negative = int(np.sum(~predictions & positive))
        return BinaryClassificationMetrics(
            auc_roc=BinaryMetrics._roc_auc(labels_array, scores),
            average_precision=BinaryMetrics._average_precision(labels_array, scores),
            threshold=float(threshold),
            false_positive_rate=false_positive / int(np.sum(negative)),
            recall=true_positive / int(np.sum(positive)),
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
        )

    @staticmethod
    def _validate(labels: list[int], probabilities: list[float]) -> None:
        """检查输入长度、标签空间和双类别存在性。"""
        if len(labels) != len(probabilities) or not labels:
            raise ValueError("标签与概率必须非空且长度一致")
        if set(labels) != {0, 1}:
            raise ValueError("指标计算需要同时包含良性和风险正类")
        if any(not 0.0 <= score <= 1.0 for score in probabilities):
            raise ValueError("概率分数必须在 [0, 1] 内")

    @staticmethod
    def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """基于秩和计算并正确处理同分的 ROC-AUC。"""
        ranks = BinaryMetrics._average_ranks(scores)
        positive_count = int(np.sum(labels == 1))
        negative_count = int(np.sum(labels == 0))
        return float((np.sum(ranks[labels == 1]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))

    @staticmethod
    def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
        """按分数从高到低聚合并列分数，计算标准平均精度。"""
        order = np.argsort(-scores, kind="stable")
        sorted_scores, sorted_labels = scores[order], labels[order]
        total_positive = int(np.sum(sorted_labels == 1))
        true_positive = 0
        average_precision = 0.0
        index = 0
        while index < len(sorted_labels):
            end = index + 1
            while end < len(sorted_labels) and sorted_scores[end] == sorted_scores[index]:
                end += 1
            group_positive = int(np.sum(sorted_labels[index:end] == 1))
            true_positive += group_positive
            average_precision += group_positive * (true_positive / end)
            index = end
        return average_precision / total_positive

    @staticmethod
    def _average_ranks(scores: np.ndarray) -> np.ndarray:
        """返回升序分数的平均名次，名次从 1 开始。"""
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=float)
        index = 0
        while index < len(scores):
            end = index + 1
            while end < len(scores) and scores[order[end]] == scores[order[index]]:
                end += 1
            ranks[order[index:end]] = (index + 1 + end) / 2
            index = end
        return ranks
