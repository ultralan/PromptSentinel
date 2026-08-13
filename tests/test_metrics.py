"""二分类评测指标的边界单元测试。"""

import unittest

from src.test.metrics import BinaryMetrics


class BinaryMetricsTest(unittest.TestCase):
    """验证阈值校准和排序指标的核心边界。"""

    def test_calibrated_threshold_respects_one_percent_false_positive_rate(self) -> None:
        """100 条良性 validation 在 1% 目标下最多放行 1 条。"""
        labels = [0] * 100 + [1] * 2
        probabilities = [index / 1000 for index in range(100)] + [0.8, 0.9]

        threshold = BinaryMetrics.calibrate_threshold(labels, probabilities, 0.01)
        metrics = BinaryMetrics.evaluate(labels, probabilities, threshold)

        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_positive_rate, 0.01)


    def test_average_precision_handles_tied_scores(self) -> None:
        """同分样本按组聚合，不把数组原始顺序当成模型排序能力。"""
        metrics = BinaryMetrics.evaluate([0, 1, 0, 1], [0.8, 0.8, 0.2, 0.2], threshold=0.5)

        self.assertEqual(metrics.auc_roc, 0.5)
        self.assertEqual(metrics.average_precision, 0.5)

    def test_calibrated_threshold_does_not_exceed_target_when_scores_tie(self) -> None:
        """最大分数并列超过预算时，宁可少报也不突破 FPR 目标。"""
        labels = [0] * 100 + [1]
        probabilities = [0.9, 0.9] + [0.1] * 98 + [0.95]

        threshold = BinaryMetrics.calibrate_threshold(labels, probabilities, 0.01)
        metrics = BinaryMetrics.evaluate(labels, probabilities, threshold)

        self.assertEqual(metrics.false_positive, 0)
        self.assertLessEqual(metrics.false_positive_rate, 0.01)
