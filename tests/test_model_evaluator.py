"""模型评测器的样本排序与逐样本分数对应测试。"""

import unittest
from unittest.mock import patch

import torch

from src.core.config_entity import EvaluationConfig
from src.core.data_entity import ClassificationRecord
from src.test.model_evaluator import ModelEvaluator


class _Tokenizer:
    """返回由文本内容编码的最小 tokenizer 替身。"""

    def __call__(self, texts, **kwargs):
        """将文本首字符映射为模型可读取的 input_ids。"""
        return {"input_ids": torch.tensor([[int(text)] for text in texts])}


class _Model:
    """按 input_ids 生成可验证风险概率的最小模型替身。"""

    def eval(self):
        """保持 Transformers 模型的 eval 接口。"""
        return self

    def to(self, device):
        """保持 Transformers 模型的设备迁移接口。"""
        return self

    def __call__(self, input_ids):
        """让风险类 logit 与输入值严格单调，便于检查回填顺序。"""
        return type("Output", (), {"logits": torch.cat([torch.zeros_like(input_ids), input_ids], dim=1).float()})()


class ModelEvaluatorTest(unittest.TestCase):
    """验证按长度分桶不会改变结果与原始记录的对应关系。"""

    def test_length_bucketing_restores_original_record_order(self) -> None:
        """乱序长度记录的风险概率仍按输入 records 顺序返回。"""
        config = EvaluationConfig.from_yaml(__import__("pathlib").Path("configs/test.yaml"), __import__("pathlib").Path.cwd())
        evaluator = ModelEvaluator(config, model_trainer=None)
        records = [
            ClassificationRecord("a", "1", 0, 30),
            ClassificationRecord("b", "2", 1, 10),
            ClassificationRecord("c", "3", 1, 20),
        ]

        with patch("src.test.model_evaluator.torch.inference_mode", torch.inference_mode):
            probabilities = evaluator.predict(_Tokenizer(), _Model(), "cpu", records, "test")

        self.assertEqual(len(probabilities), 3)
        self.assertLess(probabilities[0], probabilities[1])
        self.assertLess(probabilities[1], probabilities[2])
