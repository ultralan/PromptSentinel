"""LoRA 与全量微调的模型改造和保存策略。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model

from src.core.config_entity import LoraConfig, TrainingConfig
from src.core.training_entity import TrainingMode


class FineTuningStrategy(ABC):
    """仅负责模型参数改造与产物保存，不感知数据或运行目录。"""

    @abstractmethod
    def prepare_model(self, base_model: Any) -> Any:
        """将基础分类模型转换为本策略可训练的模型。"""
        raise NotImplementedError

    @abstractmethod
    def save_model(self, trainer: Any, tokenizer: Any, output_dir: Path) -> None:
        """按策略语义保存模型产物与 tokenizer。"""
        raise NotImplementedError


class LoraFineTuningStrategy(FineTuningStrategy):
    """仅训练 LoRA 适配器和保留的分类头。"""

    def __init__(self, config: LoraConfig) -> None:
        """绑定低秩维度、缩放、dropout 与目标模块配置。"""
        self._config = config

    def prepare_model(self, base_model: Any) -> Any:
        """以 SEQ_CLS 任务类型向基础模型注入 LoRA。"""
        return get_peft_model(
            base_model,
            PeftLoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self._config.r,
                lora_alpha=self._config.alpha,
                lora_dropout=self._config.dropout,
                target_modules=self._config.target_modules,
                modules_to_save=self._config.modules_to_save,
            ),
        )

    def save_model(self, trainer: Any, tokenizer: Any, output_dir: Path) -> None:
        """保存 LoRA adapter、分类头和 tokenizer。"""
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


class FullFineTuningStrategy(FineTuningStrategy):
    """保持全部基础模型参数可训练的全量微调策略。"""
    def prepare_model(self, base_model: Any) -> Any:
        """不改造基础模型，直接用于全量微调。"""
        return base_model

    def save_model(self, trainer: Any, tokenizer: Any, output_dir: Path) -> None:
        """保存完整分类模型和 tokenizer。"""
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


class FineTuningStrategyFactory:
    """训练模式到具体微调策略的唯一映射点。"""
    @staticmethod
    def create(config: TrainingConfig) -> FineTuningStrategy:
        """根据强类型 TrainingMode 构造对应策略。"""
        if config.run.mode is TrainingMode.LORA:
            if config.lora is None:
                raise ValueError("LoRA 模式缺少 lora 配置")
            return LoraFineTuningStrategy(config.lora)
        if config.run.mode is TrainingMode.FULL_FT:
            return FullFineTuningStrategy()
        raise ValueError(f"不支持的训练模式: {config.run.mode}")
