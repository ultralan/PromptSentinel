"""跨训练与评测共享的训练领域实体。"""

from __future__ import annotations

from enum import StrEnum


class TrainingMode(StrEnum):
    """支持的微调策略标识。"""
    LORA = "lora"
    FULL_FT = "full_ft"


class MixedPrecision(StrEnum):
    """训练时模型前向和反向计算使用的浮点精度。"""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ModelImplementation(StrEnum):
    """当前工程支持的序列分类模型实现标识。"""

    PROMPT_GUARD_2 = "prompt_guard_2"
    PROTECT_AI_DEBERTA_V2 = "protect_ai_deberta_v2"


class ModelArtifactType(StrEnum):
    """评测候选模型的产物形态。"""

    BASE = "base"
    LORA_ADAPTER = "lora_adapter"
    FULL_FINE_TUNED = "full_fine_tuned"
