"""跨训练与评测共享的训练领域实体。"""

from __future__ import annotations

from enum import StrEnum


class TrainingMode(StrEnum):
    """支持的微调策略标识。"""
    LORA = "lora"
    FULL_FT = "full_ft"


class ModelImplementation(StrEnum):
    """当前工程支持的序列分类模型实现标识。"""

    PROMPT_GUARD_2 = "prompt_guard_2"
    PROTECT_AI_DEBERTA_V2 = "protect_ai_deberta_v2"
