"""跨训练与评测共享的训练领域实体。"""

from __future__ import annotations

from enum import StrEnum


class TrainingMode(StrEnum):
    """支持的微调策略标识。"""
    LORA = "lora"
    FULL_FT = "full_ft"
