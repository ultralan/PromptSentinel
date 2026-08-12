"""跨预处理、训练和评测共享的文本分类数据实体。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class DatasetSplit(StrEnum):
    """标准数据集划分名称。"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


TRAINING_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)
ALL_SPLITS = (*TRAINING_SPLITS, DatasetSplit.TEST)
VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class ClassificationRecord:
    """prepared 数据集中的单条文本二分类记录。"""
    id: str
    text: str
    label: int
    token_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSONL 的稳定字段映射。"""
        return asdict(self)


@dataclass(frozen=True)
class PreparedSplit:
    """manifest 中一个 prepared split 的文件位置和样本数量。"""
    path: str
    count: int


@dataclass(frozen=True)
class PreparedDatasetManifest:
    """text_classification/v1 prepared 数据集的顶层描述。"""
    schema_version: int
    task: str
    max_length: int
    source_audit: str
    splits: dict[DatasetSplit, PreparedSplit]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedDatasetManifest":
        """校验并反序列化 manifest JSON。"""
        if data.get("schema_version") != 1 or data.get("task") != "text_classification":
            raise ValueError("prepared manifest 不符合 text_classification/v1 契约")
        splits = {
            DatasetSplit(name): PreparedSplit(path=str(value["path"]), count=int(value["count"]))
            for name, value in data["splits"].items()
        }
        return cls(
            schema_version=1,
            task="text_classification",
            max_length=int(data["max_length"]),
            source_audit=str(data["source_audit"]),
            splits=splits,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 manifest JSON 的稳定字段映射。"""
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "max_length": self.max_length,
            "source_audit": self.source_audit,
            "splits": {split.value: asdict(metadata) for split, metadata in self.splits.items()},
        }
