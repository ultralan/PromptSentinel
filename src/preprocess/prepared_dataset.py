"""text_classification/v1 prepared 数据集对象。"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.data_entity import ClassificationRecord, DatasetSplit, PreparedDatasetManifest, TRAINING_SPLITS, VALID_LABELS


class PreparedDataset:
    """管理标准化 manifest 和分类记录，屏蔽本地 JSONL 细节。"""

    def __init__(self, root_dir: Path) -> None:
        """绑定 prepared 数据集根目录。"""
        self._root_dir = root_dir

    @property
    def manifest_path(self) -> Path:
        """返回该 prepared 数据集的 manifest 固定位置。"""
        return self._root_dir / "manifest.json"

    def write(self, records: dict[DatasetSplit, list[ClassificationRecord]], manifest: PreparedDatasetManifest) -> None:
        """写入标准 JSONL split 与对应 manifest。"""
        self._root_dir.mkdir(parents=True, exist_ok=True)
        for split, split_records in records.items():
            path = self._root_dir / f"{split.value}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in split_records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self.manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load_manifest(self) -> PreparedDatasetManifest:
        """读取并校验 text_classification/v1 manifest。"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"缺少 prepared manifest: {self.manifest_path}")
        return PreparedDatasetManifest.from_dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))

    def load_training_split(self, split: DatasetSplit) -> list[ClassificationRecord]:
        """读取 train 或 validation 标准记录，拒绝读取 test。"""
        if split not in TRAINING_SPLITS:
            raise ValueError(f"训练模块不能读取 {split.value} split")
        return self._load_split(split)

    def load_test_split(self) -> list[ClassificationRecord]:
        """读取独立 test 标准记录，仅供评测模块消费。"""
        return self._load_split(DatasetSplit.TEST)

    def _load_split(self, split: DatasetSplit) -> list[ClassificationRecord]:
        """读取并校验 manifest 中声明的任一标准 split。"""
        manifest = self.load_manifest()
        metadata = manifest.splits.get(split)
        if metadata is None:
            raise ValueError(f"prepared manifest 不包含 {split.value}")
        path = self._root_dir / metadata.path
        records = []
        # JSONL 只允许物理换行分隔记录；str.splitlines() 会误拆文本中的 Unicode 行分隔符。
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{split.value} JSONL 第 {index + 1} 行格式错误") from error
                records.append(self._parse_record(row, split, index))
        if len(records) != metadata.count:
            raise ValueError(f"{split.value} 行数与 manifest 不一致")
        return records

    @staticmethod
    def _parse_record(row: dict, split: DatasetSplit, index: int) -> ClassificationRecord:
        """校验单行 JSONL 并重建 ClassificationRecord 实体。"""
        text, label = row.get("text"), row.get("label")
        if not isinstance(text, str) or not text.strip() or label not in VALID_LABELS:
            raise ValueError(f"{split.value}[{index}] 不符合 ClassificationRecord 契约")
        return ClassificationRecord(
            id=str(row.get("id") or f"{split.value}-{index:06d}"),
            text=text,
            label=int(label),
            token_length=int(row["token_length"]) if row.get("token_length") is not None else None,
        )
