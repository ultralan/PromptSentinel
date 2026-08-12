"""PromptShield 原始数据集对象。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.core.config_entity import DatasetConfig
from src.core.data_entity import ALL_SPLITS, ClassificationRecord, DatasetSplit, VALID_LABELS


class PromptShieldDataset:
    """负责固定 revision 的下载及 PromptShield 原始字段映射。"""

    def __init__(self, config: DatasetConfig, raw_dir: Path) -> None:
        """绑定数据集版本配置和本地原始数据缓存目录。"""
        self._config = config
        self._raw_dir = raw_dir

    def download(self) -> dict[DatasetSplit, Path]:
        """下载固定 revision 的所有官方 split 并返回本地路径。"""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        return {
            split: Path(
                hf_hub_download(
                    repo_id=self._config.repo_id,
                    repo_type="dataset",
                    revision=self._config.revision,
                    filename=self._config.files[split.value],
                    local_dir=self._raw_dir,
                )
            )
            for split in ALL_SPLITS
        }

    def load(self, paths: dict[DatasetSplit, Path]) -> dict[DatasetSplit, list[ClassificationRecord]]:
        """读取原始 split 并映射为跨模块通用分类记录。"""
        return {split: self._load_split(path, split) for split, path in paths.items()}

    @staticmethod
    def file_sha256(path: Path) -> str:
        """计算原始文件 SHA-256，用于数据版本审计。"""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def text_sha256(text: str) -> str:
        """计算文本 SHA-256，用于跨 split 精确重复检查。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_split(self, path: Path, split: DatasetSplit) -> list[ClassificationRecord]:
        """解析 JSON 数组或 JSONL 原始文件，并完成 PromptShield 字段映射。"""
        raw_text = path.read_text(encoding="utf-8")
        rows = json.loads(raw_text) if raw_text.lstrip().startswith("[") else [
            json.loads(line) for line in raw_text.splitlines() if line.strip()
        ]
        records: list[ClassificationRecord] = []
        for index, row in enumerate(rows):
            prompt, label = row.get("prompt"), row.get("label")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"PromptShield {split.value}[{index}] 缺少非空 prompt")
            if label not in VALID_LABELS:
                raise ValueError(f"PromptShield {split.value}[{index}] label 必须是 0 或 1")
            records.append(ClassificationRecord(str(row.get("id") or f"{split.value}-{index:06d}"), prompt, int(label)))
        return records
