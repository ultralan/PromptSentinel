"""jailbreak-classification 原始能力保持数据源。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from datasets import load_dataset

from src.core.config_entity import DataConfig
from src.core.data_entity import ClassificationRecord


class JailbreakClassificationDataset:
    """固定 revision 下载并映射公开 jailbreak 二分类数据集。"""

    def __init__(self, config: DataConfig) -> None:
        """绑定数据源版本与本地审计目录。"""
        self._config = config

    def load(self) -> list[ClassificationRecord]:
        """合并官方 train/test，按原始二元 type 映射标准记录并精确去重。"""
        dataset = load_dataset(
            self._config.dataset.repo_id,
            revision=self._config.dataset.revision,
            cache_dir=str(self._config.paths.raw_dir),
        )
        records: list[ClassificationRecord] = []
        seen_text_hashes: set[str] = set()
        for split_name in ("train", "test"):
            for index, row in enumerate(dataset[split_name]):
                prompt, kind = row.get("prompt"), row.get("type")
                if not isinstance(prompt, str) or not prompt.strip() or kind not in {"benign", "jailbreak"}:
                    raise ValueError(f"{split_name}[{index}] 不符合 jailbreak-classification 二元标签契约")
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if digest in seen_text_hashes:
                    continue
                seen_text_hashes.add(digest)
                records.append(
                    ClassificationRecord(
                        id=f"jailbreak-classification-{split_name}-{index:06d}",
                        text=prompt,
                        label=int(kind == "jailbreak"),
                    )
                )
        if {record.label for record in records} != {0, 1}:
            raise ValueError("能力保持集必须同时包含 benign 与 jailbreak 标签")
        return records
