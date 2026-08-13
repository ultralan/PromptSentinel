"""原始数据到 prepared 标准数据集的业务编排。"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer

from src.core.config_entity import DataConfig
from src.core.data_entity import (
    ALL_SPLITS,
    TRAINING_SPLITS,
    ClassificationRecord,
    DatasetSplit,
    PreparedDatasetManifest,
    PreparedSplit,
    VALID_LABELS,
)
from src.preprocess.prepared_dataset import PreparedDataset
from src.preprocess.promptshield_dataset import PromptShieldDataset


class DataPreprocessor:
    """对齐数据格式、审计数据质量并生成训练专用 prepared 数据集。"""

    def __init__(self, config: DataConfig, source: PromptShieldDataset, prepared: PreparedDataset) -> None:
        """注入数据配置、原始数据集对象和 prepared 数据集对象。"""
        self._config = config
        self._source = source
        self._prepared = prepared

    def run(self) -> None:
        """执行下载、审计、隔离校验、长度门槛和 prepared 写入。"""
        raw_paths = self._source.download()
        raw_records = self._source.load(raw_paths)
        measured_records = self._measure_token_lengths(raw_records)
        audit = self._build_audit(raw_paths, raw_records, measured_records)
        audit_path = self._config.paths.reports_dir / "length_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(self._to_json(audit), encoding="utf-8")
        self._assert_split_isolation(audit)
        self._assert_overlength_limit(audit, audit_path)
        self._write_prepared(measured_records, audit_path)

    def _measure_token_lengths(
        self,
        splits: dict[DatasetSplit, list[ClassificationRecord]],
    ) -> dict[DatasetSplit, list[ClassificationRecord]]:
        """使用目标模型 tokenizer 为每条标准记录补充未截断 token 长度。"""
        model = self._config.model
        # 必须与训练器的 Protect AI DeBERTa tokenizer 实现一致，避免边界长度样本在预处理和训练时漂移。
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, revision=model.revision, use_fast=False)
        return {
            split: [
                replace(record, token_length=len(tokenizer(record.text, add_special_tokens=True, truncation=False)["input_ids"]))
                for record in records
            ]
            for split, records in splits.items()
        }

    def _build_audit(
        self,
        raw_paths: dict[DatasetSplit, Path],
        raw_records: dict[DatasetSplit, list[ClassificationRecord]],
        measured_records: dict[DatasetSplit, list[ClassificationRecord]],
    ) -> dict:
        """汇总原始文件校验和、长度分布与跨 split 重复检查结果。"""
        return {
            "dataset": self._config.dataset.__dict__,
            "model": self._config.model.__dict__,
            "files": {split.value: {"path": str(raw_paths[split]), "sha256": self._source.file_sha256(raw_paths[split])} for split in ALL_SPLITS},
            "splits": {split.value: self._summarize(records) for split, records in measured_records.items()},
            "cross_split_exact_text_duplicates": self._cross_split_duplicates(raw_records),
        }

    def _summarize(self, records: list[ClassificationRecord]) -> dict:
        """统计单个 split 的标签分布、token 分位数和超长比例。"""
        lengths = sorted(record.token_length for record in records if record.token_length is not None)
        max_length = self._config.model.max_length
        overlength_count = sum(length > max_length for length in lengths)
        return {
            "count": len(records),
            "label_counts": dict(Counter(record.label for record in records)),
            "token_length": {"p50": self._percentile(lengths, 0.50), "p90": self._percentile(lengths, 0.90), "p95": self._percentile(lengths, 0.95), "p99": self._percentile(lengths, 0.99), "max": lengths[-1]},
            "overlength_count": overlength_count,
            "overlength_ratio": overlength_count / len(records),
            "by_label": {str(label): {"count": sum(record.label == label for record in records), "overlength_count": sum(record.label == label and record.token_length is not None and record.token_length > max_length for record in records)} for label in sorted(VALID_LABELS)},
        }

    @staticmethod
    def _percentile(values: list[int], percent: float) -> int:
        """按最近索引返回已排序整数序列的指定分位点。"""
        if not values:
            raise ValueError("不能对空数据集计算长度审计")
        return values[round((len(values) - 1) * percent)]

    def _cross_split_duplicates(self, splits: dict[DatasetSplit, list[ClassificationRecord]]) -> dict[str, int]:
        """基于文本 SHA-256 计算任意两个官方 split 的精确重叠数。"""
        hashes = {split: {self._source.text_sha256(record.text) for record in records} for split, records in splits.items()}
        return {
            "train_validation": len(hashes[DatasetSplit.TRAIN] & hashes[DatasetSplit.VALIDATION]),
            "train_test": len(hashes[DatasetSplit.TRAIN] & hashes[DatasetSplit.TEST]),
            "validation_test": len(hashes[DatasetSplit.VALIDATION] & hashes[DatasetSplit.TEST]),
        }

    @staticmethod
    def _to_json(data: dict) -> str:
        """序列化审计数据为 UTF-8 友好的格式化 JSON 文本。"""
        import json

        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    @staticmethod
    def _assert_split_isolation(audit: dict) -> None:
        """发现任意官方 split 精确文本重叠时阻止生成训练数据。"""
        if any(audit["cross_split_exact_text_duplicates"].values()):
            raise RuntimeError("发现官方 split 间精确文本重叠，拒绝生成训练数据")

    def _assert_overlength_limit(self, audit: dict, audit_path: Path) -> None:
        """当训练或验证超长比例越过配置门槛时停止并保留审计报告。"""
        ratios = {split: audit["splits"][split.value]["overlength_ratio"] for split in TRAINING_SPLITS}
        if any(ratio > self._config.preparation.max_overlength_ratio for ratio in ratios.values()):
            raise RuntimeError(f"超长样本比例超过门槛，已写入 {audit_path}: {ratios}")

    def _write_prepared(self, records: dict[DatasetSplit, list[ClassificationRecord]], audit_path: Path) -> None:
        """过滤超长训练记录，保留完整独立测试集并写出 prepared 数据。"""
        selected = {
            split: [record for record in records[split] if record.token_length is not None and record.token_length <= self._config.model.max_length]
            for split in TRAINING_SPLITS
        }
        # 测试集不因长度被筛掉，推理时以同一 max_length 截断，避免静默改变官方评测分布。
        selected[DatasetSplit.TEST] = records[DatasetSplit.TEST]
        manifest = PreparedDatasetManifest(
            schema_version=1,
            task="text_classification",
            max_length=self._config.model.max_length,
            source_audit=str(audit_path),
            splits={split: PreparedSplit(path=f"{split.value}.jsonl", count=len(split_records)) for split, split_records in selected.items()},
        )
        self._prepared.write(selected, manifest)
