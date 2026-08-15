"""基于数据、评估口径和模型内容指纹复用候选评估结果。"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.core.config_entity import EvaluationCandidateConfig, EvaluationConfig
from src.core.data_entity import ClassificationRecord


EVALUATION_PROTOCOL_VERSION = "text-classification/v1"


@dataclass(frozen=True)
class CachedEvaluation:
    """一个已完成候选的指标、预测文件和来源运行。"""

    report: dict
    predictions_path: Path
    source_run: Path


class EvaluationCache:
    """按内容指纹查找并恢复可复用的候选评估产物。"""

    def __init__(
        self,
        config: EvaluationConfig,
        validation_records: list[ClassificationRecord],
        test_records: list[ClassificationRecord],
        current_run_dir: Path,
    ) -> None:
        """绑定当前评估协议、标准数据内容与当前运行目录。"""
        self._config = config
        self._current_run_dir = current_run_dir
        self._data_fingerprint = self._fingerprint_records(validation_records, test_records)

    def candidate_key(self, candidate: EvaluationCandidateConfig) -> str:
        """生成不依赖候选名称和文件路径的稳定缓存键。"""
        payload = {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "data_fingerprint": self._data_fingerprint,
            "model": {
                "implementation": self._config.model.implementation.value,
                "name_or_path": self._config.model.name_or_path,
                "revision": self._config.model.revision,
                "max_length": self._config.model.max_length,
            },
            "target_false_positive_rate": self._config.evaluation.target_false_positive_rate,
            "candidate": {
                "artifact_type": candidate.artifact_type.value,
                "artifact_fingerprint": self._fingerprint_candidate(candidate),
            },
        }
        return self._sha256_json(payload)

    def find(self, cache_key: str) -> CachedEvaluation | None:
        """从历史评估运行中查找缓存键完全一致的候选产物。"""
        output_root = self._config.run.output_root
        if not output_root.is_dir():
            return None
        for run_dir in sorted(output_root.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir == self._current_run_dir:
                continue
            metrics_dir = run_dir / "candidate_metrics"
            if not metrics_dir.is_dir():
                continue
            for metrics_path in metrics_dir.glob("*.json"):
                report = json.loads(metrics_path.read_text(encoding="utf-8"))
                if report.get("cache_key") != cache_key:
                    continue
                predictions_path = run_dir / "predictions" / f"{report['name']}.jsonl"
                if predictions_path.is_file():
                    return CachedEvaluation(report, predictions_path, run_dir)
        return None

    def restore(
        self,
        cached: CachedEvaluation,
        candidate: EvaluationCandidateConfig,
        destination_predictions: Path,
    ) -> dict:
        """复制缓存预测，并将来源指标改写为当前候选的自包含报告。"""
        destination_predictions.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached.predictions_path, destination_predictions)
        report = dict(cached.report)
        report.update(
            {
                "name": candidate.name,
                "artifact_type": candidate.artifact_type.value,
                "path": str(candidate.path) if candidate.path is not None else None,
                "cache": {"hit": True, "source_run": str(cached.source_run)},
            }
        )
        logging.getLogger(__name__).info(
            "%s 命中评估缓存，跳过模型加载和推理: source=%s",
            candidate.name,
            cached.source_run,
        )
        return report

    def backfill_run(
        self,
        run_dir: Path,
        candidates: list[EvaluationCandidateConfig],
    ) -> None:
        """为升级前已完成的评估运行补写候选级缓存元数据。"""
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"评估运行缺少 metrics.json: {run_dir}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        reports_by_name = {report["name"]: report for report in metrics.get("candidates", [])}
        updated_reports = []
        candidate_metrics_dir = run_dir / "candidate_metrics"
        candidate_metrics_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            report = reports_by_name.get(candidate.name)
            predictions_path = run_dir / "predictions" / f"{candidate.name}.jsonl"
            if report is None or not predictions_path.is_file():
                raise FileNotFoundError(f"评估运行缺少 {candidate.name} 的完整指标或预测")
            updated = self.mark_computed(
                {
                    **report,
                    "artifact_type": candidate.artifact_type.value,
                    "path": str(candidate.path) if candidate.path is not None else None,
                },
                self.candidate_key(candidate),
            )
            (candidate_metrics_dir / f"{candidate.name}.json").write_text(
                json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            updated_reports.append(updated)
        metrics["candidates"] = updated_reports
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def mark_computed(report: dict, cache_key: str) -> dict:
        """为本次实际推理得到的候选报告补充缓存元数据。"""
        return {
            **report,
            "cache_key": cache_key,
            "cache": {"hit": False, "source_run": None},
        }

    @staticmethod
    def _fingerprint_records(
        validation_records: list[ClassificationRecord],
        test_records: list[ClassificationRecord],
    ) -> str:
        """按记录顺序和完整字段计算 validation、test 联合内容指纹。"""
        digest = hashlib.sha256()
        for split_name, records in (("validation", validation_records), ("test", test_records)):
            digest.update(split_name.encode("ascii"))
            digest.update(b"\0")
            for record in records:
                digest.update(
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
        return digest.hexdigest()

    def _fingerprint_candidate(self, candidate: EvaluationCandidateConfig) -> str:
        """以基座固定 revision 或本地训练权重内容标识候选模型。"""
        if candidate.path is None:
            return self._sha256_json(
                {
                    "name_or_path": self._config.model.name_or_path,
                    "revision": self._config.model.revision,
                }
            )
        if not candidate.path.is_dir():
            raise FileNotFoundError(f"候选模型产物不存在: {candidate.path}")
        identity_files = sorted(
            path
            for path in candidate.path.iterdir()
            if path.is_file()
            and (
                path.suffix == ".safetensors"
                or path.name in {"adapter_config.json", "config.json", "pytorch_model.bin"}
            )
        )
        if not identity_files:
            raise FileNotFoundError(f"候选模型目录中没有可识别的权重或配置: {candidate.path}")
        digest = hashlib.sha256()
        for path in identity_files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_json(data: dict) -> str:
        """对规范化 JSON 映射计算 SHA-256。"""
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
