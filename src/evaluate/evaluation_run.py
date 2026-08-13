"""单次评测运行的目录、配置快照和结果写入对象。"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

from src.core.config_entity import DataConfig, EvaluationConfig


class EvaluationRun:
    """管理一次可复现评测的 runs 目录和 JSON 产物。"""

    def __init__(self, config: EvaluationConfig) -> None:
        """根据 UTC 时间戳和评测名称生成唯一结果目录。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.root_dir = config.run.output_root / f"{timestamp}-{config.run.run_name}"
        self._config = config

    @property
    def log_path(self) -> Path:
        """返回与评测产物同目录的执行日志路径。"""
        return self.root_dir / "test.log"

    def create_root_dir(self) -> None:
        """在加载模型前创建唯一评测目录，使首行日志归属本次运行。"""
        self.root_dir.mkdir(parents=True, exist_ok=False)

    def initialize(self, data_config: DataConfig, dataset_sizes: dict[str, int], device: str) -> None:
        """在目录已创建后固化数据、模型、环境和样本数快照。"""
        if not self.root_dir.is_dir():
            raise RuntimeError(f"评测运行目录尚未创建: {self.root_dir}")
        self.write_json(asdict(self._config), "config.json")
        self.write_json(asdict(data_config), "data_config.json")
        self.write_json(
            {
                "datasets": dataset_sizes,
                "device": device,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "run_metadata.json",
        )

    def write_json(self, data: Any, relative_path: str) -> Path:
        """将结果对象以 UTF-8 格式化 JSON 写入当前运行目录。"""
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def predictions_path(self, candidate_name: str) -> Path:
        """返回一个候选模型固定的逐样本预测 JSONL 路径。"""
        return self.root_dir / "predictions" / f"{candidate_name}.jsonl"
