"""单次训练运行的目录和复现产物对象。"""

from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from src.core.config_entity import DataConfig, TrainingConfig


class TrainingRun:
    """管理单次训练的 runs 目录、元数据和 Trainer 状态。"""

    def __init__(self, config: TrainingConfig) -> None:
        """根据运行配置生成唯一的 UTC 时间戳产物目录。"""
        self._config = config
        if config.resume_from_checkpoint is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.root_dir = config.run.output_root / f"{timestamp}-{config.run.run_name}"
        else:
            self.root_dir = config.resume_from_checkpoint.parent.parent

    def create_root_dir(self) -> None:
        """在模型加载前创建唯一运行目录，使首行日志也归属本次运行。"""
        if self._config.resume_from_checkpoint is None:
            self.root_dir.mkdir(parents=True, exist_ok=False)
            return
        checkpoint = self._config.resume_from_checkpoint
        if not checkpoint.is_dir() or checkpoint.parent != self.checkpoint_dir:
            raise ValueError(f"恢复 checkpoint 不属于合法运行目录: {checkpoint}")
        if not self.root_dir.is_dir():
            raise RuntimeError(f"恢复运行目录不存在: {self.root_dir}")

    def initialize(self, data_config: DataConfig, parameter_summary: dict[str, int], dataset_sizes: dict[str, int], device: str) -> None:
        """持久化配置、环境、参数量和样本数。"""
        if not self.root_dir.is_dir():
            raise RuntimeError(f"训练运行目录尚未创建: {self.root_dir}")
        self._write_json(asdict(self._config), self.root_dir / "config.json")
        self._write_json(asdict(data_config), self.root_dir / "data_config.json")
        self._write_json(
            {
                "parameters": parameter_summary,
                "datasets": dataset_sizes,
                "device": device,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            self.root_dir / "run_metadata.json",
        )
        if self._config.resume_from_checkpoint is not None:
            self._write_json(
                {"resumed_from_checkpoint": str(self._config.resume_from_checkpoint)},
                self.root_dir / "resume.json",
            )

    @property
    def checkpoint_dir(self) -> Path:
        """返回 Trainer checkpoint 保存目录。"""
        return self.root_dir / "checkpoints"

    @property
    def model_dir(self) -> Path:
        """返回最终模型产物保存目录。"""
        return self.root_dir / "model"

    @property
    def log_path(self) -> Path:
        """返回与 checkpoint、配置快照同目录的训练日志路径。"""
        return self.root_dir / "train.log"

    @property
    def is_resuming(self) -> bool:
        """标识当前运行是否从已有 checkpoint 恢复。"""
        return self._config.resume_from_checkpoint is not None

    def save_state(self, trainer: Any) -> None:
        """保存 Transformers Trainer 状态，支持中断后追溯。"""
        trainer.save_state()
        trainer.state.save_to_json(self.root_dir / "trainer_state.json")

    @staticmethod
    def set_seed(seed: int) -> None:
        """统一设置 Python、NumPy、PyTorch 的训练随机种子。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _write_json(data: Any, path: Path) -> None:
        """以 UTF-8 格式化 JSON 写入单次运行元数据文件。"""
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
