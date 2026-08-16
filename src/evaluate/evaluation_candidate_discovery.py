"""从训练产物目录发现已完成的评估候选模型。"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config_entity import EvaluationCandidateConfig
from src.core.training_entity import ModelArtifactType, TrainingMode


class EvaluationCandidateDiscovery:
    """按训练模式和随机种子发现最新的完整模型产物。"""

    def __init__(self, training_output_root: Path) -> None:
        """绑定训练运行的统一输出根目录。"""
        self._training_output_root = training_output_root

    def discover_full_ft(self, seeds: list[int]) -> list[EvaluationCandidateConfig]:
        """为每个 seed 返回最新且状态为 completed 的 Full FT 模型。"""
        if len(set(seeds)) != len(seeds):
            raise ValueError("Full FT 随机种子不能重复")
        return [self._discover_full_ft_seed(seed) for seed in seeds]

    def _discover_full_ft_seed(self, seed: int) -> EvaluationCandidateConfig:
        """发现一个随机种子对应的最新完整训练运行。"""
        candidates = []
        if self._training_output_root.is_dir():
            for run_dir in self._training_output_root.iterdir():
                metadata_path = run_dir / "run_metadata.json"
                model_dir = run_dir / "model"
                if not metadata_path.is_file() or not model_dir.is_dir():
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("status") == "completed"
                    and metadata.get("mode") == TrainingMode.FULL_FT.value
                    and metadata.get("seed") == seed
                ):
                    candidates.append((run_dir.name, model_dir))
        if not candidates:
            raise FileNotFoundError(f"未找到已完成的 Full FT seed {seed} 训练产物")
        _, model_dir = max(candidates)
        return EvaluationCandidateConfig(
            name=f"full_ft_seed_{seed}",
            artifact_type=ModelArtifactType.FULL_FINE_TUNED,
            path=model_dir,
        )
