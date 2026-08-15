"""YAML 配置对应的跨模块 dataclass 实体。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.training_entity import ModelArtifactType, ModelImplementation, TrainingMode


def _load_mapping(path: Path) -> dict[str, Any]:
    """读取 YAML 顶层映射，拒绝空文件或非映射配置。"""
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {path}")
    return value


def _resolve(path: str, project_root: Path) -> Path:
    """将配置中的相对路径解析到项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


@dataclass(frozen=True)
class ModelConfig:
    """模型标识、版本和最大输入长度配置。"""
    implementation: ModelImplementation
    name_or_path: str
    revision: str
    max_length: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelConfig":
        """从 YAML 模型节点构建强类型配置。"""
        return cls(
            implementation=ModelImplementation(str(data["implementation"])),
            name_or_path=str(data["name_or_path"]),
            revision=str(data["revision"]),
            max_length=int(data["max_length"]),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """远程数据集标识、固定 revision 与 split 文件映射。"""
    implementation: str
    repo_id: str
    revision: str
    files: dict[str, str]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetConfig":
        """从 YAML 数据集节点构建强类型配置。"""
        return cls(
            implementation=str(data["implementation"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            files={str(key): str(value) for key, value in data["files"].items()},
        )


@dataclass(frozen=True)
class DataPaths:
    """原始数据、prepared 数据与预处理运行产物的本地目录。"""
    raw_dir: Path
    prepared_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class PreparationConfig:
    """数据预处理的长度过滤与阻断策略。"""
    max_overlength_ratio: float
    overlength_policy: str


@dataclass(frozen=True)
class DataConfig:
    """预处理模块运行所需的完整数据配置。"""
    dataset: DatasetConfig
    model: ModelConfig
    paths: DataPaths
    preparation: PreparationConfig

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "DataConfig":
        """加载数据 YAML，并将所有文件路径解析为绝对路径。"""
        data = _load_mapping(path)
        paths = data["paths"]
        preparation = data["preparation"]
        return cls(
            dataset=DatasetConfig.from_mapping(data["dataset"]),
            model=ModelConfig.from_mapping(data["model"]),
            paths=DataPaths(
                raw_dir=_resolve(str(paths["raw_dir"]), project_root),
                prepared_dir=_resolve(str(paths["prepared_dir"]), project_root),
                output_dir=_resolve(str(paths["output_dir"]), project_root),
            ),
            preparation=PreparationConfig(
                max_overlength_ratio=float(preparation["max_overlength_ratio"]),
                overlength_policy=str(preparation["overlength_policy"]),
            ),
        )


@dataclass(frozen=True)
class LoraConfig:
    """LoRA 低秩适配器及分类头保存配置。"""
    r: int
    alpha: int
    dropout: float
    target_modules: str | list[str]
    modules_to_save: list[str]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LoraConfig":
        """从 YAML LoRA 节点构建强类型配置。"""
        target_modules = data["target_modules"]
        if not isinstance(target_modules, (str, list)):
            raise ValueError("lora.target_modules 必须是字符串或字符串列表")
        return cls(
            r=int(data["r"]),
            alpha=int(data["alpha"]),
            dropout=float(data["dropout"]),
            target_modules=target_modules,
            modules_to_save=[str(item) for item in data["modules_to_save"]],
        )


@dataclass(frozen=True)
class RunConfig:
    """单次训练运行的模式、随机种子和产物位置。"""
    mode: TrainingMode
    seed: int
    output_root: Path
    run_name: str


@dataclass(frozen=True)
class TrainerConfig:
    """Transformers Trainer 的训练与早停超参数。"""
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_total_limit: int
    early_stopping_patience: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TrainerConfig":
        """从 YAML training 节点构建强类型配置。"""
        return cls(
            num_train_epochs=float(data["num_train_epochs"]),
            per_device_train_batch_size=int(data["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(data["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
            learning_rate=float(data["learning_rate"]),
            weight_decay=float(data["weight_decay"]),
            warmup_ratio=float(data["warmup_ratio"]),
            logging_steps=int(data["logging_steps"]),
            save_total_limit=int(data["save_total_limit"]),
            early_stopping_patience=int(data["early_stopping_patience"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """训练模块运行所需的模型、策略、训练和运行配置。"""
    data_config_path: Path
    run: RunConfig
    model: ModelConfig
    trainer: TrainerConfig
    lora: LoraConfig | None
    resume_from_checkpoint: Path | None

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "TrainingConfig":
        """加载训练 YAML，并仅在 LoRA 模式解析 LoRA 配置。"""
        data = _load_mapping(path)
        run = data["run"]
        mode = TrainingMode(str(run["mode"]))
        lora = LoraConfig.from_mapping(data["lora"]) if mode is TrainingMode.LORA else None
        raw_resume_path = data.get("resume_from_checkpoint")
        if raw_resume_path is not None and (not isinstance(raw_resume_path, str) or not raw_resume_path.strip()):
            raise ValueError("resume_from_checkpoint 必须是非空路径字符串")
        return cls(
            data_config_path=_resolve(str(data["data_config"]), project_root),
            run=RunConfig(
                mode=mode,
                seed=int(run["seed"]),
                output_root=_resolve(str(run["output_root"]), project_root),
                run_name=str(run["run_name"]),
            ),
            model=ModelConfig.from_mapping(data["model"]),
            trainer=TrainerConfig.from_mapping(data["training"]),
            lora=lora,
            resume_from_checkpoint=(
                _resolve(raw_resume_path, project_root) if raw_resume_path is not None else None
            ),
        )


@dataclass(frozen=True)
class EvaluationRunConfig:
    """单次离线评测的结果目录配置。"""

    output_root: Path
    run_name: str


@dataclass(frozen=True)
class EvaluationSettings:
    """所有候选模型共享的推理批大小与阈值校准目标。"""

    per_device_batch_size: int
    target_false_positive_rate: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvaluationSettings":
        """从 YAML evaluation 节点构建评测运行参数。"""
        batch_size = int(data["per_device_batch_size"])
        target_fpr = float(data["target_false_positive_rate"])
        if batch_size < 1:
            raise ValueError("evaluation.per_device_batch_size 必须大于 0")
        if not 0 < target_fpr < 1:
            raise ValueError("evaluation.target_false_positive_rate 必须在 (0, 1) 内")
        return cls(per_device_batch_size=batch_size, target_false_positive_rate=target_fpr)


@dataclass(frozen=True)
class EvaluationCandidateConfig:
    """一个待比较的基座、LoRA adapter 或全量微调模型。"""

    name: str
    artifact_type: ModelArtifactType
    path: Path | None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], project_root: Path) -> "EvaluationCandidateConfig":
        """从 YAML candidate 节点构建并校验候选模型定义。"""
        artifact_type = ModelArtifactType(str(data["artifact_type"]))
        raw_path = data.get("path")
        if artifact_type is ModelArtifactType.BASE:
            if raw_path is not None:
                raise ValueError("base 候选不应提供 path")
            path = None
        else:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"{artifact_type.value} 候选必须提供 path")
            path = _resolve(raw_path, project_root)
        return cls(name=str(data["name"]), artifact_type=artifact_type, path=path)


@dataclass(frozen=True)
class EvaluationConfig:
    """测试模块运行所需的固定数据版本、模型身份与候选清单。"""

    data_config_path: Path
    model: ModelConfig
    run: EvaluationRunConfig
    evaluation: EvaluationSettings
    candidates: list[EvaluationCandidateConfig]

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "EvaluationConfig":
        """加载评测 YAML，并拒绝空候选或重复候选名称。"""
        data = _load_mapping(path)
        run = data["run"]
        candidates = [EvaluationCandidateConfig.from_mapping(item, project_root) for item in data["candidates"]]
        names = [candidate.name for candidate in candidates]
        if not candidates:
            raise ValueError("评测至少需要一个候选模型")
        if len(set(names)) != len(names):
            raise ValueError("评测候选 name 必须唯一")
        return cls(
            data_config_path=_resolve(str(data["data_config"]), project_root),
            model=ModelConfig.from_mapping(data["model"]),
            run=EvaluationRunConfig(
                output_root=_resolve(str(run["output_root"]), project_root),
                run_name=str(run["run_name"]),
            ),
            evaluation=EvaluationSettings.from_mapping(data["evaluation"]),
            candidates=candidates,
        )
