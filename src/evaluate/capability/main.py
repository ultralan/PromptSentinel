"""原始能力保持评测启动入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config_entity import DataConfig, EvaluationConfig
from src.core.data_entity import DatasetSplit, PreparedDatasetManifest, PreparedSplit
from src.core.execution_log import ExecutionLog
from src.evaluate.capability_retention_job import CapabilityRetentionJob
from src.evaluate.evaluation_run import EvaluationRun
from src.evaluate.model_evaluator import ModelEvaluator
from src.preprocess.jailbreak_classification_dataset import JailbreakClassificationDataset
from src.preprocess.prepared_dataset import PreparedDataset
from src.train.model_trainer.model_trainer_factory import ModelTrainerFactory


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """下载固定能力保持集，冻结快照并评估 Base、LoRA 42 与 Full FT 42。"""
    parser = argparse.ArgumentParser(description="评估原始能力保持情况")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = EvaluationConfig.from_yaml(Path(args.config), PROJECT_ROOT)
    data_config = DataConfig.from_yaml(config.data_config_path, PROJECT_ROOT)
    records = JailbreakClassificationDataset(data_config).load()
    prepared = PreparedDataset(data_config.paths.prepared_dir)
    prepared.write(
        {DatasetSplit.TEST: records},
        PreparedDatasetManifest(
            schema_version=1,
            task="text_classification",
            max_length=config.model.max_length,
            source_audit=f"{data_config.dataset.repo_id}@{data_config.dataset.revision}",
            splits={DatasetSplit.TEST: PreparedSplit(path="test.jsonl", count=len(records))},
        ),
    )
    run = EvaluationRun(config)
    run.create_root_dir()
    with ExecutionLog(run.log_path):
        trainer = ModelTrainerFactory.create(config.model)
        run.initialize(data_config, {"test": len(records)}, trainer.select_device())
        CapabilityRetentionJob(config, ModelEvaluator(config, trainer), run).run(records)


if __name__ == "__main__":
    main()
