"""训练模块启动入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config_entity import DataConfig, TrainingConfig
from src.core.execution_log import ExecutionLog
from src.preprocess.prepared_dataset import PreparedDataset
from src.train.fine_tuning.fine_tuning_strategy import FineTuningStrategyFactory
from src.train.model_trainer.model_trainer_factory import ModelTrainerFactory
from src.train.training_job import TrainingJob
from src.train.training_run import TrainingRun


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """解析训练配置，装配依赖并启动训练任务。"""
    parser = argparse.ArgumentParser(description="训练提示注入分类模型")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with ExecutionLog(PROJECT_ROOT, "train"):
        training_config = TrainingConfig.from_yaml(Path(args.config), PROJECT_ROOT)
        data_config = DataConfig.from_yaml(training_config.data_config_path, PROJECT_ROOT)
        job = TrainingJob(
            training_config,
            data_config,
            PreparedDataset(data_config.paths.prepared_dir),
            ModelTrainerFactory.create(training_config.model, training_config.trainer),
            FineTuningStrategyFactory.create(training_config),
            TrainingRun(training_config),
        )
        job.run()


if __name__ == "__main__":
    main()
