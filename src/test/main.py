"""测试模块启动入口。"""

import argparse
from pathlib import Path

from src.core.config_entity import DataConfig, EvaluationConfig
from src.test.test_job import TestJob
from src.test.evaluation_run import EvaluationRun
from src.test.model_evaluator import ModelEvaluator
from src.preprocess.prepared_dataset import PreparedDataset
from src.train.model_trainer.model_trainer_factory import ModelTrainerFactory
from src.core.execution_log import ExecutionLog


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """解析评测配置，装配对象并启动独立测试集评测。"""
    parser = argparse.ArgumentParser(description="评测提示注入分类模型")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = EvaluationConfig.from_yaml(Path(args.config), PROJECT_ROOT)
    data_config = DataConfig.from_yaml(config.data_config_path, PROJECT_ROOT)
    evaluation_run = EvaluationRun(config)
    evaluation_run.create_root_dir()
    with ExecutionLog(evaluation_run.log_path):
        model_trainer = ModelTrainerFactory.create(config.model)
        TestJob(
            config,
            data_config,
            PreparedDataset(data_config.paths.prepared_dir),
            ModelEvaluator(config, model_trainer),
            evaluation_run,
            model_trainer,
        ).run()
if __name__ == "__main__":
    main()
