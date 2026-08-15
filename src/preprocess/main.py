"""预处理模块启动入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config_entity import DataConfig
from src.core.execution_log import ExecutionLog
from src.preprocess.data_preprocessor import DataPreprocessor
from src.preprocess.prepared_dataset import PreparedDataset
from src.preprocess.promptshield_dataset import PromptShieldDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """解析预处理配置，装配对象并启动数据预处理任务。"""
    parser = argparse.ArgumentParser(description="下载并预处理 PromptShield")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = DataConfig.from_yaml(Path(args.config), PROJECT_ROOT)
    with ExecutionLog(config.paths.output_dir / "execution.log"):
        source = PromptShieldDataset(config.dataset, config.paths.raw_dir)
        prepared = PreparedDataset(config.paths.prepared_dir)
        DataPreprocessor(config, source, prepared).run()


if __name__ == "__main__":
    main()
