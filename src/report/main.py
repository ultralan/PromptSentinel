"""报告模块启动入口。"""

import argparse
import logging
from pathlib import Path

from src.core.execution_log import ExecutionLog
from src.report.training_curve_report import TrainingCurveReport


def main() -> None:
    """解析单次训练运行目录并生成损失曲线。"""
    parser = argparse.ArgumentParser(description="生成训练运行报告")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    with ExecutionLog(run_dir / "report.log"):
        output_path = TrainingCurveReport().render(run_dir)
        logging.getLogger(__name__).info("已生成训练损失曲线: %s", output_path)


if __name__ == "__main__":
    main()
