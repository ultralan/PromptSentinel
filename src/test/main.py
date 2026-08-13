"""测试模块启动入口。"""

from pathlib import Path

from src.core.execution_log import ExecutionLog
from src.test.test_job import TestJob


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """启动后续测试任务。"""
    with ExecutionLog(PROJECT_ROOT / "logs" / "test" / "execution.log"):
        TestJob().run()


if __name__ == "__main__":
    main()
