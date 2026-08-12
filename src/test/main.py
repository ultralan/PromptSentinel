"""测试模块启动入口。"""

from src.test.test_job import TestJob


def main() -> None:
    """启动后续测试任务。"""
    TestJob().run()


if __name__ == "__main__":
    main()
