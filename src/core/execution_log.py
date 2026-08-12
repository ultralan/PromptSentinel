"""将模块执行过程完整落盘的本地日志对象。"""

from __future__ import annotations

import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TextIO


class ExecutionLog:
    """捕获标准输出、标准错误和 Python 日志到单个模块执行日志文件。"""

    def __init__(self, project_root: Path, module: str) -> None:
        """创建模块专属、带 UTC 时间戳的日志文件路径。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = project_root / "logs" / module / f"{timestamp}.log"
        self._handle: TextIO | None = None
        self._stdout_redirect = None
        self._stderr_redirect = None
        self._root_logger = logging.getLogger()
        self._handler: logging.Handler | None = None

    def __enter__(self) -> "ExecutionLog":
        """开始重定向本进程输出并记录日志会话开始时间。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", buffering=1)
        self._stdout_redirect = redirect_stdout(self._handle)
        self._stderr_redirect = redirect_stderr(self._handle)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        self._handler = logging.StreamHandler(self._handle)
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self._root_logger.addHandler(self._handler)
        self._root_logger.setLevel(logging.INFO)
        logging.getLogger(__name__).info("执行开始: log=%s", self.path)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """记录成功或完整异常堆栈，并在退出时恢复进程输出流。"""
        logger = logging.getLogger(__name__)
        if exception_type is None:
            logger.info("执行成功")
        else:
            logger.exception("执行失败", exc_info=(exception_type, exception, traceback))
        if self._handler is not None:
            self._root_logger.removeHandler(self._handler)
            self._handler.close()
        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(exception_type, exception, traceback)
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(exception_type, exception, traceback)
        if self._handle is not None:
            self._handle.close()
        return False
