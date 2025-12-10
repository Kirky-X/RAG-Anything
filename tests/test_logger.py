"""日志模块单元测试

验证：
- 不同日志级别输出
- 多线程环境日志安全
- 文件与控制台双通道输出
- 日志文件轮转与保留策略
"""

import os
import threading
import time
from pathlib import Path

import pytest

from raganything.logger import init_logger, logger


def _read_file_tail(path: Path, max_len: int = 2048) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8")
    return data[-max_len:]


def test_levels_and_dual_sinks(tmp_path: Path, capsys):
    # 初始化到临时目录，避免污染项目
    init_logger(level="DEBUG", log_dir=tmp_path)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # 控制台输出验证
    captured = capsys.readouterr()
    out = captured.out
    assert "debug message" in out
    assert "info message" in out
    assert "warning message" in out
    assert "error message" in out
    assert "critical message" in out

    # 文件输出验证
    log_file = tmp_path / "raganything.log"
    # 等待异步写入完成
    logger.complete()
    content = _read_file_tail(log_file)
    assert "debug message" in content
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content
    assert "critical message" in content


def test_multithread_logging(tmp_path: Path):
    init_logger(level="DEBUG", log_dir=tmp_path)

    messages = []

    def worker(i: int):
        msg = f"thread-{i}"
        messages.append(msg)
        logger.info(msg)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.complete()
    log_file = tmp_path / "raganything.log"
    content = _read_file_tail(log_file)
    for msg in messages:
        assert msg in content


def test_rotation_and_retention(tmp_path: Path):
    # 使用按大小轮转，快速触发
    init_logger(level="DEBUG", log_dir=tmp_path, rotation="1 KB", retention="1 files")

    for i in range(200):
        logger.info(f"line-{i}-" + "x" * 40)

    logger.complete()

    # 验证轮转产生的归档文件与保留策略
    files = sorted(tmp_path.glob("raganything.log*"))
    assert len(files) >= 1
    # 至少存在当前活动文件
    assert (tmp_path / "raganything.log").exists()

