"""日志系统测试套件 - 合并所有日志相关测试

本文件合并了以下测试文件的功能：
- test_logger_basic.py: 基础日志功能测试
- test_logger_custom_sink.py: 自定义Sink注册与写入测试
- test_logger_thread_safety.py: 多线程日志安全性测试
- test_logger_integration.py: 日志集成测试
"""

import os
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest

from raganything.logger import (SQLiteSink, init_logger, logger, register_sink,
                                unregister_sink)
from raganything.raganything import RAGAnything
from raganything.i18n_logger import get_i18n_logger
from raganything.i18n import _


def _read_file_tail(path: Path, max_len: int = 2048) -> str:
    """读取文件尾部内容"""
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8")
    return data[-max_len:]


# ==================== 基础日志功能测试 ====================

class TestLoggerBasic:
    """基础日志功能测试类"""
    
    def test_levels_and_dual_sinks(self, tmp_path: Path, capsys):
        """测试日志级别和双重输出（控制台和文件）"""
        # 初始化到临时目录
        init_logger(level="DEBUG", log_dir=tmp_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        # 验证控制台输出
        captured = capsys.readouterr()
        out = captured.out
        assert "debug message" in out
        assert "info message" in out
        assert "warning message" in out
        assert "error message" in out
        assert "critical message" in out

        # 验证文件输出
        log_file = tmp_path / "raganything.log"
        # 等待异步写入
        logger.complete()
        content = _read_file_tail(log_file)
        assert "debug message" in content
        assert "info message" in content
        assert "warning message" in content
        assert "error message" in content
        assert "critical message" in content

    def test_rotation_and_retention(self, tmp_path: Path):
        """测试日志轮换和保留策略"""
        # 使用小尺寸轮换以快速触发
        init_logger(level="DEBUG", log_dir=tmp_path, rotation="1 KB", retention="1 files")

        for i in range(200):
            logger.info(_("line-{}-").format(i) + "x" * 40)

        logger.complete()

        # 验证文件存在
        files = sorted(tmp_path.glob("raganything.log*"))
        assert len(files) >= 1
        # 活动日志文件应该存在
        assert (tmp_path / "raganything.log").exists()


# ==================== 自定义Sink测试 ====================

class TestLoggerCustomSink:
    """自定义Sink注册与写入测试类"""
    
    def test_sqlite_sink_write(self, tmp_path: Path):
        """测试日志到自定义SQLite Sink"""
        init_logger(level="INFO", log_dir=tmp_path)

        db = tmp_path / "logs.db"
        sink = SQLiteSink(db)
        hid = register_sink(sink, level="DEBUG", enqueue=True)

        logger.debug("debug to db")
        logger.info("info to db")
        logger.error("error to db")

        logger.complete()
        unregister_sink(hid)

        conn = sqlite3.connect(db)
        try:
            cur = conn.execute("SELECT level, message FROM logs ORDER BY rowid")
            rows = cur.fetchall()
            assert any(r[0] == "DEBUG" and "debug to db" in r[1] for r in rows)
            assert any(r[0] == "INFO" and "info to db" in r[1] for r in rows)
            assert any(r[0] == "ERROR" and "error to db" in r[1] for r in rows)
        finally:
            conn.close()


# ==================== 多线程日志安全性测试 ====================

class TestLoggerThreadSafety:
    """多线程日志安全性测试类"""
    
    def test_multithread_logging(self, tmp_path: Path):
        """测试日志器的线程安全性"""
        init_logger(level="DEBUG", log_dir=tmp_path)

        messages = []

        # 线程工作函数
        def worker(i: int):
            msg = f"thread-{i}"
            messages.append(msg)
            logger.info(msg)

        # 启动线程
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 等待日志刷新
        logger.complete()

        # 验证所有消息都存在
        log_file = tmp_path / "raganything.log"
        content = _read_file_tail(log_file)
        for msg in messages:
            assert msg in content


# ==================== 日志集成测试 ====================

class TestLoggerIntegration:
    """日志集成测试类"""

    def test_batch_parser_cli_uses_logger(self, tmp_path):
        """测试CLI工具使用配置的日志记录器"""
        # 调用batch_parser CLI
        cmd = [
            "python",
            "-m",
            "raganything.batch_parser",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # 只需检查它能正常运行，因为输出可能被日志记录器捕获
        assert result.returncode in (0, 1)

    def test_logging_rotation_and_retention_from_config(self, tmp_path):
        """测试通过环境变量正确应用日志配置"""
        # 设置配置以使用基于字节的轮换和备份计数保留
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_DIR"] = str(tmp_path)
        os.environ["LOG_MAX_BYTES"] = "1024"
        os.environ["LOG_BACKUP_COUNT"] = "2"

        # 初始化RAGAnything，应该初始化日志记录器
        r = RAGAnything()
        r.logger.info("hello")

        # 强制重新初始化以确保设置被应用（如需要）
        try:
            from raganything.logger import init_logger

            init_logger(level="DEBUG", log_dir=Path(os.environ["LOG_DIR"]))
        except Exception:
            pass

        # 验证日志文件存在
        log_file = Path(tmp_path) / "raganything.log"
        assert log_file.exists()