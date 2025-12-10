"""
专业日志模块，使用 Loguru 提供统一的彩色控制台与文件双通道日志输出。

此模块在导入时完成一次性初始化，并向项目暴露统一的 `logger` 实例。

功能特性：
- 控制台（stdio）彩色输出，适配不同日志级别
- 文件输出到项目根目录 `logs/`，按天自动分割（每日轮转）
- 支持多线程安全写入（enqueue 模式）
- 支持 DEBUG/INFO/WARNING/ERROR/CRITICAL 等日志级别

环境变量（可选）用于覆盖默认行为：
- `RAG_LOG_LEVEL`：日志级别，默认 `INFO`
- `RAG_LOG_DIR`：日志目录，默认项目根目录下 `logs/`
- `RAG_LOG_RETENTION`：日志保留策略，默认 `7 days`

示例使用：
    from raganything.logger import logger
    logger.info("系统启动")

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


_CONFIGURED = False


def _project_root() -> Path:
    """获取项目根目录路径。

    Returns:
        Path: 项目根目录路径。
    """
    # 当前文件位于 `raganything/` 包内，项目根目录为上一级
    return Path(__file__).resolve().parents[1]


def init_logger(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    rotation: str = "00:00",
    retention: str = "7 days",
    enqueue: bool = True,
) -> None:
    """初始化 Loguru 日志系统。

    Args:
        level: 初始日志级别，默认 "INFO"。
        log_dir: 日志目录，默认项目根目录下的 `logs/`。
        rotation: 文件轮转策略，默认每日 00:00 轮转。
        retention: 日志保留策略，默认保留 7 天。
        enqueue: 是否开启多线程/多进程安全队列，默认 True。
    """
    global _CONFIGURED
    # 允许重复调用进行重配置：每次都会移除并重新添加 sink

    # 环境变量覆盖
    env_level = os.getenv("RAG_LOG_LEVEL")
    env_dir = os.getenv("RAG_LOG_DIR")
    env_retention = os.getenv("RAG_LOG_RETENTION")

    level = (env_level or level).upper()
    base_log_dir = Path(env_dir) if env_dir else (log_dir or Path("/tmp/raganything/logs"))
    retention = env_retention or retention

    base_log_dir.mkdir(parents=True, exist_ok=True)

    # 清空已有 sink，确保统一配置
    _logger.remove()

    # 控制台彩色输出
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <white>{message}</white>"
    )
    _logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=console_format,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )

    # 文件输出（每日轮转）
    # 使用固定文件名，便于轮转时生成归档文件
    file_path = base_log_dir / "raganything.log"
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    # 兼容类似 "1 files" 的保留写法
    retention_value = retention
    if isinstance(retention_value, str) and retention_value.strip().lower().endswith(" files"):
        try:
            retention_value = int(retention_value.strip().split()[0])
        except Exception:
            retention_value = retention

    _logger.add(
        str(file_path),
        level=level,
        rotation=rotation,
        retention=retention_value,
        encoding="utf-8",
        enqueue=enqueue,
        colorize=False,
        format=file_format,
        backtrace=False,
        diagnose=False,
    )

    _CONFIGURED = True


def set_level(level: str) -> None:
    """动态调整日志级别。

    Args:
        level: 新的日志级别，例如 "DEBUG"、"INFO" 等。
    """
    # 通过重新初始化达到统一变更（保持双 sink 一致）
    # 仅修改级别不改变目录与轮转策略
    init_logger(level=level)


def get_logger():
    """获取统一的 logger 实例。

    Returns:
        loguru.Logger: 项目统一日志实例。
    """
    if not _CONFIGURED:
        init_logger()
    return _logger


# 模块导入即完成初始化，并暴露统一实例
init_logger()
logger = _logger


class BaseLogSink:
    """日志 Sink 接口。

    提供扩展自定义日志输出通道的统一协议，例如数据库（Postgres/SQLite）、消息队列等。

    用法：实现 `__call__(message)` 或 `write(message)` 以接收 Loguru 的 `Message`。
    推荐实现 `__call__`，Loguru 会将每条日志以 `Message` 对象回调至该方法。

    Attributes:
        无
    """

    def __call__(self, message):  # noqa: D401
        """处理一条日志消息。

        Args:
            message: Loguru 的 `Message` 对象，包含 `record` 字段（时间、级别、文本、位置等）。
        """
        raise NotImplementedError


def register_sink(sink: BaseLogSink | object, level: str = "INFO", enqueue: bool = True, **kwargs) -> int:
    """注册自定义日志 Sink。

    该函数为扩展点，对外暴露统一注册入口。`sink` 可以是实现了 `__call__(message)` 的对象，
    或者实现了 `write(message)` 的类实例。

    Args:
        sink: 自定义 Sink 对象（实现 `__call__` 或 `write`）。
        level: 最低日志级别，默认 `INFO`。
        enqueue: 是否开启队列（多线程/多进程安全），默认 True。
        **kwargs: 透传给 Loguru 的其他参数，如 `filter`、`format`、`serialize` 等。

    Returns:
        int: 该 sink 的 handler_id，可用于后续移除。
    """
    handler_id = _logger.add(
        sink,
        level=level.upper(),
        enqueue=enqueue,
        backtrace=False,
        diagnose=False,
        **kwargs,
    )
    return handler_id


def unregister_sink(handler_id: int) -> None:
    """移除已注册的自定义日志 Sink。

    Args:
        handler_id: `register_sink()` 返回的标识符。
    """
    try:
        _logger.remove(handler_id)
    except ValueError:
        # 已被移除或无效 id
        pass


class SQLiteSink(BaseLogSink):
    """SQLite 日志 Sink，实现结构化日志入库。

    无外部依赖，便于在本地或测试环境验证日志入库逻辑。未来可以用同样接口接入 Postgres 等。

    Args:
        db_path: SQLite 文件路径。
        table_name: 表名，默认 `logs`。

    表结构：
        - time TEXT
        - level TEXT
        - name TEXT
        - function TEXT
        - line INTEGER
        - message TEXT
    """

    def __init__(self, db_path: Path | str, table_name: str = "logs"):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    time TEXT,
                    level TEXT,
                    name TEXT,
                    function TEXT,
                    line INTEGER,
                    message TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def __call__(self, message) -> None:
        import sqlite3

        record = message.record
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.execute(
                f"INSERT INTO {self.table_name} (time, level, name, function, line, message) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(record["time"]),
                    record["level"].name,
                    record["name"],
                    record["function"],
                    int(record["line"]),
                    record["message"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
