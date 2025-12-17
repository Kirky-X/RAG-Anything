"""自定义 Sink 注册与写入测试"""

import sqlite3
from pathlib import Path

from raganything.logger import register_sink, unregister_sink, SQLiteSink, init_logger, logger


def test_sqlite_sink_write(tmp_path: Path):
    """Test logging to a custom SQLite sink."""
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
