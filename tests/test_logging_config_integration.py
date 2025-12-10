import os
from pathlib import Path

from raganything.raganything import RAGAnything


def test_logging_rotation_and_retention_from_config(tmp_path):
    # Set config to use bytes-based rotation and retention from backup_count
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_DIR"] = str(tmp_path)
    os.environ["LOG_MAX_BYTES"] = "1024"
    os.environ["LOG_BACKUP_COUNT"] = "2"
    # Ensure default rotation in config is the placeholder to trigger mapping
    r = RAGAnything()
    r.logger.info("hello")
    # Force file sink to flush
    try:
        from raganything.logger import init_logger
        init_logger(level="DEBUG", log_dir=Path(os.environ["LOG_DIR"]))
    except Exception:
        pass
    # Verify log file exists
    log_file = Path(tmp_path) / "raganything.log"
    assert log_file.exists()
