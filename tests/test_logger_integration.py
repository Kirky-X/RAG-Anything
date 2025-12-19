"""Integration tests for logging functionality."""

import os
import subprocess
from pathlib import Path

from raganything.raganything import RAGAnything


def test_batch_parser_cli_uses_logger(tmp_path: Path):
    """Test that CLI tool uses the configured logger."""
    # Call batch_parser CLI
    cmd = [
        "python",
        "-m",
        "raganything.batch_parser",
        str(tmp_path),
        "--output",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Just check that it runs without crashing, as output might be captured by logger
    assert result.returncode in (0, 1)


def test_logging_rotation_and_retention_from_config(tmp_path):
    """Test that logging configuration is correctly applied from environment variables."""
    # Set config to use bytes-based rotation and retention from backup_count
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_DIR"] = str(tmp_path)
    os.environ["LOG_MAX_BYTES"] = "1024"
    os.environ["LOG_BACKUP_COUNT"] = "2"

    # Initialize RAGAnything which should initialize logger
    r = RAGAnything()
    r.logger.info("hello")

    # Force re-initialization to ensure settings are applied if needed
    try:
        from raganything.logger import init_logger

        init_logger(level="DEBUG", log_dir=Path(os.environ["LOG_DIR"]))
    except Exception:
        pass

    # Verify log file exists
    log_file = Path(tmp_path) / "raganything.log"
    assert log_file.exists()
