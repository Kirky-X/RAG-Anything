from pathlib import Path

from raganything.logger import init_logger, logger


def _read_file_tail(path: Path, max_len: int = 2048) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8")
    return data[-max_len:]


def test_levels_and_dual_sinks(tmp_path: Path, capsys):
    """Test log levels and output to both console and file."""
    # Initialize to temp directory
    init_logger(level="DEBUG", log_dir=tmp_path)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # Verify console output
    captured = capsys.readouterr()
    out = captured.out
    assert "debug message" in out
    assert "info message" in out
    assert "warning message" in out
    assert "error message" in out
    assert "critical message" in out

    # Verify file output
    log_file = tmp_path / "raganything.log"
    # Wait for async write
    logger.complete()
    content = _read_file_tail(log_file)
    assert "debug message" in content
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content
    assert "critical message" in content


def test_rotation_and_retention(tmp_path: Path):
    """Test log rotation and retention policies."""
    # Use small size for rotation to trigger it quickly
    init_logger(level="DEBUG", log_dir=tmp_path, rotation="1 KB", retention="1 files")

    for i in range(200):
        logger.info(f"line-{i}-" + "x" * 40)

    logger.complete()

    # Verify files exist
    files = sorted(tmp_path.glob("raganything.log*"))
    assert len(files) >= 1
    # Active log file should exist
    assert (tmp_path / "raganything.log").exists()
