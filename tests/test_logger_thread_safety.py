import threading
from pathlib import Path

from raganything.logger import init_logger, logger


def _read_file_tail(path: Path, max_len: int = 2048) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8")
    return data[-max_len:]


def test_multithread_logging(tmp_path: Path):
    """Test thread safety of the logger."""
    init_logger(level="DEBUG", log_dir=tmp_path)

    messages = []

    # Thread worker function
    def worker(i: int):
        msg = f"thread-{i}"
        messages.append(msg)
        logger.info(msg)

    # Launch threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Wait for logs to flush
    logger.complete()

    # Verify all messages are present
    log_file = tmp_path / "raganything.log"
    content = _read_file_tail(log_file)
    for msg in messages:
        assert msg in content
