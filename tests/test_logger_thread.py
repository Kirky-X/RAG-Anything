"""多线程日志并发安全测试"""

import threading
from pathlib import Path

from raganything.logger import init_logger, logger


def test_thread_safety(tmp_path: Path):
    init_logger(level="INFO", log_dir=tmp_path)

    results = []

    def worker(n: int):
        for i in range(100):
            msg = f"T{n}-{i}"
            logger.info(msg)
            results.append(msg)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.complete()
    data = (tmp_path / "raganything.log").read_text(encoding="utf-8")
    for msg in results:
        assert msg in data

