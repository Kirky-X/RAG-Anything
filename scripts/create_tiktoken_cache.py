import os
from pathlib import Path

import tiktoken

try:
    from raganything.config import RAGAnythingConfig

    cfg = RAGAnythingConfig()
    cache_dir = cfg.tiktoken.cache_dir or "./tiktoken_cache"
except Exception:
    cache_dir = "./tiktoken_cache"

os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache_dir)
Path(cache_dir).mkdir(parents=True, exist_ok=True)

print("Downloading and caching tiktoken models...")
tiktoken.get_encoding("cl100k_base")
print(f"tiktoken models have been cached in '{cache_dir}'")
