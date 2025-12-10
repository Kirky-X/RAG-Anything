import os
from pathlib import Path

import tempfile

import pytest

from raganything.config import RAGAnythingConfig


def _write_tmp_toml(content: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(path)


def test_env_fallback_and_properties(monkeypatch):
    # Ensure no default config.toml is loaded
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    monkeypatch.setenv("CONFIG_TOML", p)
    monkeypatch.setenv("WORKING_DIR", "/tmp/rag_storage")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("PARSER", "docling")
    monkeypatch.setenv("PARSE_METHOD", "txt")
    monkeypatch.setenv("DISPLAY_CONTENT_STATS", "false")
    cfg = RAGAnythingConfig()
    assert cfg.working_dir == "/tmp/rag_storage"
    assert cfg.parser_output_dir == "/tmp/output"
    assert cfg.parser == "docling"
    assert cfg.parse_method == "txt"
    assert cfg.display_content_stats is False
    assert cfg.directory.working_dir == "/tmp/rag_storage"
    assert cfg.parsing.parser == "docling"


def test_toml_loading_overrides_env(monkeypatch):
    monkeypatch.setenv("WORKING_DIR", "/env/storage")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    content = """
    [directory]
    working_dir = "/toml/storage"
    parser_output_dir = "/toml/output"

    [raganything.parsing]
    parser = "mineru"
    parse_method = "ocr"
    display_content_stats = true

    [raganything.llm]
    provider = "openrouter"
    model = "openrouter/model"
    api_base = "https://openrouter.ai/api/v1"
    api_key = "k"
    timeout = 30
    max_retries = 1

    [logging]
    level = "DEBUG"
    verbose = true
    max_bytes = 1024
    backup_count = 2
    dir = "/logs"
    rotation = "1 KB"
    retention = "1 files"

    [tiktoken]
    cache_dir = "/tkcache"

    [query]
    max_token_entity_desc = 6000
    """
    path = _write_tmp_toml(content)
    try:
        monkeypatch.setenv("CONFIG_TOML", str(path))
        cfg = RAGAnythingConfig()
        assert cfg.working_dir == "/toml/storage"
        assert cfg.parser_output_dir == "/toml/output"
        assert cfg.parser == "mineru"
        assert cfg.parse_method == "ocr"
        assert cfg.llm_provider == "openrouter"
        assert cfg.llm_api_base == "https://openrouter.ai/api/v1"
        assert cfg.llm_timeout == 30
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.verbose is True
        assert cfg.logging.max_bytes == 1024
        assert cfg.logging.backup_count == 2
        assert cfg.logging.dir == "/logs"
        assert cfg.logging.rotation == "1 KB"
        assert cfg.logging.retention == "1 files"
        assert cfg.tiktoken.cache_dir == "/tkcache"
        assert cfg.query.max_token_entity_desc == 6000
    finally:
        path.unlink(missing_ok=True)


def test_validation_invalid_parser(monkeypatch):
    content = """
    [raganything.parsing]
    parser = "unknown"
    parse_method = "auto"
    display_content_stats = true
    """
    path = _write_tmp_toml(content)
    try:
        monkeypatch.setenv("CONFIG_TOML", str(path))
        with pytest.raises(ValueError):
            RAGAnythingConfig()
    finally:
        path.unlink(missing_ok=True)
