import os
import tempfile
from pathlib import Path

import pytest

from raganything.server_config import load_server_configs


def _write_tmp_toml(content: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(path)


def test_server_toml_loading_and_env_override(monkeypatch):
    content = """
    [server]
    host = "0.0.0.0"
    port = 9000
    workers = 2
    cors_origins = ["http://localhost:3000"]
    webui_title = "KB"
    webui_description = "Desc"

    [ssl]
    enabled = true
    certfile = "/path/cert.pem"
    keyfile = "/path/key.pem"

    [api]
    lightrag_api_key = "toml-key"
    whitelist_paths = ["/health", "/api/*"]
    """
    path = _write_tmp_toml(content)
    try:
        monkeypatch.setenv("CONFIG_TOML", str(path))
        # Override sensitive via env
        monkeypatch.setenv("LIGHTRAG_API_KEY", "env-key")
        # Override server port via env
        monkeypatch.setenv("SERVER_PORT", "9100")
        # Ensure ambient env does not override host
        monkeypatch.delenv("SERVER_HOST", raising=False)
        monkeypatch.delenv("HOST", raising=False)
        srv, api = load_server_configs()
        assert srv.port == 9100
        assert srv.host == "0.0.0.0"
        assert srv.workers == 2
        assert srv.cors_origins == ["http://localhost:3000"]
        assert srv.ssl.enabled is True
        assert srv.ssl.certfile == "/path/cert.pem"
        assert srv.ssl.keyfile == "/path/key.pem"
        assert api.lightrag_api_key == "env-key"
        assert api.whitelist_paths == ["/health", "/api/*"]
    finally:
        path.unlink(missing_ok=True)


def test_server_env_only(monkeypatch):
    # No TOML, env-only
    monkeypatch.delenv("CONFIG_TOML", raising=False)
    # Ensure default root config.toml is ignored
    monkeypatch.setenv("CONFIG_TOML", "")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8088")
    monkeypatch.setenv("WORKERS", "3")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com")
    srv, api = load_server_configs()
    assert srv.host == "127.0.0.1"
    assert srv.port == 8088
    assert srv.workers == 3
    assert srv.cors_origins == ["http://a.com", "http://b.com"]
