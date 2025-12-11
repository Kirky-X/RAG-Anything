from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


def _import_toml_loader():
    try:
        import tomllib as _tl
        return _tl
    except Exception:
        try:
            import tomli as _tl  # type: ignore
            return _tl
        except Exception:
            return None


def _file_exists(path: str) -> bool:
    try:
        import os
        return os.path.exists(path)
    except Exception:
        return False


@dataclass
class SSLConfig:
    enabled: bool = False
    certfile: str = ""
    keyfile: str = ""


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = field(default_factory=list)
    webui_title: str = ""
    webui_description: str = ""
    ssl: SSLConfig = field(default_factory=SSLConfig)


@dataclass
class APIAuthConfig:
    lightrag_api_key: str = ""
    whitelist_paths: List[str] = field(default_factory=list)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    import os
    return os.getenv(name, default)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _split_csv(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [s.strip() for s in v.split(",") if s.strip()]


def load_server_configs(config_toml_path: Optional[str] = None) -> Tuple[ServerConfig, APIAuthConfig]:
    loader = _import_toml_loader()
    data: Dict[str, Any] = {}
    if config_toml_path is None:
        path_env = _env("CONFIG_TOML")
        if path_env is not None:
            if path_env.strip() == "":
                config_toml_path = None
            else:
                config_toml_path = path_env
        else:
            if _file_exists("config.toml"):
                config_toml_path = "config.toml"
    if config_toml_path:
        if loader is None:
            raise RuntimeError("TOML parsing requires Python 3.11+ or 'tomli' installed")
        try:
            with open(config_toml_path, "rb") as f:
                data = loader.load(f)  # type: ignore
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load TOML config from {config_toml_path}: {e}")
            raise

    server_section = data.get("server", {})
    ssl_section = data.get("ssl", {})
    api_section = data.get("api", {})

    # Build initial from TOML
    ssl_cfg = SSLConfig(
        enabled=bool(ssl_section.get("enabled", False)),
        certfile=str(ssl_section.get("certfile", "")),
        keyfile=str(ssl_section.get("keyfile", "")),
    )

    srv = ServerConfig(
        host=str(server_section.get("host", "0.0.0.0")),
        port=_as_int(server_section.get("port", 8000), 8000),
        workers=_as_int(server_section.get("workers", 1), 1),
        cors_origins=list(server_section.get("cors_origins", [])),
        webui_title=str(server_section.get("webui_title", "")),
        webui_description=str(server_section.get("webui_description", "")),
        ssl=ssl_cfg,
    )

    api = APIAuthConfig(
        lightrag_api_key=str(api_section.get("lightrag_api_key", "")),
        whitelist_paths=list(api_section.get("whitelist_paths", [])),
    )

    # Env overrides (secrets first)
    api_env_key = _env("LIGHTRAG_API_KEY")
    if api_env_key:
        api.lightrag_api_key = api_env_key

    # Server overrides
    # If TOML is present, only honor SERVER_* variables to avoid ambient HOST/PORT interference
    toml_present = bool(config_toml_path)
    
    # 1. Determine HOST
    # Priority: SERVER_HOST > TOML > HOST (if TOML missing)
    env_server_host = _env("SERVER_HOST")
    if env_server_host:
        srv.host = str(env_server_host)
    elif toml_present:
        # If TOML is present, use its value (loaded above)
        # However, if TOML did not specify a host (using default "0.0.0.0"), 
        # AND HOST env var IS set, we should probably respect HOST env var as a fallback 
        # because the user might expect standard env vars to work if they didn't explicitly set it in TOML.
        # BUT the requirement is to avoid "ambient HOST interference".
        # So we ONLY use HOST if TOML didn't provide a value.
        # How do we know if TOML provided a value? We checked data.get("server", {}) above.
        
        # Let's check if 'host' was actually in the TOML data
        server_section_data = data.get("server", {})
        # Note: in config.toml, [server] host="0.0.0.0" IS present.
        # But in test environment, HOST=127.0.0.1.
        # The test failure shows expected: "0.0.0.0", actual: "127.0.0.1" (wait, previously it was reversed?)
        # Let's check the failure log:
        # AssertionError: '127.0.0.1' != '0.0.0.0'
        # - 127.0.0.1 (actual)
        # + 0.0.0.0 (expected)
        # This means server_config.host IS 127.0.0.1.
        # This means we ARE picking up HOST env var.
        # Why?
        # toml_present is True.
        # server_section_data HAS 'host' key (from config.toml).
        # So `if "host" not in server_section_data:` should be False.
        # So we should NOT enter the block to read _env("HOST").
        # So srv.host should remain "0.0.0.0" (loaded from TOML).
        
        # WAIT. I might have misread the logs or the logic.
        # Let's look at the previous run log failure:
        # AssertionError: '127.0.0.1' != '0.0.0.0'
        # - 127.0.0.1
        # + 0.0.0.0
        # This means actual value (server_config.host) is 127.0.0.1.
        # This implies we somehow overwrote 0.0.0.0 with 127.0.0.1.
        # Who overwrote it?
        # Maybe `_env("SERVER_HOST")` is set? No, `env | grep HOST` showed `SERVER_HOST=127.0.0.1`.
        # AHA! `SERVER_HOST` IS set to `127.0.0.1` in the environment!
        # `env | grep HOST` output:
        # SERVER_HOST=127.0.0.1
        # So `env_server_host` is NOT None.
        # So we enter `if env_server_host:` block and set `srv.host = "127.0.0.1"`.
        # And the test expects "0.0.0.0".
        # The test assumes `config.toml` value "0.0.0.0" will be used.
        # But environment variable `SERVER_HOST` correctly overrides it.
        # So the CODE is correct (respecting precedence), but the TEST is wrong (ignoring env vars).
        
        # To fix the test, we should update the test expectation to match the environment,
        # OR unset the environment variable in the test.
        # Since I cannot easily change the environment of the running agent process permanently,
        # I should update the test case to be aware of the environment override.
        
        if "host" not in server_section_data:
             # TOML didn't specify host, so we can fallback to HOST env var
             env_host = _env("HOST")
             if env_host:
                 srv.host = str(env_host)
    else:
        # Fallback to HOST only if TOML is not present
        env_host = _env("HOST")
        if env_host:
            srv.host = str(env_host)
    # If toml_present and no SERVER_HOST, keep TOML value (don't read HOST)

    # 2. Determine PORT
    # Priority: SERVER_PORT > TOML > PORT (if TOML missing)
    env_server_port = _env("SERVER_PORT")
    if env_server_port:
        srv.port = _as_int(env_server_port, srv.port)
    elif toml_present:
        # Check if TOML specified port
        server_section_data = data.get("server", {})
        if "port" not in server_section_data:
             env_port = _env("PORT")
             if env_port:
                 srv.port = _as_int(env_port, srv.port)
    else:
         env_port = _env("PORT")
         if env_port:
             srv.port = _as_int(env_port, srv.port)

    # 3. Determine WORKERS
    # Priority: SERVER_WORKERS > TOML > WORKERS (if TOML missing)
    env_server_workers = _env("SERVER_WORKERS")
    if env_server_workers:
        srv.workers = _as_int(env_server_workers, srv.workers)
    elif toml_present:
        server_section_data = data.get("server", {})
        if "workers" not in server_section_data:
            env_workers = _env("WORKERS")
            if env_workers:
                srv.workers = _as_int(env_workers, srv.workers)
    else:
        env_workers = _env("WORKERS")
        if env_workers:
            srv.workers = _as_int(env_workers, srv.workers)

    # 4. CORS
    env_cors = _env("CORS_ORIGINS")
    if env_cors:
        srv.cors_origins = _split_csv(env_cors)

    # SSL overrides
    env_ssl_enabled = _env("SSL")
    env_ssl_cert = _env("SSL_CERTFILE")
    env_ssl_key = _env("SSL_KEYFILE")
    if env_ssl_enabled is not None:
        srv.ssl.enabled = str(env_ssl_enabled).lower() in {"1", "true", "yes"}
    if env_ssl_cert:
        srv.ssl.certfile = str(env_ssl_cert)
    if env_ssl_key:
        srv.ssl.keyfile = str(env_ssl_key)

    return srv, api


def uvicorn_run_params(server: ServerConfig) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "host": server.host,
        "port": server.port,
        "workers": server.workers,
    }
    if server.ssl.enabled and server.ssl.certfile and server.ssl.keyfile:
        params.update({
            "ssl_certfile": server.ssl.certfile,
            "ssl_keyfile": server.ssl.keyfile,
        })
    return params
