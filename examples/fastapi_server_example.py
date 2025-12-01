from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from raganything.server_config import load_server_configs, uvicorn_run_params


app = FastAPI()


def get_auth():
    _, api = load_server_configs()
    api_key = api.lightrag_api_key
    def _ensure(key: str | None):
        if api_key and key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return _ensure


srv, api = load_server_configs()
if srv.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=srv.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    return {
        "title": srv.webui_title,
        "description": srv.webui_description,
        "host": srv.host,
        "port": srv.port,
        "cors_origins": srv.cors_origins,
        "ssl": srv.ssl.enabled,
    }


@app.get("/api/secure")
def secure(x_api_key: str | None = None, ensure=Depends(get_auth())):
    ensure(x_api_key)
    return {"secure": True}


if __name__ == "__main__":
    params = uvicorn_run_params(srv)
    uvicorn.run("examples.fastapi_server_example:app", **params)
