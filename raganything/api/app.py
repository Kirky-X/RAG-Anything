# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import Optional
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from raganything.server_config import load_server_configs, uvicorn_run_params
from raganything import RAGAnything, RAGAnythingConfig
from .models import (
    HealthResp,
    InfoResp,
    QueryReq,
    QueryResp,
    QueryMultiReq,
    UploadResp,
    ContentListInsertReq,
    DocStatusResp,
)
from .auth import get_auth


srv, api = load_server_configs()
app = FastAPI(
    title=srv.webui_title or "RAG-Anything API",
    description=srv.webui_description or "RAG-Anything HTTP API",
)

if srv.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=srv.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


rag: Optional[RAGAnything] = None


@app.on_event("startup")
async def startup_event():
    global rag
    rag = RAGAnything(config=RAGAnythingConfig())


@app.get("/health", response_model=HealthResp)
async def health():
    return HealthResp(ok=True)


@app.get("/api/info", response_model=InfoResp)
async def info():
    return InfoResp(
        title=srv.webui_title,
        description=srv.webui_description,
        host=srv.host,
        port=srv.port,
        cors_origins=srv.cors_origins,
        ssl_enabled=srv.ssl.enabled,
    )


@app.get("/api/secure")
async def secure(ensure=Depends(get_auth)):
    return {"secure": True}


@app.post("/api/query", response_model=QueryResp)
async def query(body: QueryReq, ensure=Depends(get_auth)):
    try:
        kwargs = {"vlm_enhanced": False}
        if body.top_k is not None:
            kwargs["top_k"] = body.top_k
        result = await rag.aquery(
            body.query,
            mode=body.mode,
            system_prompt=body.system_prompt,
            **kwargs,
        )
        return QueryResp(result=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/api/query/multimodal", response_model=QueryResp)
async def query_multimodal(body: QueryMultiReq, ensure=Depends(get_auth)):
    try:
        result = await rag.aquery_with_multimodal(
            body.query,
            multimodal_content=[
                item.dict() for item in body.multimodal_content
            ],
            mode=body.mode,
            top_k=body.top_k,
            vlm_enhanced=False,
        )
        return QueryResp(result=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/api/doc/upload", response_model=UploadResp)
async def upload_document(
    file: UploadFile = File(...),
    output_dir: Optional[str] = Form(None),
    parse_method: Optional[str] = Form(None),
    display_stats: Optional[bool] = Form(None),
    split_by_character: Optional[str] = Form(None),
    split_by_character_only: bool = Form(False),
    doc_id: Optional[str] = Form(None),
    ensure=Depends(get_auth),
):
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAGAnything not initialized",
        )
    try:
        import tempfile
        import os
        suffix = f"_{file.filename}"
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        await rag.process_document_complete(
            file_path=tmp_path,
            output_dir=output_dir or "",
            parse_method=parse_method or "",
            display_stats=bool(display_stats) if display_stats is not None else False,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            doc_id=doc_id,
        )
        os.unlink(tmp_path)
        return UploadResp(doc_id=doc_id, file_name=str(file.filename), status="processed")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/api/doc/insert", response_model=UploadResp)
async def insert_content_list(body: ContentListInsertReq, ensure=Depends(get_auth)):
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAGAnything not initialized",
        )
    try:
        await rag.insert_content_list(
            content_list=[item.dict() for item in body.content_list],
            file_path=body.file_path or "unknown_document",
            split_by_character=body.split_by_character,
            split_by_character_only=body.split_by_character_only,
            doc_id=body.doc_id,
            display_stats=bool(body.display_stats) if body.display_stats is not None else False,
        )
        return UploadResp(
            doc_id=body.doc_id,
            file_name=body.file_path or "unknown_document",
            status="processed",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/api/doc/status/{doc_id}", response_model=DocStatusResp)
async def doc_status(doc_id: str, ensure=Depends(get_auth)):
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAGAnything not initialized",
        )
    try:
        status = await rag.get_document_processing_status(doc_id)
        return DocStatusResp(**status)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def run():
    import uvicorn
    params = uvicorn_run_params(srv)
    uvicorn.run("raganything.api.app:app", **params)
