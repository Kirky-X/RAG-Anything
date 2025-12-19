# Copyright (c) 2025 Kirky.X
# All rights reserved.

import os
import uuid
from typing import Any, Optional

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form,
                     HTTPException, Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware

from raganything import RAGAnything, RAGAnythingConfig
from raganything.api.auth import get_auth
from raganything.api.models import (BatchProcessResp, ConfigResp,
                                    ContentListInsertReq, DocStatusResp,
                                    HealthResp, InfoResp, QueryMultiReq,
                                    QueryReq, QueryResp, StatsResp, UploadResp)
from raganything.health import (ComponentStatus, ConsoleNotifier,
                                HealthMonitor, OllamaHealthCheck,
                                SystemResourceCheck)
from raganything.logger import logger
from raganything.server_config import load_server_configs, uvicorn_run_params

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
    # Apply patches before initializing anything else
    import raganything.patches.lightrag_patch

    # Initialize Health Monitor and run pre-checks
    monitor = HealthMonitor()
    monitor.add_check(OllamaHealthCheck("config.toml"))
    monitor.add_check(SystemResourceCheck())
    monitor.add_notifier(ConsoleNotifier())

    logger.info("Running system health pre-checks...")

    results = await monitor.run_checks()
    unhealthy = [r for r in results.values() if r.status == ComponentStatus.UNHEALTHY]

    if unhealthy:
        logger.error("⚠️ SYSTEM STARTUP WARNING: Some components are unhealthy!")
        for r in unhealthy:
            logger.error(f"  - {r.component_name}: {r.message}")
        # We continue startup but logged critical warnings
    else:
        logger.info("✅ All system pre-checks passed.")

    rag = RAGAnything()
    await rag.initialize()
    rag.logger.info("Server has started and RAGAnything is initialized.")


@app.get("/health", response_model=HealthResp)
async def health():
    return HealthResp(ok=True)


@app.get("/health/detailed")
async def health_detailed():
    """Run detailed health checks on demand."""
    monitor = HealthMonitor()
    monitor.add_check(OllamaHealthCheck("config.toml"))
    monitor.add_check(SystemResourceCheck())

    results = await monitor.run_checks()

    response = {}
    for name, res in results.items():
        response[name] = {
            "status": res.status.name,
            "message": res.message,
            "metadata": res.metadata,
            "timestamp": res.timestamp.isoformat(),
        }
    return response


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

        # Add smart processing wait if requested
        if getattr(body, "wait_for_processing", False):
            kwargs["wait_for_processing"] = True
            kwargs["max_wait_time"] = getattr(body, "max_wait_time", 30)

        result = await rag.aquery(
            body.query,
            mode=body.mode,
            system_prompt=body.system_prompt,
            **kwargs,
        )

        # Enhanced response with processing status
        response_data = {"result": result or ""}

        # If result is empty, check if documents are still processing
        if not result and hasattr(rag, "get_system_stats"):
            try:
                stats = await rag.get_system_stats()
                if stats.processing_queue > 0:
                    response_data["warning"] = (
                        f"Query returned empty results. {stats.processing_queue} documents are still processing. Try again later or use wait_for_processing=true parameter."
                    )
                else:
                    response_data["info"] = (
                        "Query returned empty results. No relevant content found in processed documents."
                    )
            except Exception:
                pass

        return QueryResp(**response_data)
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
            multimodal_content=[item.dict() for item in body.multimodal_content],
            mode=body.mode,
            top_k=body.top_k,
            vlm_enhanced=False,
        )
        return QueryResp(result=result or "")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


import asyncio


async def process_document_background(file_path: str, doc_id: str, user: str):
    rag.logger.info(f"ENTERING process_document_background for doc_id: {doc_id}")
    try:
        rag.logger.info(
            f"Background processing started for doc_id {doc_id} (file: {file_path})"
        )
        rag.logger.info(f"RAG instance ID: {id(rag)}")

        # Ensure initial status is set immediately before calling process_document_complete
        try:
            # Check if we need to initialize
            rag.logger.info(f"Calling _ensure_lightrag_initialized for {doc_id}")
            init_res = await rag._ensure_lightrag_initialized()
            rag.logger.info(f"_ensure_lightrag_initialized returned for {doc_id}")
            if not init_res.get("success"):
                rag.logger.error(
                    f"Failed to initialize LightRAG in background task: {init_res}"
                )

            if rag.lightrag:
                rag.logger.info(f"LightRAG instance ID: {id(rag.lightrag)}")
                import os
                import time

                from raganything.base import DocStatus

                rag.logger.info(f"Getting status for {doc_id}")
                current_status = await rag.lightrag.doc_status.get_by_id(doc_id)
                rag.logger.info(f"Status for {doc_id} is: {current_status}")
                if not current_status:
                    rag.logger.info(
                        f"Pre-initializing status for {doc_id} in background task wrapper"
                    )
                    await rag.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                "content_summary": "Document being processed...",
                                "content_length": 0,
                                "file_path": os.path.basename(file_path),
                                "status": DocStatus.HANDLING,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                                "chunks_count": 0,
                                "multimodal_processed": False,
                            }
                        }
                    )
                    await rag.lightrag.doc_status.index_done_callback()

                    # Verify immediate write
                    verify = await rag.lightrag.doc_status.get_by_id(doc_id)
                    rag.logger.info(
                        f"Verified status in background task for {doc_id}: {verify}"
                    )
                else:
                    rag.logger.info(
                        f"Status already exists for {doc_id} in background task: {current_status}"
                    )
        except Exception as pre_init_error:
            rag.logger.warning(
                f"Failed to pre-initialize status for {doc_id}: {pre_init_error}"
            )
            import traceback

            rag.logger.warning(traceback.format_exc())

        rag.logger.info(f"About to call process_document_complete for doc_id: {doc_id}")
        await rag.process_document_complete(file_path, doc_id=doc_id, user=user)
        rag.logger.info(f"Background processing completed for doc_id {doc_id}")
    except Exception as e:
        rag.logger.error(
            f"Error processing file in background for doc_id {doc_id}: {e}",
            exc_info=True,
        )


@app.post("/api/doc/upload", response_model=UploadResp, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    user: str = Form("default"),
    ensure=Depends(get_auth),
):
    try:
        rag.logger.info(f"Received upload request for doc_id: {doc_id}")
        file_path, final_doc_id = await rag._save_upload_file(file, doc_id=doc_id)
        rag.logger.info(f"File saved to: {file_path}, final_doc_id: {final_doc_id}")

        background_tasks.add_task(
            process_document_background, file_path, final_doc_id, user
        )
        rag.logger.info(
            f"Added process_document_background to background_tasks for doc_id: {final_doc_id}"
        )

        return UploadResp(
            doc_id=final_doc_id,
            file_name=file_path,
            status="processing",
        )
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
            display_stats=(
                bool(body.display_stats) if body.display_stats is not None else False
            ),
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


@app.post("/api/batch/folder", response_model=BatchProcessResp)
async def process_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    recursive: bool = Form(True),
    user: str = Form("default"),
    ensure=Depends(get_auth),
):
    """Batch process an entire folder"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

    total_files = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            total_files += len(files)
            if not recursive:
                break
    except Exception:
        pass

    batch_id = f"batch-{uuid.uuid4()}"

    async def run_batch():
        await rag.process_folder_complete(
            folder_path=folder_path,
            recursive=recursive,
        )

    background_tasks.add_task(run_batch)

    return BatchProcessResp(
        batch_id=batch_id, total_files=total_files, status="processing"
    )


@app.get("/api/config", response_model=ConfigResp)
async def get_config(ensure=Depends(get_auth)):
    """Get current system configuration"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")
    return ConfigResp(
        multimodal=rag.config.multimodal,
        batch=rag.config.batch,
        parsing=rag.config.parsing,
    )


@app.post("/api/config/reload")
async def reload_config(ensure=Depends(get_auth)):
    """Reload configuration"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")
    try:
        rag.reload_config()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResp)
async def get_stats(ensure=Depends(get_auth)):
    """Get system statistics"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")
    stats = await rag.get_system_stats()
    return StatsResp(
        total_documents=stats.total_docs,
        processing_queue=stats.queue_size,
        storage_usage=stats.storage_usage,
        average_processing_time=stats.avg_processing_time,
    )


@app.delete("/api/storage/doc/{doc_id}")
async def delete_document(doc_id: str, ensure=Depends(get_auth)):
    """Delete a document"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")
    try:
        await rag.delete_document(doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/storage/cleanup")
async def cleanup_storage(ensure=Depends(get_auth)):
    """Cleanup storage"""
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGAnything not initialized")
    try:
        result = await rag.cleanup_storage()
        return {"status": "success", "deleted_items": result.deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run():
    import argparse
    import sys

    import uvicorn

    parser = argparse.ArgumentParser(description="RAG-Anything API Server")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument("--host", type=str, help="Host to bind the server to")

    args = parser.parse_args()

    # Override server config if provided
    if args.port:
        srv.port = args.port
    if args.host:
        srv.host = args.host

    params = uvicorn_run_params(srv)
    uvicorn.run("raganything.api.app:app", **params)


if __name__ == "__main__":
    run()
