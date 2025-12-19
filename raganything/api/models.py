# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class HealthResp(BaseModel):
    ok: bool = True


class InfoResp(BaseModel):
    title: str
    description: str
    host: str
    port: int
    cors_origins: List[str] = []
    ssl_enabled: bool = False


class QueryReq(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(
        "hybrid",
        pattern=r"^(local|global|hybrid|naive|mix|bypass)$",
    )
    system_prompt: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    max_tokens: Optional[int] = Field(default=None, ge=64, le=32768)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class QueryResp(BaseModel):
    result: Optional[str] = ""


class MultiModalItem(BaseModel):
    type: str = Field(..., description="image|table|equation|generic")
    img_path: Optional[str] = None
    table_body: Optional[str] = None
    table_data: Optional[str] = None
    latex: Optional[str] = None
    text: Optional[str] = None
    page_idx: Optional[int] = None
    image_caption: Optional[List[str]] = None
    image_footnote: Optional[List[str]] = None
    table_caption: Optional[List[str]] = None
    table_footnote: Optional[List[str]] = None


class QueryMultiReq(BaseModel):
    query: str = Field(..., min_length=1)
    multimodal_content: List[MultiModalItem] = Field(default_factory=list)
    mode: str = Field("hybrid", pattern=r"^(local|global|hybrid|mix|bypass)$")
    top_k: Optional[int] = None


class DocumentProcessParams(BaseModel):
    output_dir: Optional[str] = None
    parse_method: Optional[str] = Field(
        default=None,
        description="auto|ocr|txt",
    )
    display_stats: Optional[bool] = None
    split_by_character: Optional[str] = None
    split_by_character_only: bool = False
    doc_id: Optional[str] = None


class UploadResp(BaseModel):
    doc_id: Optional[str] = None
    file_name: str
    status: str


class ContentListInsertReq(BaseModel):
    content_list: List[MultiModalItem]
    file_path: Optional[str] = "unknown_document"
    split_by_character: Optional[str] = None
    split_by_character_only: bool = False
    doc_id: Optional[str] = None
    display_stats: Optional[bool] = None


class DocStatusResp(BaseModel):
    exists: bool
    text_processed: bool = False
    multimodal_processed: bool = False
    fully_processed: bool = False
    chunks_count: int = 0
    chunks_list: Optional[List[Any]] = None
    status: Optional[str] = None
    updated_at: Optional[str] = None
    raw_status: Optional[Any] = None


class BatchProcessResp(BaseModel):
    batch_id: str
    total_files: int
    status: str


class ConfigResp(BaseModel):
    multimodal: Any
    batch: Any
    parsing: Any


class StatsResp(BaseModel):
    total_documents: int
    processing_queue: int
    storage_usage: str
    average_processing_time: float
