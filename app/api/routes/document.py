# -*- coding: utf-8 -*-
"""文档管理 API：上传与列表。"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.api.upload_validation import validate_upload
from app.api.routes.chat import _get_rag
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
) -> DocumentUploadResponse:
    """统一上传入口：解析、分块、向量写入与元数据持久化。"""
    raw = await file.read()
    safe_name = validate_upload(file.filename, raw)
    try:
        document = await _get_rag().ingest_document(raw, safe_name, file.content_type)
    except Exception as exc:
        logger.exception("文档摄入失败: {}", exc)
        raise HTTPException(status_code=422, detail=f"文档处理失败: {exc!s}") from exc

    return DocumentUploadResponse(
        document_id=document["id"],
        filename=document["filename"],
        status=document["status"],
        chunk_count=document["chunk_count"],
        message="上传、分块并写入知识库成功",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """列出统一知识库中的文档元数据。"""
    return [
        DocumentInfo(
            id=str(doc["id"]), filename=str(doc["filename"]),
            mime_type=doc.get("mime_type"), status=str(doc.get("status", "ready")),
            created_at=str(doc.get("uploaded_at")) if doc.get("uploaded_at") else None,
        )
        for doc in _get_rag().list_documents()
    ]
