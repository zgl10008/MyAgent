"""文件上传校验：统一保护文档入库和 RAG 上传入口。"""

from __future__ import annotations

import re
from pathlib import PurePath

from fastapi import HTTPException


ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown"})
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def validate_upload(
    filename: str | None,
    data: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> str:
    """校验上传文件并返回规范化文件名。"""
    name = (filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if len(name) > 255 or "\x00" in name:
        raise HTTPException(status_code=400, detail="文件名不合法")
    if PurePath(name).name != name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名不能包含路径")
    if not re.fullmatch(r"[\w .()\-一-鿿]+", name, flags=re.UNICODE):
        raise HTTPException(status_code=400, detail="文件名包含不支持的字符")
    suffix = PurePath(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 PDF、TXT、MD 文件")
    if not data:
        raise HTTPException(status_code=400, detail="不能上传空文件")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件大小不能超过 {max_bytes // 1024 // 1024} MB")
    return name
