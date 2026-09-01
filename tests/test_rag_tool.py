# -*- coding: utf-8 -*-
"""rag_search 工具：空库 / 无结果 / 异常降级文案单元测试。"""

from __future__ import annotations

import pytest

from app.core.tools.builtin.rag_search import RAGSearchTool
from app.models.schemas import RetrievalResult


async def _search_ok(query: str, top_k: int):
    return [
        RetrievalResult(id="c1", content="高血压患者应控制钠盐摄入", score=0.9, source="hybrid")
    ]


async def _search_empty(query: str, top_k: int):
    return []


async def _search_error(query: str, top_k: int):
    raise RuntimeError("milvus down")


@pytest.mark.asyncio
async def test_missing_query_returns_error():
    tool = RAGSearchTool(search_func=_search_ok)
    out = await tool.execute()
    assert "query" in out


@pytest.mark.asyncio
async def test_empty_kb_hints_upload_or_web_search():
    tool = RAGSearchTool(search_func=_search_ok, chunks_count=lambda: 0)
    out = await tool.execute(query="高血压")
    assert "知识库当前为空" in out
    assert "web_search" in out


@pytest.mark.asyncio
async def test_empty_results_suggest_alternatives():
    tool = RAGSearchTool(search_func=_search_empty, chunks_count=lambda: 10)
    out = await tool.execute(query="罕见病")
    assert "未找到" in out
    assert "web_search" in out


@pytest.mark.asyncio
async def test_exception_returns_degraded_message():
    tool = RAGSearchTool(search_func=_search_error, chunks_count=lambda: 10)
    out = await tool.execute(query="高血压")
    assert "检索失败" in out


@pytest.mark.asyncio
async def test_success_returns_formatted_results():
    tool = RAGSearchTool(search_func=_search_ok, chunks_count=lambda: 10)
    out = await tool.execute(query="高血压饮食", top_k=5)
    assert "高血压患者应控制钠盐摄入" in out


@pytest.mark.asyncio
async def test_top_k_clamped_to_1_20():
    captured = {}

    async def _capture(query: str, top_k: int):
        captured["top_k"] = top_k
        return []

    tool = RAGSearchTool(search_func=_capture, chunks_count=lambda: 10)
    await tool.execute(query="x", top_k=999)
    assert captured["top_k"] == 20
