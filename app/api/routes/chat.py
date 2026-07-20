# -*- coding: utf-8 -*-
"""对话 API：非流式与流式输出。"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.agent_service import AgentService
from app.core.intent.recognizer import IntentRecognizer
from app.core.rag_service import RAGService
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.tracer import Tracer
from app.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    RAGChatRequest,
    RAGChatResponse,
    RAGIngestResponse,
)

router = APIRouter(tags=["chat"])

_tracer = Tracer()
_intent = IntentRecognizer()

# Agent 服务（懒加载）
_agent_service: AgentService | None = None


def _get_agent() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


def _build_router() -> ModelRouter:
    """根据配置构造模型路由器。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY，无法调用模型")
    cfg = ModelConfig(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        priority=0,
        weight=1.0,
    )
    return ModelRouter([cfg])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """非流式对话：经意图识别与链路追踪后返回完整回复。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")

    try:
        user_text = request.messages[-1].content if request.messages else ""
        intent = await _intent.recognize(user_text)
        clarify = await _intent.clarify(user_text, intent)
        logger.info(
            "意图 intent={} conf={} sub={}",
            intent.intent,
            intent.confidence,
            intent.sub_intent,
        )

        router_llm = _build_router()
        messages = [m.model_dump() for m in request.messages]
        if clarify and intent.confidence < _intent.confidence_threshold:
            messages.append(
                {
                    "role": "system",
                    "content": f"（系统提示：{clarify}）",
                }
            )

        resp = await router_llm.chat(
            messages,
            model_preference=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        _tracer.end_span(
            span,
            result={"model": resp.model_id, "usage": resp.usage},
        )

        return ChatResponse(
            id=str(uuid.uuid4()),
            model=resp.model_id,
            content=resp.content,
            trace_id=trace_id,
            usage=resp.usage,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"对话失败: {exc!s}") from exc


async def _stream_generator(
    request: ChatRequest,
    trace_id: str,
) -> AsyncIterator[bytes]:
    """SSE 风格流：每行 data: {json}\\n\\n。"""
    settings = get_settings()
    if not settings.openai_api_key:
        yield b"data: " + json.dumps({"error": "未配置 API Key"}, ensure_ascii=False).encode() + b"\n\n"
        return

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
    )
    messages = [m.model_dump() for m in request.messages]
    model = request.model or settings.openai_model

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                payload = {"content": delta, "trace_id": trace_id}
                yield b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
        yield b"data: " + json.dumps({"done": True}, ensure_ascii=False).encode() + b"\n\n"
    except Exception as exc:
        logger.exception("chat_stream 失败: {}", exc)
        yield b"data: " + json.dumps({"error": str(exc)}, ensure_ascii=False).encode() + b"\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式输出（Server-Sent Events 兼容格式）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_stream")
    _tracer.end_span(span, result={"mode": "stream"})

    return StreamingResponse(
        _stream_generator(request, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Trace-Id": trace_id,
        },
    )


@router.post("/chat/agent", response_model=AgentChatResponse)
async def chat_agent(request: AgentChatRequest) -> AgentChatResponse:
    """Agent 模式对话：经意图识别 → Agent 编排 → ReAct/Plan 循环 → 工具调用 → 反思。"""
    agent = _get_agent()
    try:
        return await agent.chat(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("agent_chat 失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"Agent 对话失败: {exc!s}") from exc


# ==============================================================================
# RAG 路由
# ==============================================================================

_rag_service: RAGService | None = None


def _get_rag() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post("/chat/rag", response_model=RAGChatResponse)
async def chat_rag(request: RAGChatRequest) -> RAGChatResponse:
    """RAG 增强对话：检索知识库 → 生成带引用的回答。"""
    rag = _get_rag()
    try:
        query = request.query or request.messages[-1].content
        history = [m.model_dump() for m in request.messages[:-1]]

        resp = await rag.chat(
            query=query,
            chat_history=history,
            mode=request.mode,
            top_k=request.top_k,
        )

        return RAGChatResponse(
            answer=resp.answer,
            citations=resp.citations,
            contexts=resp.raw_contexts,
            model=resp.model or "",
            total_chunks=rag.chunks_count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("rag_chat 失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"RAG 对话失败: {exc!s}") from exc


@router.post("/rag/ingest", response_model=RAGIngestResponse)
async def rag_ingest(
    file: UploadFile = File(..., description="上传的文件"),
) -> RAGIngestResponse:
    """上传文档到 RAG 知识库（解析 → 分块 → 向量化 → 入库）。"""
    rag = _get_rag()
    try:
        data = await file.read()
        count = await rag.ingest_file(
            data=data,
            filename=file.filename or "unnamed",
            mime_type=file.content_type,
        )
        return RAGIngestResponse(
            chunk_count=count,
            message=f"已处理 {count} 个文本块",
        )
    except Exception as exc:
        logger.exception("rag_ingest 失败: {}", exc)
        raise HTTPException(status_code=422, detail=f"文档处理失败: {exc!s}") from exc


@router.get("/rag/status")
async def rag_status() -> dict[str, Any]:
    """RAG 知识库状态。"""
    rag = _get_rag()
    return {
        "total_chunks": rag.chunks_count,
        "status": "ready" if rag.chunks_count > 0 else "empty",
    }


@router.get("/rag/documents")
async def rag_list_documents() -> list[dict[str, Any]]:
    """列出 RAG 知识库中的文档。"""
    rag = _get_rag()
    return rag.list_documents()


@router.delete("/rag/documents/{filename:path}")
async def rag_delete_document(filename: str) -> dict[str, Any]:
    """从 RAG 知识库删除文档。"""
    rag = _get_rag()
    ok = rag.delete_document(filename)
    if not ok:
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")
    return {"status": "deleted", "filename": filename}


class SearchCompareRequest(BaseModel):
    """搜索对比请求。"""
    query: str = Field(..., min_length=1, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="每模式返回数量")


@router.post("/rag/search_compare")
async def rag_search_compare(req: SearchCompareRequest) -> dict[str, Any]:
    """三种检索模式并行对比：向量 / 关键词 / 混合。"""
    rag = _get_rag()
    query = req.query
    top_k = req.top_k

    import asyncio, time

    results: dict[str, Any] = {"query": query, "top_k": top_k, "total_chunks": rag.chunks_count}

    # 并行执行三种检索模式
    t0 = time.time()
    vec_task = rag.search(query, top_k=top_k, mode="vector")
    kw_task = rag.search(query, top_k=top_k, mode="keyword")
    hyb_task = rag.search(query, top_k=top_k, mode="hybrid")
    vec, kw, hyb = await asyncio.gather(vec_task, kw_task, hyb_task)
    elapsed = time.time() - t0

    def _fmt(results_list: list) -> list[dict]:
        return [
            {
                "rank": i + 1,
                "id": r.id,
                "content": r.content[:200],
                "score": round(r.score, 4),
                "source": r.source,
            }
            for i, r in enumerate(results_list)
        ]

    results["elapsed_seconds"] = round(elapsed, 3)
    results["vector"] = _fmt(vec)
    results["keyword"] = _fmt(kw)
    results["hybrid"] = _fmt(hyb)

    return results
