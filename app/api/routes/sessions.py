# -*- coding: utf-8 -*-
"""会话管理 API：列表、查看、删除。数据来源：PostgreSQL（持久化）+ 内存降级。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.agent_service import AgentService

router = APIRouter(tags=["sessions"])

_agent: AgentService | None = None


def _get_agent() -> AgentService:
    global _agent
    if _agent is None:
        from app.api.routes.chat import _get_agent as _ga

        _agent = _ga()
    return _agent


@router.get("/sessions")
async def list_sessions() -> list[dict]:
    """列出所有会话（按最近更新降序）。"""
    agent = _get_agent()
    return await agent.memory.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """获取会话详情与历史消息。"""
    agent = _get_agent()
    session = await agent.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "id": session["id"],
        "title": session.get("title", "新对话"),
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in session.get("messages", [])
        ],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """删除会话（PG + Redis + 内存三方同步）。"""
    agent = _get_agent()
    ok = await agent.memory.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "deleted"}
