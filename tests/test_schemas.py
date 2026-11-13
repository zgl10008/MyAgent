# -*- coding: utf-8 -*-
"""Pydantic 模型校验单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.enums import MessageRole
from app.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatRequest,
    Message,
    RAGChatRequest,
    RetrievalResult,
)


def test_chat_request_requires_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_request_temperature_bounds():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hi"}], temperature=3.0)


def test_agent_chat_response_defaults():
    resp = AgentChatResponse(id="r1", model="gpt-4o-mini", content="你好")
    assert resp.agent_mode == "react"
    assert resp.tool_calls == []
    assert resp.degraded is False


def test_message_role_enum_validation():
    m = Message(role=MessageRole.USER, content="hi")
    assert m.role.value == "user"
    with pytest.raises(ValidationError):
        Message(role="invalid", content="hi")


def test_retrieval_result_defaults():
    r = RetrievalResult(id="c1", content="内容")
    assert r.score == 0.0
    assert r.source == "vector"
    assert r.metadata == {}


def test_rag_chat_request_top_k_bounds():
    with pytest.raises(ValidationError):
        RAGChatRequest(messages=[{"role": "user", "content": "hi"}], top_k=100)


def test_rag_chat_request_accepts_conversation_id():
    request = RAGChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        conversation_id="session-1",
    )
    assert request.conversation_id == "session-1"
