# -*- coding: utf-8 -*-
"""ModeRouter 多轮上下文（方案 A）单元测试。"""

from __future__ import annotations

import pytest

from app.core.agent.mode_router import ModeRouter
from app.core.intent.recognizer import IntentResult


class _CapturingRouterLLM:
    """记录 LLM 收到的 messages；默认返回 react。"""

    def __init__(self, mode: str = "react") -> None:
        self._mode = mode
        self.calls: list[list[dict[str, str]]] = []

    async def acomplete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return (
            f'{{"mode": "{self._mode}", "rationale": "ok", "confidence": 0.8}}'
        )


@pytest.mark.asyncio
async def test_resolve_passes_history_to_llm_router():
    llm = _CapturingRouterLLM()
    router = ModeRouter(llm=llm, tool_names=["rag_search", "calculator"])
    intent = IntentResult(intent="未知", confidence=0.3)
    decision = await router.resolve(
        "那它呢？",
        intent,
        history=[
            "user: 高血压能吃柚子吗？",
            "assistant: 一般不建议。",
        ],
    )
    assert decision.mode == "react"
    user_msg = llm.calls[0][1]["content"]
    assert "对话历史" in user_msg
    assert "高血压能吃柚子吗" in user_msg


@pytest.mark.asyncio
async def test_resolve_without_history_omits_history_block():
    llm = _CapturingRouterLLM()
    router = ModeRouter(llm=llm, tool_names=[])
    intent = IntentResult(intent="未知", confidence=0.3)
    await router.resolve("今天有什么新闻", intent)
    user_msg = llm.calls[0][1]["content"]
    assert "对话历史" not in user_msg


@pytest.mark.asyncio
async def test_resolve_rule_fast_path_ignores_history():
    llm = _CapturingRouterLLM(mode="plan_execute")
    router = ModeRouter(llm=llm, tool_names=[])
    intent = IntentResult(intent="数学计算", confidence=0.95)
    decision = await router.resolve("帮我算 1+1", intent, history=["user: 你好"])
    assert decision.source == "rule"
    assert decision.mode == "react"
    assert llm.calls == []  # 规则命中后不再调用 LLM
