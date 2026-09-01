# -*- coding: utf-8 -*-
"""意图识别：关键词降级 + LLM 分类单元测试。"""

from __future__ import annotations

import pytest

from app.core.intent.recognizer import IntentRecognizer, _keyword_classify


def test_keyword_classify_math():
    intent, child, conf, why = _keyword_classify("帮我计算 123 * 456 等于多少")
    assert intent == "数学计算"
    assert child == "表达式计算"
    assert conf > 0.5


def test_keyword_classify_greeting():
    intent, child, conf, why = _keyword_classify("你好呀")
    assert intent == "闲聊对话"
    assert child == "日常寒暄"


def test_keyword_classify_fallback_low_confidence():
    intent, child, conf, why = _keyword_classify("随便说说")
    assert intent == "知识问答"
    assert child is None
    assert conf < 0.5


@pytest.mark.asyncio
async def test_recognize_falls_back_without_llm():
    rec = IntentRecognizer(llm=None)
    result = await rec.recognize("帮我算一下 1+1")
    assert result.intent == "数学计算"


class _FakeLLM:
    async def acomplete(self, messages, **kwargs) -> str:
        return (
            '{"intent": "实时搜索", "confidence": 0.92, '
            '"rationale": "包含最新资讯关键词", "slots": {}}'
        )


@pytest.mark.asyncio
async def test_recognize_uses_llm_result():
    rec = IntentRecognizer(llm=_FakeLLM())
    result = await rec.recognize("今天有什么最新新闻")
    assert result.intent == "实时搜索"
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_recognize_degrades_when_llm_fails():
    class _BrokenLLM:
        async def acomplete(self, messages, **kwargs) -> str:
            raise RuntimeError("api down")

    rec = IntentRecognizer(llm=_BrokenLLM())
    result = await rec.recognize("你好")
    assert result.intent == "闲聊对话"


class _CapturingLLM:
    """记录 LLM 收到的 messages，便于断言上下文注入。"""

    def __init__(self, result: str) -> None:
        self._result = result
        self.calls: list[list[dict[str, str]]] = []

    async def acomplete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self._result


@pytest.mark.asyncio
async def test_recognize_injects_history_into_llm_prompt():
    llm = _CapturingLLM(
        '{"intent": "知识问答", "confidence": 0.9, '
        '"rationale": "追问延续上一话题", "slots": {}}'
    )
    rec = IntentRecognizer(llm=llm)
    result = await rec.recognize(
        "那它呢？",
        context={
            "history": [
                "user: 高血压能吃柚子吗？",
                "assistant: 一般不建议，柚子会影响降压药代谢。",
            ]
        },
    )
    assert result.intent == "知识问答"
    user_msg = llm.calls[0][1]["content"]
    assert "高血压能吃柚子吗" in user_msg
    assert "当前输入" in user_msg
    assert "那它呢" in user_msg


@pytest.mark.asyncio
async def test_recognize_without_context_keeps_single_turn_prompt():
    llm = _CapturingLLM(
        '{"intent": "闲聊对话", "confidence": 0.9, "rationale": "", "slots": {}}'
    )
    rec = IntentRecognizer(llm=llm)
    await rec.recognize("你好")
    user_msg = llm.calls[0][1]["content"]
    assert "对话历史" not in user_msg
    assert "请对以下输入做意图分类" in user_msg
