# -*- coding: utf-8 -*-
"""短期记忆（滑动窗口 + 摘要压缩触发）单元测试。"""

from __future__ import annotations

import pytest

from app.core.memory.short_term import ShortTermMemory
from app.models.enums import MessageRole
from app.models.schemas import Message
from tests.conftest import FakeLLM, FakeRedis


def _msg(role: str, content: str) -> Message:
    return Message(role=MessageRole(role), content=content)


@pytest.mark.asyncio
async def test_serialize_roundtrip():
    stm = ShortTermMemory(redis_client=None, llm=None)
    m = _msg("user", "高血压怎么控制")
    raw = stm._serialize(m)
    restored = stm._deserialize(raw)
    assert restored.role == MessageRole.USER
    assert restored.content == "高血压怎么控制"


@pytest.mark.asyncio
async def test_add_and_get_history_in_order():
    redis = FakeRedis()
    stm = ShortTermMemory(redis_client=redis, llm=None)
    await stm.add_message("s1", _msg("user", "问题一"))
    await stm.add_message("s1", _msg("assistant", "回答一"))
    history = await stm.get_history("s1")
    assert [m.content for m in history] == ["问题一", "回答一"]


@pytest.mark.asyncio
async def test_compression_triggers_over_window_size():
    redis = FakeRedis()
    stm = ShortTermMemory(redis_client=redis, llm=FakeLLM(), window_size=4, max_tokens=10000)
    for i in range(6):
        await stm.add_message("s1", _msg("user", f"消息{i}"))
    history = await stm.get_history("s1")
    # 压缩后：1 条摘要 + 保留尾部 keep=2 条
    assert any("[历史摘要]" in m.content for m in history)
    assert len(history) <= 4


def test_token_count_fallback_without_encoding():
    stm = ShortTermMemory(redis_client=None, llm=None)
    stm._encoding = None
    assert stm._count_tokens("你好") == 1


@pytest.mark.asyncio
async def test_history_isolated_per_session():
    redis = FakeRedis()
    stm = ShortTermMemory(redis_client=redis, llm=None)
    await stm.add_message("s1", _msg("user", "A"))
    await stm.add_message("s2", _msg("user", "B"))
    assert len(await stm.get_history("s1")) == 1
    assert len(await stm.get_history("s2")) == 1
