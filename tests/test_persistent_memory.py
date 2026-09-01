# -*- coding: utf-8 -*-
"""PersistentMemoryManager 长期语义记忆写入（Milvus 接线）测试。"""

from __future__ import annotations

import pytest

from app.core.persistent_memory import PersistentMemoryManager


class _FakePG:
    """最小 PG 仓库替身：只记录行为，不真正落库。"""

    def __init__(self) -> None:
        self.added: list[tuple] = []
        self.add_calls = 0
        self.fail_add = False

    async def add_message(self, session_id, role, content, metadata=None):
        self.add_calls += 1
        if self.fail_add:
            raise ConnectionError("postgres unavailable")
        self.added.append((session_id, role, content))

    async def get_recent_messages(self, session_id, limit=20):
        return []

    async def list_sessions(self):
        return []

    async def get_session(self, session_id):
        return None

    async def delete_session(self, session_id):
        return False


class _FakeLTM:
    """记录 store 调用的长期语义记忆替身。"""

    def __init__(self) -> None:
        self.stored: list[tuple] = []

    async def store(self, session_id, content, metadata=None):
        self.stored.append((session_id, content, metadata or {}))
        return "memory-id"


def _make_memory(ltm: _FakeLTM | None = None, pg: _FakePG | None = None) -> PersistentMemoryManager:
    mem = PersistentMemoryManager(
        redis_client=None,
        compress_llm=None,
        pg_repo=pg or _FakePG(),
    )
    if ltm is not None:
        mem.set_long_term(ltm)
    return mem


@pytest.mark.asyncio
async def test_append_turn_stores_user_and_assistant_to_ltm():
    """每轮 user/assistant 消息都应写入长期语义记忆，并携带元数据。"""
    ltm = _FakeLTM()
    mem = _make_memory(ltm)

    await mem.append_turn("s1", "user", "我最近血压偏高，医生开了氨氯地平", {"k": 1})
    await mem.append_turn("s1", "assistant", "氨氯地平是钙通道阻滞剂", {"trace_id": "t1"})

    assert [(s, c) for s, c, _ in ltm.stored] == [
        ("s1", "我最近血压偏高，医生开了氨氯地平"),
        ("s1", "氨氯地平是钙通道阻滞剂"),
    ]
    assert ltm.stored[0][2] == {"k": 1}


@pytest.mark.asyncio
async def test_append_turn_skips_empty_and_system_for_ltm():
    """system 消息与空内容不进长期语义记忆，避免噪声。"""
    ltm = _FakeLTM()
    mem = _make_memory(ltm)

    await mem.append_turn("s1", "system", "系统消息", {})
    await mem.append_turn("s1", "user", "   ", {})

    assert ltm.stored == []


@pytest.mark.asyncio
async def test_ltm_store_failure_does_not_break_append():
    """Milvus 写入失败只降级，不影响 PG/Redis/内存主流程。"""

    class _BrokenLTM(_FakeLTM):
        async def store(self, session_id, content, metadata=None):
            raise RuntimeError("milvus down")

    ltm = _BrokenLTM()
    mem = _make_memory(ltm)

    await mem.append_turn("s1", "user", "高血压要注意什么", {})

    # 内存兜底仍然写入了该条消息
    history = await mem.get_session_history("s1")
    assert any(m["content"] == "高血压要注意什么" for m in history)


@pytest.mark.asyncio
async def test_pg_write_failure_disables_retries_and_keeps_memory_copy():
    """PG 写入失败后不重复尝试，内存副本仍可读取。"""
    pg = _FakePG()
    pg.fail_add = True
    mem = _make_memory(pg=pg)

    await mem.append_turn("s1", "user", "第一条")
    await mem.append_turn("s1", "assistant", "第二条")

    assert pg.add_calls == 1
    history = await mem.get_session_history("s1")
    assert [item["content"] for item in history] == ["第一条", "第二条"]


    """未注入长期记忆时保持原行为，不抛异常。"""
    mem = _make_memory(ltm=None)
    await mem.append_turn("s1", "user", "你好", {})
    history = await mem.get_session_history("s1")
    assert len(history) == 1
