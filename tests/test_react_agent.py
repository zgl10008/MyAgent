# -*- coding: utf-8 -*-
"""ReAct 解析与循环单元测试。"""

from __future__ import annotations

import pytest

from app.core.agent.react_agent import ReActAgent, _parse_react_step


def test_parse_full_react_step():
    text = (
        'Thought: 需要计算\n'
        'Action: calculator\n'
        'Action Input: {"expression": "1+1"}\n'
        "Observation: 2"
    )
    out = _parse_react_step(text)
    assert out["thought"] == "需要计算"
    assert out["action"] == "calculator"
    assert out["action_input"] == {"expression": "1+1"}
    assert "done" not in out


def test_parse_final_answer_marks_done():
    out = _parse_react_step("Thought: 想好了\nFinal Answer: 答案是 42")
    assert out["done"] is True
    assert out["final_answer"] == "答案是 42"


def test_parse_multiline_final_answer():
    out = _parse_react_step("Thought: 开始\nFinal Answer: 第一行\n第二行")
    assert out["done"] is True
    assert out["final_answer"] == "第一行\n第二行"


def test_parse_invalid_json_marks_parse_error():
    text = "Action: search\nAction Input: {not-json}"
    out = _parse_react_step(text)
    assert out["action"] == "search"
    assert out["action_input"] == {}
    assert "parse_error" in out


def test_parse_without_action():
    out = _parse_react_step("Thought: 直接回答")
    assert "action" not in out
    assert "done" not in out
    assert out["raw"] == "Thought: 直接回答"


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def acomplete(self, messages, **kwargs) -> str:
        self.calls += 1
        return self._responses.pop(0)


class _FakeTools:
    async def invoke(self, name: str, arguments: dict) -> str:
        return "42"


class _FakeMemory:
    async def get_relevant(self, session_id: str, query: str, limit: int = 8):
        return []

    async def append_turn(self, session_id, role, content, metadata=None):
        pass


class _RecordingMemory(_FakeMemory):
    def __init__(self) -> None:
        self.writes: list[tuple] = []

    async def append_turn(self, session_id, role, content, metadata=None):
        self.writes.append((session_id, role, content, metadata))


@pytest.mark.asyncio
async def test_react_loop_ends_on_final_answer():
    agent = ReActAgent(
        llm=_FakeLLM(["Final Answer: 42"]),
        tools=_FakeTools(),
        memory=_FakeMemory(),
        max_steps=3,
    )
    result = await agent.run("1+1 等于多少", {"tool_names": ["calculator"]})
    assert result.success is True
    assert result.final_answer == "42"


@pytest.mark.asyncio
async def test_react_loop_calls_tool_then_final():
    agent = ReActAgent(
        llm=_FakeLLM(
            [
                'Thought: 先算\nAction: calculator\nAction Input: {"expression": "1+1"}',
                "Final Answer: 结果是 2",
            ]
        ),
        tools=_FakeTools(),
        memory=_FakeMemory(),
        max_steps=3,
    )
    result = await agent.run("算一下 1+1", {"tool_names": ["calculator"]})
    assert result.success is True
    assert result.final_answer == "结果是 2"
    assert len(result.steps) >= 1


@pytest.mark.asyncio
async def test_react_loop_does_not_write_memory_internally():
    """记忆写入统一由 agent_service.chat 负责，避免与服务层重复写入。"""
    mem = _RecordingMemory()
    agent = ReActAgent(
        llm=_FakeLLM(["Final Answer: 42"]),
        tools=_FakeTools(),
        memory=mem,
        max_steps=3,
    )
    result = await agent.run("1+1 等于多少", {"tool_names": ["calculator"]})
    assert result.success is True
    assert mem.writes == []
