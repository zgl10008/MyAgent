# -*- coding: utf-8 -*-
"""FunctionCall Agent 会话记忆注入的回归测试。"""

from __future__ import annotations

import pytest

from app.core.agent.function_call_agent import FunctionCallAgent


class _FakeResponse:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeInvokeLLM:
    """模拟 ChatOpenAI.bind_tools() 的 ainvoke 返回，记录每次的 messages。"""

    def __init__(self) -> None:
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return _FakeResponse("直接回答：前面问过 2 个问题")


class _FakeMemory:
    def __init__(self, snippets: list[str]) -> None:
        self._snippets = snippets

    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> list[str]:
        return self._snippets

    async def append_turn(self, session_id, role, content, metadata=None):
        pass


@pytest.mark.asyncio
async def test_fc_agent_injects_session_history_into_system_prompt():
    """回归：此前 _build_messages 硬编码空记忆，模型永远看不到历史。"""
    agent = FunctionCallAgent(
        model_name="test-model",
        api_key="test-key",
        base_url="http://localhost:1/v1",
        tools=[],
        max_steps=3,
        memory=_FakeMemory(
            [
                "user: 你是什么大模型",
                "assistant: 我是 DeepSeek 最新版本模型……",
                "user: 同类型公司有哪些",
            ]
        ),
    )
    fake_llm = _FakeInvokeLLM()
    agent._llm = fake_llm  # 替换真实 ChatOpenAI，离线验证记忆注入逻辑

    result = await agent.run(
        "我前面提问了几个问题?",
        {"session_id": "s1", "tool_descriptions": "", "extra_system": ""},
    )

    assert result.success is True
    assert len(fake_llm.calls) == 1
    system_text = str(fake_llm.calls[0][0].content)
    assert "检索记忆摘要" in system_text
    assert "你是什么大模型" in system_text
    assert "同类型公司有哪些" in system_text


@pytest.mark.asyncio
async def test_fc_agent_without_memory_still_works():
    """无记忆时保持原有单轮行为，不抛异常。"""
    agent = FunctionCallAgent(
        model_name="test-model",
        api_key="test-key",
        base_url="http://localhost:1/v1",
        tools=[],
        max_steps=3,
        memory=None,
    )
    fake_llm = _FakeInvokeLLM()
    agent._llm = fake_llm

    result = await agent.run("你好", {"session_id": "s1"})
    assert result.success is True
    assert result.final_answer == "直接回答：前面问过 2 个问题"


class _FakeToolResp:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _SequencedLLM:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_fc_agent_multiple_tool_calls_builds_valid_history():
    """回归：一次返回多个 tool_calls 时，assistant 消息只能追加一次，
    且每个 tool_call 必须有一条对应 tool_call_id 的 ToolMessage，
    否则下一轮请求会被 API 拒绝（400），表现为执行到步骤 1 后卡住。"""
    from langchain_core.messages import AIMessage, ToolMessage

    agent = FunctionCallAgent(
        model_name="test-model",
        api_key="test-key",
        base_url="http://localhost:1/v1",
        tools=[],
        max_steps=5,
        memory=None,
    )
    fake_llm = _SequencedLLM(
        [
            _FakeToolResp(
                "",
                [
                    {"id": "call_1", "name": "web_search", "args": {"query": "A"}},
                    {"id": "call_2", "name": "web_search", "args": {"query": "B"}},
                ],
            ),
            _FakeToolResp("最终答案"),
        ]
    )
    agent._llm = fake_llm

    async def fake_invoke_tool(name: str, args: dict) -> str:
        return f"obs:{name}:{args.get('query')}"

    agent._invoke_tool = fake_invoke_tool

    result = await agent.run("问题", {"session_id": "s1"})
    assert result.success is True
    assert result.final_answer == "最终答案"

    # 第二轮请求的消息结构：[System, Human, AIMessage(tool_calls), ToolMessage x2]
    messages = fake_llm.calls[1]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(ai_messages) == 1
    assert len(tool_messages) == 2
    assert [t.tool_call_id for t in tool_messages] == ["call_1", "call_2"]
    assert [c["name"] for c in ai_messages[0].tool_calls] == ["web_search", "web_search"]
