# -*- coding: utf-8 -*-
"""
Function Calling Agent：基于 LangChain ChatOpenAI.bind_tools() 的 ReAct Agent。

与 react_agent.py 对比：
┌──────────────────────┬─────────────────────┬─────────────────────────┐
│ 维度                 │ react_agent (自研)   │ function_call_agent     │
├──────────────────────┼─────────────────────┼─────────────────────────┤
│ 工具调用方式         │ 正则解析文本         │ Function Calling (JSON) │
│ 模型依赖             │ 任意 LLM             │ 需支持 function calling │
│ 解析失败率           │ ~10-15% (JSON 格式)  │ ~0% (结构化原生)        │
│ 编排控制             │ 自研 for 循环        │ 自研 for 循环 ← 相同    │
│ 步数上限 / 记忆 / 历史│ 自研                 │ 自研 ← 相同             │
└──────────────────────┴─────────────────────┴─────────────────────────┘

设计理念：LangChain 只负责"让模型输出结构化工具调用"这一件事，
编排逻辑（循环、步数上限、历史管理、错误处理）全部自研。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

# 与 react_agent 共用 AgentResult 与 MemoryLike 协议
from app.core.agent.react_agent import AgentResult, MemoryLike

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
FC_SYSTEM_PROMPT = """你是一个严谨的智能助手。需要外部信息时调用可用工具；不需要时直接回答。
每次工具调用后结合 Observation 判断是否已有足够信息给出最终答案。使用中文回答。"""


def _build_messages(
    query: str,
    system_text: str,
    history: list[BaseMessage],
    mem_snippets: list[str],
) -> list[BaseMessage]:
    mem_block = "\n".join(f"- {s}" for s in mem_snippets) if mem_snippets else "（无）"
    full_system = system_text + f"\n\n## 检索记忆摘要\n{mem_block}"
    return [SystemMessage(content=full_system), HumanMessage(content=query), *history]


def _extract_token_usage(response: AIMessage) -> dict[str, int] | None:
    """从 LangChain AIMessage.response_metadata 提取 token 用量。

    兼容不同 provider/版本把 token 放在 ``token_usage`` 或 ``usage`` 下，
    字段名可能是 snake_case 或 camelCase；三者任一有效即返回。
    """
    md = getattr(response, "response_metadata", None) or {}
    if not isinstance(md, dict):
        return None
    tu = md.get("token_usage") or md.get("usage") or md.get("tokenUsage")
    if not isinstance(tu, dict):
        return None
    pt = tu.get("prompt_tokens") or tu.get("promptTokens") or 0
    ct = tu.get("completion_tokens") or tu.get("completionTokens") or 0
    tt = tu.get("total_tokens") or tu.get("totalTokens") or (int(pt) + int(ct))
    return {
        "prompt_tokens": int(pt),
        "completion_tokens": int(ct),
        "total_tokens": int(tt),
    }


# ---------------------------------------------------------------------------
# Function Calling Agent
# ---------------------------------------------------------------------------
class FunctionCallAgent:
    """
    基于 LangChain ChatOpenAI + Function Calling 的 ReAct Agent。

    LangChain 负责：ChatOpenAI 模型调用 + bind_tools() 注册工具 Schema → 模型返回结构化 tool_calls
    自研负责：for 循环控制、步数上限、历史消息管理、记忆读写、错误降级
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        tools: Sequence[Any] = (),
        max_steps: int = 10,
        memory: Optional[MemoryLike] = None,
        usage_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._all_tools = list(tools)
        self._tools = {t.name: t for t in tools} if tools else {}
        self.max_steps = max(1, max_steps)
        self._memory = memory
        self._usage_callback = usage_callback
        self._llm: Any = None

    def set_tools(self, tools: Sequence[Any]) -> None:
        """Replace the tools exposed through Function Calling.

        Some tools (notably ``rag_search``) are registered after the agent is
        constructed.  Keeping the binding refreshable prevents the FC backend
        from holding a stale snapshot of the registry.
        """
        self._all_tools = list(tools)
        self._tools = {tool.name: tool for tool in self._all_tools}

    def _bind_llm(self, tools: Sequence[Any]) -> Any:
        """按给定工具子集构造 bind_tools 的 LLM。

        每次 run 依据 context["tool_names"] 过滤后再 bind，确保 FC 后端
        尊重编排器为当前意图和子任务生成的工具白名单。
        """
        if self._llm is not None:
            return self._llm
        return ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=0.2,
        ).bind_tools(list(tools))

    def _build_system(self, tool_descriptions: str, extra_system: str) -> str:
        prompt = FC_SYSTEM_PROMPT
        if tool_descriptions:
            prompt += f"\n\n## 可用工具\n{tool_descriptions}"
        if extra_system:
            prompt += f"\n\n{extra_system}"
        return prompt

    async def _invoke_tool(self, name: str, args: Dict[str, Any], tool_map: Dict[str, Any]) -> str:
        """执行单个工具调用并返回字符串结果。"""
        tool = tool_map.get(name)
        if tool is None:
            return f"错误：工具 [{name}] 未注册"
        try:
            result = await tool.ainvoke(args) if hasattr(tool, "ainvoke") else tool.invoke(args)
            return str(result)
        except Exception as exc:
            return f"工具 [{name}] 执行异常: {exc}"

    async def run(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        执行 Function Calling ReAct 循环。

        context 可包含：
        - tool_descriptions: 工具描述文本（写入 system prompt）
        - extra_system: 额外系统提示（意图、槽位等）
        - session_id: 会话 ID
        - trace_callback: 可选 async 回调(step_dict)
        """
        trace_cb = context.get("trace_callback")
        tool_descriptions = str(context.get("tool_descriptions", ""))
        extra_system = str(context.get("extra_system", ""))
        system_text = self._build_system(tool_descriptions, extra_system)

        # 依据 context["tool_names"] 过滤本次可用工具，保证每个意图或
        # Plan-Execute 子任务只能调用编排器分配的工具。
        tool_names = context.get("tool_names")
        if tool_names:
            allowed = set(tool_names)
            run_tools = [t for t in self._all_tools if t.name in allowed]
            tool_map = {t.name: t for t in run_tools}
        else:
            run_tools = self._all_tools
            tool_map = self._tools
        llm = self._bind_llm(run_tools)

        # 会话记忆：与 ReActAgent 一致，在循环开始前读取最近历史并注入 system prompt。
        # 修复点：此前此处硬编码空列表，导致 Function Calling 后端完全丢失多轮上下文
        # （表现为"这是我们的第一次交流/前面提问 0 个问题"）。
        session_id = str(context.get("session_id", "default"))
        mem_snippets: list[str] = []
        if self._memory is not None:
            try:
                mem_snippets = await self._memory.get_relevant(session_id, query, limit=8)
            except Exception as exc:
                logger.warning("读取记忆失败，将以无历史继续: %s", exc)

        steps: List[Dict[str, Any]] = []
        history: list[BaseMessage] = []

        for step_idx in range(self.max_steps):
            messages = _build_messages(query, system_text, history, mem_snippets)

            # ── 1. 调用 LLM（bind_tools 已注入 Schema，模型自动决定是否调用工具） ──
            try:
                response: AIMessage = await llm.ainvoke(messages)  # type: ignore[assignment]
                if self._usage_callback is not None:
                    _u = _extract_token_usage(response)
                    if _u:
                        self._usage_callback(_u)
            except Exception as exc:
                logger.exception("LLM 调用失败 step=%d", step_idx)
                return AgentResult(
                    success=False, final_answer="",
                    steps=steps, error=f"LLM 调用失败: {exc}",
                )

            # ── 2. 有 tool_calls → 执行工具 ──
            if response.tool_calls:
                _tc_names = [str(tc.get("name", "")) for tc in response.tool_calls]
                rec: Dict[str, Any] = {
                    "step": step_idx,
                    "phase": "react",
                    "thought": str(response.content or "")[:2000],
                    "action": _tc_names[0] if _tc_names else "",
                    "action_input": response.tool_calls[0].get("args", {}),
                    "tools": _tc_names,
                }

                # 修复点：assistant 消息只能追加一次，随后为每个 tool_call 追加一条
                # ToolMessage（tool_call_id 一一对应）。此前在循环内重复追加带全部
                # tool_calls 的 assistant 消息，多工具调用场景下下一轮请求会被 API
                # 拒绝（"must be followed by tool messages responding to each
                # tool_call_id"），表现为执行到步骤 1 后卡住。
                history.append(AIMessage(
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {}) or {}
                    obs = await self._invoke_tool(tool_name, tool_args, tool_map)
                    obs_short = obs[:2000]

                    # ToolMessage 必须带 tool_call_id，LangChain 用它关联 AIMessage.tool_calls
                    history.append(ToolMessage(
                        content=obs_short,
                        tool_call_id=tc.get("id", f"call_{step_idx}"),
                    ))

                rec["observation"] = obs_short
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                continue

            # ── 3. 无 tool_calls → 最终答案 ──
            answer = str(response.content or "")
            rec = {
                "step": step_idx,
                "phase": "react",
                "thought": answer[:2000],
                "final": True,
            }
            steps.append(rec)
            if trace_cb:
                await trace_cb(rec)
            return AgentResult(success=True, final_answer=answer, steps=steps)

        # 耗尽步数
        return AgentResult(
            success=False, final_answer="",
            steps=steps,
            error=f"已达到最大步数限制 ({self.max_steps})，未得到最终答案。",
        )
