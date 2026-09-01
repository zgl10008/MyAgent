# -*- coding: utf-8 -*-
"""
LangGraph ReAct Agent：使用 LangGraph StateGraph + LangChain ChatOpenAI 实现。
与 react_agent.py 的自研实现互备，展示对工业级 Agent 框架的掌握。

图结构：
  START → agent → (有 tool_calls?) → tools ──→ agent
                → (无 tool_calls)  → evaluator
                                        ├── 达标 或 重试≥2 → END
                                        └── 不达标 → agent（带改进反馈）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
QUALITY_THRESHOLD = 60


# ---------------------------------------------------------------------------
# State & Result
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """LangGraph ReAct 状态：消息列表累加 + 重试计数。"""
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: int  # 评估不达标重试次数（scalar 默认 last-write-wins）


@dataclass
class AgentResult:
    """与 ReActAgent.AgentResult 保持兼容。"""
    success: bool
    final_answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    trace_id: Optional[str] = None


# ---------------------------------------------------------------------------
# System Prompt (复用与自研版一致的 ReAct 思维链约束)
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT = """你是一个严谨的智能助手，必须使用 ReAct（推理+行动）方式回答问题。

## 规则
- 需要外部信息时调用可用工具；不需要时可直接回答。
- 每次工具调用后结合 Observation 判断是否已有足够信息给出最终答案。
- 不要编造工具名称或 Observation。
- 使用中文回答。
"""


EVAL_SYSTEM_PROMPT = """你是严格的输出质量审查员。对以下回答从正确性、完整性、清晰度三个维度评分。

必须只输出一个 JSON 对象：
{"score": <0-100的整数>, "is_complete": <true/false>, "feedback": "<一句中文改进建议>"}

评分标准：
- 90-100: 完全正确且完整
- 75-89: 基本正确，有小瑕疵
- 60-74: 存在明显缺失或表述不清
- 0-59: 错误或严重不完整，需重写
"""


def _build_system_prompt(tool_descriptions: str, extra_system: str = "") -> str:
    prompt = REACT_SYSTEM_PROMPT
    if tool_descriptions:
        prompt += f"\n\n## 可用工具\n{tool_descriptions}"
    if extra_system:
        prompt += f"\n\n{extra_system}"
    return prompt


# ---------------------------------------------------------------------------
# LangGraph ReAct Agent
# ---------------------------------------------------------------------------
class LangGraphReActAgent:
    """
    基于 LangGraph StateGraph 构建的 ReAct Agent，含 Evaluator 质量评估节点。

    架构：
      START → agent → (有 tool_calls?) → tools ──→ agent
                    → (无 tool_calls)  → evaluator
                                            ├── score≥60 或 retries≥2 → END
                                            └── score<60 → agent（注入改进反馈）

    对比自研版（react_agent.py）：
    - 自研：纯正则解析 Thought/Action/Final Answer，for 循环控制
    - LangGraph：StateGraph 状态机 + ToolNode + Evaluator + 条件重试
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        tools: Sequence[Any] = (),
        max_steps: int = 10,
    ):
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )
        self._tools = list(tools)
        self._max_steps = max_steps
        self._graph = self._build_graph()

    def _build_graph(self):
        """构建含 Evaluator 的 LangGraph StateGraph。"""
        tool_node = ToolNode(self._tools)

        # ── 节点 1: Agent（LLM 推理 + 工具调用决策） ──
        def call_model(state: AgentState) -> dict:
            response = self._llm.invoke(state["messages"])
            return {"messages": [response]}

        # ── 条件边: 有 tool_calls → tools，否则 → evaluator ──
        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return "evaluator"

        # ── 节点 2: Evaluator（质量评估） ──
        def evaluate(state: AgentState) -> dict:
            # 取最后一条非工具调用的 AI 回答
            answer = ""
            for m in reversed(state["messages"]):
                if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                    answer = str(m.content)
                    # 跳过评估消息自身
                    if "[质量评估:" in answer or answer.startswith("[评估"):
                        continue
                    break

            if not answer:
                return {
                    "messages": [AIMessage(content="[评估: 跳过（无有效回答）]")],
                    "retry_count": state.get("retry_count", 0) + 1,
                }

            eval_messages = [
                SystemMessage(content=EVAL_SYSTEM_PROMPT),
                HumanMessage(content=f"回答:\n{answer[:3000]}"),
            ]

            try:
                raw = self._llm.invoke(eval_messages)
                data = json.loads(str(raw.content))
                score = int(data.get("score", 50))
                feedback = str(data.get("feedback", ""))
                is_complete = bool(data.get("is_complete", True))
            except Exception:
                score, feedback, is_complete = 50, "", True

            retry_count = state.get("retry_count", 0) + 1
            logger.info(
                "Evaluator: score=%d complete=%s retry=%d/%d feedback=%s",
                score, is_complete, retry_count, MAX_RETRIES, feedback[:80],
            )

            eval_msg = f"[质量评估: {score}/100]" + (f" 建议: {feedback}" if feedback else "")
            result: dict = {
                "messages": [AIMessage(content=eval_msg)],
                "retry_count": retry_count,
            }

            # 不达标且未耗尽重试 → 注入改进提示
            if score < QUALITY_THRESHOLD and retry_count <= MAX_RETRIES:
                hint = HumanMessage(
                    content=f"[系统] 上一轮回答质量评分 {score}/100（不达标），"
                    f"改进建议: {feedback}\n请根据建议重新回答。"
                )
                result["messages"].append(hint)

            return result

        # ── 条件边: evaluator 判断是否需要重试 ──
        def should_retry(state: AgentState) -> str:
            retry_count = state.get("retry_count", 0)
            for m in reversed(state["messages"]):
                msg_str = str(m.content)
                if "[质量评估:" in msg_str:
                    score_match = re.search(r'质量评估:\s*(\d+)/100', msg_str)
                    if score_match:
                        score = int(score_match.group(1))
                        if score < QUALITY_THRESHOLD and retry_count <= MAX_RETRIES:
                            return "agent"
                    break
            return END

        # ── 组装图 ──
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_node("evaluator", evaluate)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "evaluator": "evaluator"})
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges("evaluator", should_retry, {"agent": "agent", END: END})

        return workflow.compile()

    def _collect_steps(self, messages: list[BaseMessage]) -> list[dict]:
        """从 LangGraph 消息历史提取步骤记录（供前端可视化）。"""
        steps: list[dict] = []
        step_idx = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = str(msg.content or "")
                # evaluator 消息
                if "[质量评估:" in content:
                    score_match = re.search(r'质量评估:\s*(\d+)/100', content)
                    steps.append({
                        "step": step_idx,
                        "phase": "evaluator",
                        "thought": content,
                        "reflection_score": int(score_match.group(1)) if score_match else None,
                    })
                    step_idx += 1
                    continue
                # 正常 agent 消息
                rec: dict = {"step": step_idx, "phase": "react", "thought": content}
                if msg.tool_calls:
                    rec["action"] = msg.tool_calls[0].get("name", "")
                    rec["action_input"] = msg.tool_calls[0].get("args", {})
                steps.append(rec)
                step_idx += 1
            elif isinstance(msg, ToolMessage):
                if steps:
                    steps[-1]["observation"] = str(msg.content)[:2000]
        return steps

    async def run(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        执行 LangGraph ReAct 循环。context 可包含：
        - tool_descriptions: 工具描述文本
        - extra_system: 额外系统提示
        """
        tool_descriptions = str(context.get("tool_descriptions", ""))
        extra_system = str(context.get("extra_system", ""))
        system_text = _build_system_prompt(tool_descriptions, extra_system)

        initial_messages: list[BaseMessage] = [
            SystemMessage(content=system_text),
            HumanMessage(content=query),
        ]

        config = {"configurable": {"thread_id": context.get("session_id", "default")},
                  "recursion_limit": self._max_steps * 3 + 3}  # agent+eval+tool per cycle

        try:
            result = await self._graph.ainvoke(
                {"messages": initial_messages, "retry_count": 0},
                config,
            )
            messages: list[BaseMessage] = result["messages"]
            steps = self._collect_steps(messages)

            # 提取最终答案：最后一条非 evaluator 的 AI 消息
            final_answer = ""
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                    msg_text = str(m.content)
                    if "[质量评估:" in msg_text:
                        continue  # 跳过 evaluator 的内部消息
                    final_answer = msg_text
                    break

            if not final_answer:
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=steps,
                    error="LangGraph Agent 未生成最终答案",
                )

            return AgentResult(success=True, final_answer=final_answer, steps=steps)

        except Exception as e:  # noqa: BLE001
            logger.exception("LangGraph Agent 执行失败")
            return AgentResult(
                success=False,
                final_answer="",
                steps=[],
                error=f"LangGraph 异常: {e!s}",
            )
