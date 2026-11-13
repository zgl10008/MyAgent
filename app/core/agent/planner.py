# -*- coding: utf-8 -*-
"""
规划 Agent：Plan-and-Execute，含任务分解、ReAct 执行与重规划。

Plan-and-Execute = 规划 + 执行，其中执行阶段复用 ReActAgent。
与纯 ReAct 的关系：纯 ReAct 是一层循环（Thought→Action→Obs），
Plan-and-Execute 是两层——上层规划拆解，下层每个子任务走 ReAct 循环。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from app.core.agent.react_agent import AgentResult, LLMCallable, MemoryLike, ToolInvoker

logger = logging.getLogger(__name__)


PLAN_SYSTEM_PROMPT = """你是任务规划专家。请将用户目标拆分为有序、可执行的子任务列表。

## 输出格式（仅输出 JSON，不要其他文字）
{
  "subtasks": [
    {
      "id": "t1",
      "title": "子任务标题",
      "description": "具体要做什么——会交给 ReAct Agent 执行，它可以用工具或直接推理",
      "expected_tools": ["可能需要调用的工具名，没有则为空数组"]
    }
  ]
}

规则：
- 子任务数量建议 2～8 个，避免过细或过粗。
- id 必须唯一。
- 每个子任务的 description 要足够具体，让 ReAct Agent 能独立执行。
"""


REPLAN_SYSTEM_PROMPT = """你是任务规划专家。根据已执行结果与错误信息，修订后续计划。

## 输出格式（仅输出 JSON）
{
  "subtasks": [ ... 同上结构 ... ],
  "notes": "简要说明为何如此调整"
}

若任务已完成，返回：
{ "subtasks": [], "notes": "已完成，原因说明" }
"""


# ---------------------------------------------------------------------------
# ReAct Agent 工厂协议（用于在 Plan 中创建 ReAct Agent 执行子任务）
# ---------------------------------------------------------------------------


class ReActAgentFactory(Protocol):
    """创建 ReAct Agent 的工厂函数。

    Plan-and-Execute 的每个子任务都通过此工厂创建一个 ReAct Agent 实例执行。
    这样保证了"执行阶段 = ReAct 循环"的架构一致性。
    """

    async def __call__(
        self,
        subtask_description: str,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        ...


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """单个子任务。"""

    id: str
    title: str
    description: str
    expected_tools: list[str] = field(default_factory=list)


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法从模型输出中解析 JSON")
    return json.loads(m.group(0))


def _parse_subtasks(data: dict[str, Any]) -> list[SubTask]:
    raw_list = data.get("subtasks") or []
    out: list[SubTask] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        out.append(
            SubTask(
                id=str(item.get("id", f"task_{len(out)}")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                expected_tools=list(item.get("expected_tools") or []),
            )
        )
    return out


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Plan-and-Execute Agent：规划 → ReAct 执行 → 重规划 → 汇总。

    执行阶段通过 react_factory 创建 ReAct Agent 来执行每个子任务，
    保证"执行 = ReAct 循环"的架构一致性。
    """

    def __init__(
        self,
        llm: LLMCallable,
        summary_llm: LLMCallable | None = None,
        max_replan_attempts: int = 2,
        max_steps_per_subtask: int = 5,
    ) -> None:
        """
        :param llm: 用于规划与重规划的 LLM
        :param summary_llm: 用于最终汇总的 LLM（None 则复用 llm）
        :param max_replan_attempts: 最大重规划次数
        :param max_steps_per_subtask: 每个子任务的 ReAct 最大步数
        """
        self._llm = llm
        self._summary_llm = summary_llm or llm
        self.max_replan_attempts = max(0, max_replan_attempts)
        self.max_steps_per_subtask = max(1, max_steps_per_subtask)

    async def plan(self, query: str) -> list[SubTask]:
        """根据用户目标生成子任务列表。"""
        messages: Sequence[dict[str, str]] = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户目标：\n{query}\n\n请输出 JSON 计划。"},
        ]
        try:
            raw = await self._llm.acomplete(messages, temperature=0.3)
            data = _extract_json_object(raw)
            return _parse_subtasks(data)
        except Exception:
            logger.exception("生成计划失败，降级为单步推理")
            return [
                SubTask(
                    id="fallback_1",
                    title="直接回答",
                    description=query,
                    expected_tools=[],
                )
            ]

    async def replan(
        self,
        query: str,
        previous_plan: list[SubTask],
        results: list[dict[str, Any]],
    ) -> list[SubTask]:
        """根据执行结果重新生成子任务列表。"""
        payload = {
            "original_query": query,
            "previous_plan": [
                {"id": t.id, "title": t.title, "description": t.description}
                for t in previous_plan
            ],
            "results_so_far": results,
        }
        messages: Sequence[dict[str, str]] = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "上下文：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)[:16000]
                + "\n\n请输出修订后的 JSON 计划。",
            },
        ]
        try:
            raw = await self._llm.acomplete(messages, temperature=0.3)
            data = _extract_json_object(raw)
            new_plan = _parse_subtasks(data)
            if not new_plan:
                logger.info("重规划返回空计划，任务完成")
            return new_plan
        except Exception:
            logger.exception("重规划失败")
            return []

    async def run_with_replan(
        self,
        query: str,
        react_factory: ReActAgentFactory,
        tool_names: list[str] | None = None,
        trace_callback: Any | None = None,
    ) -> AgentResult:
        """
        Plan-and-Execute 主流程：
        1. LLM 制定计划 → 子任务列表
        2. 对每个子任务，通过 react_factory 创建 ReAct Agent 执行
        3. 失败时重规划，最多重试 max_replan_attempts 次
        4. LLM 汇总所有子任务结果生成最终答案

        :param react_factory: 创建 ReAct Agent 的工厂函数，
            签名为 async (subtask_description, allowed_tools) -> AgentResult
        :param tool_names: 允许使用的工具名列表
        :param trace_callback: 可选追踪回调
        """
        # ── 1. 制定计划 ──
        current_plan = await self.plan(query)
        if trace_callback:
            await trace_callback({
                "phase": "plan",
                "subtask_count": len(current_plan),
                "subtasks": [t.title for t in current_plan],
            })

        all_results: list[dict[str, Any]] = []
        last_error: str | None = None

        for replan_attempt in range(self.max_replan_attempts + 1):
            # ── 2. 逐个执行子任务（每个子任务 = 一次 ReAct 循环）──
            subtask_results: list[dict[str, Any]] = []
            failed = False

            for task in current_plan:
                if trace_callback:
                    await trace_callback({
                        "phase": "execute_subtask",
                        "subtask_id": task.id,
                        "title": task.title,
                    })

                try:
                    result: AgentResult = await react_factory(
                        subtask_description=task.description,
                        allowed_tools=task.expected_tools or tool_names,
                    )
                except Exception as exc:
                    logger.exception("子任务 %s 执行异常", task.id)
                    result = AgentResult(
                        success=False,
                        final_answer="",
                        steps=[],
                        error=str(exc),
                    )

                # 透传该子任务实际调用的工具名，供上层汇总 tool_calls（否则
                # plan_execute 的 tool_calls 恒为空——见 agent_service 的汇总逻辑）。
                subtask_tools: list[str] = []
                for _s in result.steps:
                    _a = _s.get("action")
                    if _a and str(_a) not in subtask_tools:
                        subtask_tools.append(str(_a))

                rec = {
                    "subtask_id": task.id,
                    "title": task.title,
                    "success": result.success,
                    "output": result.final_answer[:4000] if result.final_answer else "",
                    "error": result.error,
                    "steps": len(result.steps),
                    "tools": subtask_tools,
                }
                subtask_results.append(rec)
                if trace_callback:
                    await trace_callback({"phase": "subtask_done", "record": rec})

                if not result.success:
                    failed = True
                    last_error = result.error
                    # 不立即中断——收集所有结果后再决定是否重规划

            all_results.extend(subtask_results)

            # ── 3. 判断是否需要重规划 ──
            if not failed:
                break  # 全部成功，跳出重规划循环

            if replan_attempt >= self.max_replan_attempts:
                break

            if trace_callback:
                await trace_callback({
                    "phase": "replan",
                    "attempt": replan_attempt + 1,
                })

            new_plan = await self.replan(query, current_plan, subtask_results)
            if not new_plan:
                break
            current_plan = new_plan
            if trace_callback:
                await trace_callback({
                    "phase": "replan_done",
                    "new_subtasks": [t.title for t in current_plan],
                })

        # ── 4. LLM 汇总最终答案 ──
        try:
            summary_msgs: Sequence[dict[str, str]] = [
                {
                    "role": "system",
                    "content": "你是总结助手。根据子任务执行记录，给出面向用户的完整最终答案。",
                },
                {
                    "role": "user",
                    "content": (
                        f"原始问题：{query}\n\n"
                        f"子任务执行记录：\n"
                        f"{json.dumps(all_results, ensure_ascii=False, indent=2)[:14000]}"
                    ),
                },
            ]
            final = await self._summary_llm.acomplete(summary_msgs, temperature=0.2)
        except Exception:
            logger.exception("最终汇总失败")
            # 降级：拼接成功子任务的输出
            parts = [r["output"] for r in all_results if r["success"] and r["output"]]
            final = "\n\n".join(parts) if parts else "抱歉，任务执行失败。"

        overall_success = any(r["success"] for r in all_results)
        return AgentResult(
            success=overall_success,
            final_answer=final.strip(),
            steps=all_results,
            error=None if overall_success else (last_error or "所有子任务执行失败"),
        )
