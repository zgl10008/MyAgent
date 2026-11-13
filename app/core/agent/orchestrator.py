# -*- coding: utf-8 -*-
"""
Agent 编排器：根据意图与模式选择 ReAct 或 Plan-and-Execute，串联记忆、工具与追踪。

两种 Agent 架构范式：
  react        — ReAct 循环（Thought→Action→Observation→...→Final Answer）
  plan_execute — Plan-and-Execute（规划 → ReAct 执行每个子任务 → 汇总）

Function Calling 不是第三种模式，而是 ReAct 的主工具调用后端。
当原生工具调用不可用或执行失败时，内部可降级到兼容后端，保持上层模式不变。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence

from app.core.agent.function_call_agent import FunctionCallAgent
from app.core.agent.planner import PlannerAgent
from app.core.agent.react_agent import AgentResult, LLMCallable, ReActAgent
from app.core.agent.reflection import (
    ReflectionAgent,
    ReflectionLoop,
    ReflectionLoopResult,
    ReflectionReport,
    SelfCorrector,
)
from app.core.agent.reflection_gating import ReflectionGating, RiskLevel

logger = logging.getLogger(__name__)

OrchestrationMode = Literal["react", "plan_execute"]


# ---------------------------------------------------------------------------
# 依赖抽象（由基础设施层实现，此处仅约定接口）
# ---------------------------------------------------------------------------
class OrchestratorConfig(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


class ModelRouter(Protocol):
    def get_llm(self, purpose: str) -> Any: ...


class MemoryManager(Protocol):
    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> List[str]: ...
    async def append_turn(self, session_id: str, role: str, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None: ...


class ToolRegistry(Protocol):
    def list_tool_names(self) -> List[str]: ...
    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str: ...


class Tracer(Protocol):
    def new_trace_id(self) -> str: ...
    def start_span(self, name: str, trace_id: str, attributes: Optional[Dict[str, Any]] = None) -> Any: ...
    def end_span(self, span: Any, error: Optional[BaseException] = None) -> None: ...
    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# 响应与意图上下文
# ---------------------------------------------------------------------------
@dataclass
class IntentContext:
    intent: str = "general"
    confidence: float = 1.0
    slots: Dict[str, Any] = field(default_factory=dict)
    preferred_mode: Optional[OrchestrationMode] = None
    allowed_tools: Optional[List[str]] = None
    domain: str = "general"


@dataclass
class AgentResponse:
    answer: str
    mode_used: OrchestrationMode
    success: bool
    trace_id: str
    intent: IntentContext
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reflection: Optional[ReflectionReport] = None
    error: Optional[str] = None
    degraded: bool = False
    correction_applied: bool = False
    reflection_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM 适配器
# ---------------------------------------------------------------------------
class _LLMAdapter:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        fn = getattr(self._llm, "acomplete", None)
        if callable(fn):
            return await fn(messages, **kwargs)
        fn2 = getattr(self._llm, "complete", None)
        if callable(fn2):
            import asyncio
            return await asyncio.to_thread(fn2, messages, **kwargs)
        raise TypeError("LLM 需实现 acomplete 或 complete")


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------
class AgentOrchestrator:
    """Agent 编排器：react / plan_execute 两种范式，FC 作为 ReAct 后端可配置。"""

    def __init__(
        self,
        config: OrchestratorConfig,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        tool_registry: ToolRegistry,
        tracer: Tracer,
    ) -> None:
        self._config = config
        self._model_router = model_router
        self._memory = memory_manager
        self._tools = tool_registry
        self._tracer = tracer
        self._enable_reflection = bool(config.get("enable_reflection", True))
        self._fallback_on_plan_failure = bool(config.get("fallback_react_on_plan_failure", True))
        self._use_function_calling = bool(config.get("use_function_calling", False))
        self._fc_agent: Optional[FunctionCallAgent] = None

        # 反思闭环（评估 → 修正 → 可能重试）
        self._reflection_max_retries = int(config.get("reflection_max_retries", 2))
        self._reflection_loop: Optional[ReflectionLoop] = None

        # 反思风险分级（默认关闭，保持原行为）
        self._enable_reflection_gating = bool(config.get("enable_reflection_gating", False))
        self._gating = ReflectionGating()

        self._init_reflection_if_needed()

    def _init_reflection_if_needed(self) -> None:
        """延迟初始化反思闭环组件。"""
        if not self._enable_reflection:
            return
        try:
            llm = _LLMAdapter(self._model_router.get_llm("reflection"))
            reflection_agent = ReflectionAgent(
                llm=llm,
                min_quality_to_pass=int(self._config.get("reflection_min_quality", 60)),
            )
            corrector = SelfCorrector(llm=llm)
            self._reflection_loop = ReflectionLoop(
                reflection_agent=reflection_agent,
                self_corrector=corrector,
            )
        except Exception as e:
            logger.warning("反思闭环初始化失败，反思降级为仅展示评分: %s", e)
            self._reflection_loop = None

    # ------------------------------------------------------------------
    # 配置接口
    # ------------------------------------------------------------------
    def set_function_call_agent(self, agent: FunctionCallAgent) -> None:
        """注入 FunctionCall Agent 作为 ReAct 的 FC 后端。
        当 use_function_calling=True 时，ReAct 和 Plan-and-Execute 的子任务
        执行都会使用此 agent。"""
        self._fc_agent = agent
        self._use_function_calling = True

    def _tool_names(self, intent: IntentContext) -> List[str]:
        all_names = self._tools.list_tool_names()
        if intent.allowed_tools:
            return [n for n in intent.allowed_tools if n in all_names]
        return all_names

    def _safe_react_tools(self, intent: IntentContext) -> List[str]:
        """Return the intent-scoped tool whitelist used by fallback paths."""
        return self._tool_names(intent)

    # ------------------------------------------------------------------
    # 创建 ReAct 执行器（供 react 模式和 plan_execute 子任务复用）
    # ------------------------------------------------------------------
    async def _invoke_react(
        self,
        query: str,
        session_id: str,
        intent: IntentContext,
        trace_id: str,
        steps: List[Dict[str, Any]],
        max_steps: int = 10,
        suffix: str = "",
        allowed_tools: Optional[List[str]] = None,
    ) -> AgentResult:
        """执行一次 ReAct 循环，优先使用 FC，失败时内部降级。"""
        tool_names = allowed_tools or self._tool_names(intent)

        async def trace_cb(rec: Dict[str, Any]) -> None:
            payload = {"ts": time.time(), **rec}
            steps.append(payload)
            self._tracer.log_event(trace_id, "react.step", payload)

        ctx: Dict[str, Any] = {
            "session_id": session_id,
            "tool_names": tool_names,
            "tool_descriptions": getattr(self._tools, "get_tools_description", lambda names=None: "")(tool_names),
            "trace_callback": trace_cb,
            "extra_system": suffix + f"当前识别意图：{intent.intent}，槽位：{intent.slots}",
        }

        if self._use_function_calling and self._fc_agent is not None:
            # Function Calling 是主路径；文本后端只作为内部兼容降级。
            self._fc_agent.max_steps = max_steps
            try:
                result = await self._fc_agent.run(query, ctx)
            except Exception as exc:
                fallback_event = {
                    "event": "react.backend_fallback",
                    "from": "function_calling",
                    "to": "compatibility",
                    "error": str(exc),
                }
                used_fallback = True
                steps.append(fallback_event)
                self._tracer.log_event(trace_id, "react.backend_fallback", fallback_event)
                logger.warning("Function Calling 执行异常，降级到兼容后端: %s", exc)
            else:
                if result.success:
                    return result
                fallback_event = {
                    "event": "react.backend_fallback",
                    "from": "function_calling",
                    "to": "compatibility",
                    "error": result.error or "Function Calling 未返回最终答案",
                }
                used_fallback = True
                steps.append(fallback_event)
                self._tracer.log_event(trace_id, "react.backend_fallback", fallback_event)
                logger.warning(
                    "Function Calling 执行失败，降级到兼容后端: %s",
                    result.error or "unknown error",
                )

        # 内部兼容后端：不作为独立上层模式暴露。
        llm = _LLMAdapter(self._model_router.get_llm("react"))
        agent = ReActAgent(
            llm=llm,
            tools=self._tools,
            memory=self._memory,
            max_steps=max_steps,
            session_id=session_id,
        )
        return await agent.run(query, ctx)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def run(
        self,
        user_input: str,
        session_id: str,
        mode: str = "react",
        intent: Optional[IntentContext] = None,
    ) -> AgentResponse:
        """执行 Agent 编排。两种范式：react / plan_execute。"""
        intent_ctx = intent or IntentContext()
        if intent_ctx.preferred_mode:
            mode = intent_ctx.preferred_mode

        trace_id = self._tracer.new_trace_id()
        span = self._tracer.start_span(
            "orchestrator.run", trace_id,
            attributes={"session_id": session_id, "mode": mode, "intent": intent_ctx.intent},
        )

        self._tracer.log_event(trace_id, "orchestrator.start",
                               {"user_input_len": len(user_input), "mode": mode})

        steps: List[Dict[str, Any]] = []
        degraded = False
        error_msg: Optional[str] = None
        result: Optional[AgentResult] = None
        mode_used: OrchestrationMode = "react"

        try:
            if mode == "plan_execute":
                mode_used = "plan_execute"
                result = await self._run_plan_execute(
                    user_input, session_id, intent_ctx, trace_id, steps,
                )
                if not result.success and self._fallback_on_plan_failure:
                    self._tracer.log_event(trace_id, "orchestrator.fallback", {"to": "react"})
                    degraded = True
                    result = await self._invoke_react(
                        user_input, session_id, intent_ctx, trace_id, steps,
                        suffix="[Plan 降级] ",
                        allowed_tools=self._safe_react_tools(intent_ctx),
                    )
                    mode_used = "react"
            else:
                result = await self._invoke_react(
                    user_input, session_id, intent_ctx, trace_id, steps,
                    max_steps=int(self._config.get("react_max_steps", 10)),
                )

            answer = (result.final_answer if result else "") or ""
            degraded = degraded or any(
                step.get("event") == "react.backend_fallback" for step in steps
            )
            success = bool(result and result.success)

            # ── 反思闭环：先分级，再决定执行策略 ──
            reflection_report: Optional[ReflectionReport] = None
            correction_applied = False
            reflection_warnings: list[str] = []
            retries_made = 0

            risk_level = RiskLevel.R3_HIGH  # 分级未启用时默认保持原行为
            if self._enable_reflection_gating:
                risk_level = self._gating.decide(
                    intent=intent_ctx.intent,
                    confidence=intent_ctx.confidence,
                    steps=steps,
                    degraded=degraded,
                    domain=getattr(intent_ctx, "domain", "general"),
                )
                self._tracer.log_event(
                    trace_id, "reflection.gating", {"risk": risk_level.value},
                )

            allow_retry = risk_level == RiskLevel.R3_HIGH

            if risk_level == RiskLevel.R1_LOW:
                # 低风险：跳过反思，直接返回
                self._tracer.log_event(trace_id, "reflection.skip", {"risk": risk_level.value})

            elif self._enable_reflection and answer and self._reflection_loop is not None:
                try:
                    self._tracer.log_event(trace_id, "reflection.loop.begin", {})
                    loop_result = await self._reflection_loop.evaluate_and_improve(
                        user_query=user_input,
                        answer=answer,
                        steps=steps,
                        allow_retry=allow_retry,
                    )
                    answer = loop_result.final_answer
                    reflection_report = loop_result.reflection_report
                    correction_applied = loop_result.corrections_made > 0
                    reflection_warnings = list(loop_result.warnings)
                    self._tracer.log_event(
                        trace_id, "reflection.loop.end",
                        {
                            "score": loop_result.final_score,
                            "action": loop_result.action_taken,
                            "corrections": loop_result.corrections_made,
                        },
                    )

                    # 需要完整重试：重新执行 Agent（携带反思反馈）
                    while (
                        allow_retry
                        and loop_result.action_taken == "retry_needed"
                        and retries_made < self._reflection_max_retries
                        and loop_result.retry_context is not None
                    ):
                        retries_made += 1
                        retry_hint = loop_result.retry_context.to_hint()
                        logger.warning(
                            "反思触发 Agent 重试 #%d session=%s score=%d",
                            retries_made, session_id, loop_result.final_score,
                        )
                        self._tracer.log_event(
                            trace_id, "reflection.retry",
                            {"attempt": retries_made, "hint": retry_hint[:500]},
                        )

                        retry_result = await self._invoke_react(
                            user_input, session_id, intent_ctx, trace_id, steps,
                            suffix=f"\n[反思重试 #{retries_made}]\n{retry_hint}",
                        )
                        if retry_result and retry_result.final_answer:
                            answer = retry_result.final_answer

                        loop_result = await self._reflection_loop.evaluate_and_improve(
                            user_query=user_input,
                            answer=answer,
                            steps=steps,
                        )
                        if loop_result.final_score > (
                            reflection_report.quality_score if reflection_report else 0
                        ):
                            answer = loop_result.final_answer
                            reflection_report = loop_result.reflection_report
                            correction_applied = loop_result.corrections_made > 0
                            reflection_warnings = list(loop_result.warnings)

                except Exception as e:
                    logger.warning("反思闭环失败，已跳过: %s", e)
                    self._tracer.log_event(trace_id, "reflection.loop.error", {"error": str(e)})

            resp = AgentResponse(
                answer=answer, mode_used=mode_used, success=success,
                trace_id=trace_id, intent=intent_ctx, steps=steps,
                reflection=reflection_report,
                error=result.error if result and not success else None,
                degraded=degraded,
                correction_applied=correction_applied,
                reflection_warnings=reflection_warnings,
            )
            self._tracer.end_span(span, error=None)
            return resp

        except Exception as e:
            logger.exception("编排器未捕获异常")
            error_msg = str(e)
            self._tracer.log_event(trace_id, "orchestrator.error", {"error": error_msg})
            self._tracer.end_span(span, error=e)
            return AgentResponse(
                answer="", mode_used="react", success=False,
                trace_id=trace_id, intent=intent_ctx, steps=steps,
                error=error_msg, degraded=degraded,
            )

    # ------------------------------------------------------------------
    # Plan-and-Execute
    # ------------------------------------------------------------------
    async def _run_plan_execute(
        self,
        user_input: str,
        session_id: str,
        intent: IntentContext,
        trace_id: str,
        steps: List[Dict[str, Any]],
    ) -> AgentResult:
        """Plan-and-Execute：规划 → ReAct 执行每个子任务 → 重规划 → 汇总。

        执行阶段复用 _invoke_react()，保证"执行 = ReAct 循环"的架构一致性。
        """
        plan_llm = _LLMAdapter(self._model_router.get_llm("planner"))
        summary_llm = _LLMAdapter(self._model_router.get_llm("react"))

        async def trace_cb(payload: Dict[str, Any]) -> None:
            entry = {"ts": time.time(), **payload}
            steps.append(entry)
            self._tracer.log_event(trace_id, "plan_execute", entry)

        # Plan-Execute 与 ReAct 使用同一份意图级工具白名单；具体工具失败时
        # 返回可观察的错误信息，由执行器决定重试、重规划或改用其它工具。
        pe_tools = self._safe_react_tools(intent)

        def _sanitize_tools(allowed_tools: list[str] | None) -> list[str]:
            raw = list(allowed_tools) if allowed_tools else list(pe_tools)
            # 去重保序，避免重复注册同一工具
            uniq: list[str] = []
            seen: set[str] = set()
            for t in raw:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            return uniq or pe_tools

        # ReAct 工厂：每个子任务创建一个 ReAct 执行器
        async def react_factory(
            subtask_description: str,
            allowed_tools: list[str] | None = None,
        ) -> AgentResult:
            subtask_steps: List[Dict[str, Any]] = []
            result = await self._invoke_react(
                query=subtask_description,
                session_id=session_id,
                intent=intent,
                trace_id=trace_id,
                steps=subtask_steps,
                max_steps=int(self._config.get("react_max_steps", 5)),
                suffix=f"[子任务] ",
                allowed_tools=_sanitize_tools(allowed_tools),
            )
            return result

        planner = PlannerAgent(
            llm=plan_llm,
            summary_llm=summary_llm,
            max_replan_attempts=int(self._config.get("max_replan_attempts", 2)),
            max_steps_per_subtask=int(self._config.get("react_max_steps", 5)),
        )

        self._tracer.log_event(trace_id, "planner.begin", {})
        result = await planner.run_with_replan(
            query=user_input,
            react_factory=react_factory,
            tool_names=pe_tools,
            trace_callback=trace_cb,
        )
        self._tracer.log_event(
            trace_id, "planner.end",
            {"success": result.success, "subtasks": len(result.steps)},
        )
        return result

    # ------------------------------------------------------------------
    # 反思
    # ------------------------------------------------------------------
    async def _run_reflection(
        self, user_query: str, answer: str, trace_id: str,
        steps: List[Dict[str, Any]],
    ) -> ReflectionReport:
        llm = _LLMAdapter(self._model_router.get_llm("reflection"))
        agent = ReflectionAgent(
            llm=llm,
            min_quality_to_pass=int(self._config.get("reflection_min_quality", 60)),
        )
        trace_summary = str([s.get("phase") or s.get("subtask_id") for s in steps[-20:]])
        self._tracer.log_event(trace_id, "reflection.begin", {})
        report = await agent.reflect(user_query, answer, trace_summary=trace_summary)
        self._tracer.log_event(
            trace_id, "reflection.end",
            {"quality": report.quality_score, "hallucination": report.likely_hallucination},
        )
        return report


# ---------------------------------------------------------------------------
# 简易内存追踪器
# ---------------------------------------------------------------------------
class InMemoryTracer:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    def start_span(self, name: str, trace_id: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"name": name, "trace_id": trace_id, "attributes": attributes or {}}

    def end_span(self, span: Any, error: Optional[BaseException] = None) -> None:
        self.events.append({"type": "end_span", "span": span, "error": str(error) if error else None})

    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None:
        self.events.append({"trace_id": trace_id, "name": name, "payload": payload})
