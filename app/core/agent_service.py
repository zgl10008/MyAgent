# -*- coding: utf-8 -*-
"""
Agent 服务：串联编排器、记忆、工具、追踪、意图识别的顶层服务。

为无 Redis/Milvus 环境提供内存版实现，方便本地演示。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from loguru import logger

from app.config import get_settings
from app.core.agent.orchestrator import (
    AgentOrchestrator,
    InMemoryTracer,
    IntentContext,
    OrchestrationMode,
)
from app.core.intent.recognizer import IntentRecognizer
from app.core.persistent_memory import PersistentMemoryManager, _CompressLLMAdapter
from app.core.session_repository import SessionRepository
from app.core.tools.builtin.calculator import CalculatorTool
from app.core.tools.builtin.search import WebSearchTool
from app.core.tools.registry import ToolRegistry
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatMessage,
    IntentInfo,
    StepInfo,
)


# ==============================================================================
# 工具注册表适配器（使 ToolRegistry 满足 orchestrator.ToolRegistry 协议）
# ==============================================================================


class _ToolRegistryAdapter:
    """适配 app.core.tools.registry.ToolRegistry 到 orchestrator 协议。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def list_tool_names(self) -> list[str]:
        return [t.name for t in self._registry.get_all_tools()]

    def get_tools_description(self) -> str:
        return self._registry.get_tools_description()

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            tool = self._registry.get_tool(name)
            result = await tool.execute(**arguments)
            return str(result)
        except KeyError:
            return f"错误：工具 [{name}] 未注册"
        except Exception as e:
            return f"工具 [{name}] 执行失败: {e!s}"


# ==============================================================================
# 模型路由适配器（使 ModelRouter 满足 orchestrator.ModelRouter 协议）
# ==============================================================================


class _ModelRouterAdapter:
    """适配 ModelRouter 使 get_llm(purpose) 返回可 acomplete 的对象。"""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def get_llm(self, purpose: str = "react") -> Any:
        """返回一个带有 acomplete 方法的简易 LLM 包装。"""
        return _LLMWrapper(self._router)


class _LLMWrapper:
    """简易 LLM 包装：使 ModelRouter.chat 适配 acomplete 接口。"""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        resp = await self._router.chat(messages, **kwargs)
        return resp.content


# ==============================================================================
# 配置适配器（使 dict 满足 OrchestratorConfig 协议）
# ==============================================================================


class _DictConfig:
    """基于 dict 的配置，满足 OrchestratorConfig 协议。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._data = dict(config or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


# ==============================================================================
# AgentService — 顶层服务
# ==============================================================================


class AgentService:
    """Agent 服务：组装依赖、编排对话、返回结构化结果。"""

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("正在初始化 AgentService ...")

        # 1. 模型路由
        model_cfg = ModelConfig(
            model_id=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
        self._model_router = ModelRouter([model_cfg])
        self._model_adapter = _ModelRouterAdapter(self._model_router)

        # 2. 工具注册
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(CalculatorTool())
        self._tool_registry.register(WebSearchTool(api_key=settings.serpapi_api_key))
        self._tool_adapter = _ToolRegistryAdapter(self._tool_registry)

        # 3. 记忆（Redis 热路径 + PostgreSQL 冷存储 + 内存降级）
        redis_client = None
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                settings.redis_url, decode_responses=False
            )
            logger.info("Redis 客户端已创建: {}", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis 不可用，将使用内存降级: {}", exc)

        pg_repo = SessionRepository()
        self._memory = PersistentMemoryManager(
            redis_client=redis_client,
            compress_llm=_CompressLLMAdapter(self._model_adapter),
            pg_repo=pg_repo,
        )

        # 4. 追踪与意图（LLM 分类 + 关键词降级）
        self._tracer = InMemoryTracer()
        intent_llm = self._model_adapter.get_llm("intent")
        self._intent_recognizer = IntentRecognizer(llm=intent_llm)

        # 5. 编排器配置
        orchestrator_config = _DictConfig(
            {
                "enable_reflection": True,
                "fallback_react_on_plan_failure": True,
                "react_max_steps": 10,
                "max_replan_attempts": 2,
                "reflection_min_quality": 60,
            }
        )

        # 6. 编排器
        self._orchestrator = AgentOrchestrator(
            config=orchestrator_config,
            model_router=self._model_adapter,
            memory_manager=self._memory,
            tool_registry=self._tool_adapter,
            tracer=self._tracer,
        )

        logger.info("AgentService 初始化完成")

    @property
    def memory(self) -> PersistentMemoryManager:
        """暴露记忆管理器供外部访问（如会话列表路由）。"""
        return self._memory

    async def chat(
        self,
        request: AgentChatRequest,
    ) -> AgentChatResponse:
        """执行 Agent 编排对话，返回结构化响应。"""
        trace_id = str(uuid.uuid4())
        session_id = request.conversation_id or trace_id
        user_text = request.messages[-1].content if request.messages else ""

        # 1. 意图识别
        intent_result = await self._intent_recognizer.recognize(user_text)

        intent_ctx = IntentContext(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            slots=intent_result.slots,
            preferred_mode=request.agent_mode,  # type: ignore[arg-type]
        )

        # 2. 执行编排
        agent_resp = await self._orchestrator.run(
            user_input=user_text,
            session_id=session_id,
            mode=request.agent_mode,
            intent=intent_ctx,
        )

        # 3. 保存记忆
        await self._memory.append_turn(session_id, "user", user_text)
        if agent_resp.answer:
            await self._memory.append_turn(
                session_id,
                "assistant",
                agent_resp.answer,
                metadata={"trace_id": trace_id, "agent_mode": agent_resp.mode_used},
            )

        # 4. 转换步骤为 StepInfo
        steps: list[StepInfo] = []
        tool_names: set[str] = set()
        for s in agent_resp.steps:
            parsed = s.get("parsed") or {}
            action = s.get("action") or parsed.get("action")
            if action:
                tool_names.add(str(action))
            steps.append(
                StepInfo(
                    step=int(s.get("step", 0)),
                    phase=str(s.get("phase", "react")),
                    thought=str(parsed.get("thought") or ""),
                    action=str(action) if action else None,
                    action_input=s.get("action_input") or parsed.get("action_input"),
                    observation=str(s.get("observation") or ""),
                    final_answer=str(parsed.get("final_answer") or s.get("final_answer") or ""),
                    error=str(s.get("error") or ""),
                )
            )

        # 5. 构建响应
        usage = {"total_steps": len(agent_resp.steps)}

        return AgentChatResponse(
            id=trace_id,
            model=request.model or get_settings().openai_model,
            content=agent_resp.answer,
            trace_id=agent_resp.trace_id,
            usage=usage,
            agent_mode=agent_resp.mode_used,
            intent=IntentInfo(
                intent=intent_result.intent,
                confidence=intent_result.confidence,
                sub_intent=intent_result.sub_intent,
                rationale=intent_result.rationale,
            ),
            steps=steps,
            tool_calls=sorted(tool_names),
            reflection_score=(
                agent_resp.reflection.quality_score
                if agent_resp.reflection
                else None
            ),
            degraded=agent_resp.degraded,
        )
