# -*- coding: utf-8 -*-
"""
Agent 服务：串联编排器、记忆、工具、追踪、意图识别的顶层服务。

为无 Redis/Milvus 环境提供内存版实现，方便本地演示。
"""

from __future__ import annotations

import contextvars
import time
import uuid
from typing import Any

from loguru import logger

from app.config import get_settings
from app.core.agent.function_call_agent import FunctionCallAgent
from app.core.agent.mode_router import ModeRouter
from app.core.domain.policy import get_domain_policy
from app.core.domain.risk_classifier import (
    RiskDecision,
    RiskDisposition,
    classify_risk,
)
from app.core.agent.orchestrator import (
    AgentOrchestrator,
    InMemoryTracer,
    IntentContext,
    OrchestrationMode,
)
from app.core.intent.recognizer import IntentRecognizer
from app.core.memory.long_term import LongTermMemory
from app.core.persistent_memory import PersistentMemoryManager, _CompressLLMAdapter
from app.core.session_repository import SessionRepository
from app.core.tools.builtin.calculator import CalculatorTool
from app.core.tools.builtin.rag_search import RAGSearchTool
from app.core.tools.builtin.search import WebSearchTool
from app.core.tools.registry import ToolRegistry
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatMessage,
    IntentInfo,
    RetrievalResult,
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

    def get_tools_description(self, names: list[str] | None = None) -> str:
        return self._registry.get_tools_description(names)

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


class _UsageAccumulator:
    """单次请求的 token 用量累加器。

    由 contextvar 承载，保证并发请求各自独立统计，互不串扰。
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)

    def snapshot(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# 当前请求的 token 用量累加器（contextvar：每个请求任务独立上下文）
_usage_ctx: contextvars.ContextVar[_UsageAccumulator | None] = contextvars.ContextVar(
    "agent_token_usage", default=None
)

# 当前请求 Agent 通过 rag_search 检索到的文档上下文（contextvar，供前端右侧面板展示）
_rag_contexts_ctx: contextvars.ContextVar[list[RetrievalResult] | None] = contextvars.ContextVar(
    "agent_rag_contexts", default=None
)


class _ModelRouterAdapter:
    """适配 ModelRouter 使 get_llm(purpose) 返回可 acomplete 的对象。"""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def get_llm(self, purpose: str = "react") -> Any:
        """返回一个带有 acomplete 方法的简易 LLM 包装。"""
        return _LLMWrapper(self._router)

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        """供 FunctionCall 后端把本次 LLM 调用 token 计入当前请求。"""
        acc = _usage_ctx.get()
        if acc is not None:
            acc.add(usage)


class _LLMWrapper:
    """简易 LLM 包装：使 ModelRouter.chat 适配 acomplete 接口。"""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def acomplete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        resp = await self._router.chat(messages, **kwargs)
        acc = _usage_ctx.get()
        if acc is not None:
            acc.add(resp.usage)
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
        self._tool_registry.register(WebSearchTool(
            timeout=settings.search_timeout,
            mock=settings.mock_web_search,
        ))
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

        # 3b. 长期语义记忆（Milvus，可选）
        try:
            from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
            try:
                from pymilvus import IndexParams  # pymilvus 2.x
            except ImportError:
                from pymilvus.milvus_client.index import IndexParams  # pymilvus 3.x

            _ltm_client = MilvusClient(settings.milvus_db_path)
            _ltm_coll_name = "long_term_memory"

            # 长期记忆向量维度必须与当前 embedding 模型一致（bge-m3=1024，all-MiniLM-L6-v2=384），
            # 从模型动态读取，避免换模型后 schema 维度错位导致读写失败。
            from app.core.rag_service import (
                SentenceTransformerEmbeddingAdapter,
                _get_embedding_model,
            )

            try:
                _ltm_dim = _get_embedding_model().get_sentence_embedding_dimension()
            except Exception:
                _ltm_dim = 384

            def _ensure_ltm_collection() -> None:
                """按 LongTermMemory 的字段约定建集合，并迁移旧版错误 schema。

                旧版用 create_collection(dimension=...) 默认建出 id/vector 两个字段，
                与 LongTermMemory 读写使用的 pk/embedding/content/session_id/meta
                完全错位，且度量 L2 与集合 COSINE 不一致，导致读写双双失败。
                此处仅在旧集合确实为空时重建，避免误删数据。
                """
                required_fields = {"pk", "embedding", "content", "session_id", "meta"}
                if _ltm_client.has_collection(_ltm_coll_name):
                    fields_desc = _ltm_client.describe_collection(_ltm_coll_name)["fields"]
                    field_names = {f["name"] for f in fields_desc}
                    # 校验 embedding 维度是否与当前模型一致，换模型（384→1024）后需重建
                    existing_dim = None
                    for f in fields_desc:
                        if f["name"] == "embedding":
                            existing_dim = (f.get("params") or {}).get("dim")
                            break
                    if required_fields.issubset(field_names) and existing_dim == _ltm_dim:
                        _ltm_client.load_collection(_ltm_coll_name)
                        return
                    if existing_dim is not None and existing_dim != _ltm_dim:
                        logger.warning(
                            "长期语义记忆集合 embedding 维度 {} 与模型 {} 不一致，尝试重建: {}",
                            existing_dim,
                            _ltm_dim,
                            _ltm_coll_name,
                        )
                    logger.warning(
                        "长期语义记忆集合 schema 不兼容（旧版 id/vector），尝试重建: {}",
                        _ltm_coll_name,
                    )
                    try:
                        _ltm_client.load_collection(_ltm_coll_name)
                        cnt = _ltm_client.query(
                            _ltm_coll_name,
                            filter="",
                            output_fields=["count(*)"],
                        )[0]["count(*)"]
                    except Exception:
                        cnt = 0
                    if cnt:
                        logger.error(
                            "长期语义记忆集合含数据且 schema 不兼容，跳过重建，请手动迁移: {}",
                            _ltm_coll_name,
                        )
                        return
                    _ltm_client.drop_collection(_ltm_coll_name)

                fields = [
                    FieldSchema(
                        name="pk",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=64,
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=_ltm_dim,
                    ),
                    FieldSchema(
                        name="content",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                    ),
                    FieldSchema(
                        name="session_id",
                        dtype=DataType.VARCHAR,
                        max_length=128,
                    ),
                    FieldSchema(
                        name="meta",
                        dtype=DataType.VARCHAR,
                        max_length=8192,
                    ),
                ]
                schema = CollectionSchema(fields, description="长期语义记忆（按会话语义召回）")
                try:
                    index_params = IndexParams()
                    index_params.add_index(
                        field_name="embedding",
                        index_type="AUTOINDEX",
                        metric_type="COSINE",
                    )
                    _ltm_client.create_collection(
                        collection_name=_ltm_coll_name,
                        schema=schema,
                        index_params=index_params,
                    )
                except Exception:
                    # 兼容旧版 pymilvus（index_params 需为 dict）；
                    # 若 IndexParams 方式已半建集合，先清理再重建。
                    logger.warning("IndexParams 建集合失败，退回 dict 方式")
                    if _ltm_client.has_collection(_ltm_coll_name):
                        _ltm_client.drop_collection(_ltm_coll_name)
                    _ltm_client.create_collection(
                        collection_name=_ltm_coll_name,
                        schema=schema,
                        index_params={
                            "index_type": "AUTOINDEX",
                            "metric_type": "COSINE",
                            "params": {},
                        },
                    )
                _ltm_client.load_collection(_ltm_coll_name)
                logger.info("长期语义记忆集合已就绪: {}", _ltm_coll_name)

            _ensure_ltm_collection()

            # 适配器：MilvusClient → ORM Collection 接口（LongTermMemory 使用 ORM 风格 API）
            class _LTMCollectionAdapter:
                def __init__(self, client: Any, coll_name: str) -> None:
                    self._c = client
                    self._n = coll_name

                def insert(self, data: Any, **kwargs: Any) -> Any:
                    # LongTermMemory.store 传入的是行字典列表，直接透传即可。
                    # 此前按 5 元组解包，与实际调用不匹配，导致写入在适配器层就失败。
                    return self._c.insert(collection_name=self._n, data=data)

                def search(self, data: Any, anns_field: str, param: Any,
                           limit: int, expr: str | None = None,
                           output_fields: list[str] | None = None, **kwargs: Any) -> Any:
                    return self._c.search(
                        collection_name=self._n, data=data, anns_field=anns_field,
                        search_params=param, limit=limit, filter=expr,
                        output_fields=output_fields,
                    )

                def delete(self, expr: str, **kwargs: Any) -> Any:
                    return self._c.delete(collection_name=self._n, filter=expr)

                def flush(self, **kwargs: Any) -> Any:
                    return self._c.flush(collection_name=self._n)

            _ltm_embed = SentenceTransformerEmbeddingAdapter(_get_embedding_model())
            _ltm_coll = _LTMCollectionAdapter(_ltm_client, _ltm_coll_name)
            ltm = LongTermMemory(milvus_collection=_ltm_coll, embedding_model=_ltm_embed)
            self._memory.set_long_term(ltm)
            logger.info("长期语义记忆已启用（Milvus 语义召回，collection={}）", _ltm_coll_name)
        except Exception as exc:
            logger.warning("长期语义记忆不可用，跳过: {}", exc)

        # 4. 追踪与意图（LLM 分类 + 关键词降级）
        self._tracer = InMemoryTracer()
        intent_llm = self._model_adapter.get_llm("intent")
        self._intent_recognizer = IntentRecognizer(llm=intent_llm)

        # 4b. ModeRouter（意图驱动的自动模式选择）
        # RAG 可用性通过工具列表传递（含 rag_search 即表示知识库可用）
        self._rag_service: Any = None
        self._mode_router = ModeRouter(
            llm=intent_llm,
            tool_names=self._tool_adapter.list_tool_names(),
        )
        logger.info("ModeRouter 已就绪（规则快路径 + LLM Router 兜底）")

        # 5. 编排器配置（三开关来自 settings，默认全开；评测 A/B 消融通过 env 关闭）
        orchestrator_config = _DictConfig(
            {
                "enable_reflection": settings.agent_enable_reflection,
                "fallback_react_on_plan_failure": settings.agent_fallback_react_on_plan_failure,
                "react_max_steps": 10,
                "max_replan_attempts": settings.agent_max_replan_attempts,
                "reflection_min_quality": 60,
                "use_function_calling": settings.agent_use_function_calling,
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

        # 7. FunctionCall Agent（ReAct 的 FC 后端，非独立模式）
        # 当 use_function_calling=True 时，ReAct 和 Plan-and-Execute 子任务都使用此 backend
        try:
            _fc_tools = self._build_function_call_tools()

            self._function_call_agent = FunctionCallAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base or "https://api.openai.com/v1",
                tools=_fc_tools,
                max_steps=int(orchestrator_config.get("react_max_steps", 10)),
                memory=self._memory,
                usage_callback=self._model_adapter.add_usage,
            )
            self._orchestrator.set_function_call_agent(self._function_call_agent)
            logger.info("FunctionCall Agent 已注入编排器（function_call 模式可用）")
        except Exception as exc:
            logger.warning("FunctionCall Agent 创建失败，function_call 模式不可用: {}", exc)

        logger.info("AgentService 初始化完成")

    def _build_function_call_tools(self) -> list[Any]:
        """Build LangChain tools from the current registry snapshot."""
        from langchain_core.tools import StructuredTool
        from pydantic import Field, create_model

        parameter_types: dict[str, type[Any]] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        fc_tools: list[Any] = []
        for tool in self._tool_registry.get_all_tools():
            fields: dict[str, Any] = {}
            for param in tool.parameters:
                python_type = parameter_types.get(param.type, str)
                field_kwargs = {"description": param.description}
                if param.required:
                    fields[param.name] = (python_type, Field(**field_kwargs))
                else:
                    fields[param.name] = (
                        python_type | None,
                        Field(default=None, **field_kwargs),
                    )
            if not fields:
                fields["query"] = (str, Field(description="查询参数"))

            tool_args = create_model(f"{tool.name}_args", **fields)

            def _make_executor(current_tool=tool):
                async def _execute(**kwargs: Any) -> str:
                    filtered = {k: v for k, v in kwargs.items() if v is not None}
                    return await current_tool.execute(**filtered)

                _execute.__name__ = current_tool.name
                return _execute

            fc_tools.append(
                StructuredTool.from_function(
                    coroutine=_make_executor(),
                    name=tool.name,
                    description=tool.description,
                    args_schema=tool_args,
                )
            )
        return fc_tools

    @property
    def memory(self) -> PersistentMemoryManager:
        """暴露记忆管理器供外部访问（如会话列表路由）。"""
        return self._memory

    def set_rag_service(self, rag_service: Any) -> None:
        """注入 RAGService 并将其注册为 Agent 可调用的知识库检索工具。

        RAG 不再是独立端点——Agent 在 ReAct 循环中可以自主决定：
        - 何时用 rag_search 查私有知识库
        - 何时用 web_search 搜公开互联网
        - 何时两者结果交叉验证
        """
        self._rag_service = rag_service

        # 将 RAG 检索封装为工具，注册到工具系统中
        async def _search(query: str, top_k: int = 5) -> list:
            results = await rag_service.search(query, top_k=top_k, mode="hybrid")
            # 记录结构化检索结果，供前端右侧面板展示（Agent 模式也可见）
            acc = _rag_contexts_ctx.get()
            if acc is not None:
                existing_ids = {getattr(r, "id", None) for r in acc}
                for r in results:
                    rid = getattr(r, "id", None)
                    if rid not in existing_ids:
                        acc.append(r)
                        existing_ids.add(rid)
            return results

        rag_tool = RAGSearchTool(
            search_func=_search,
            chunks_count=lambda: rag_service.chunks_count,
        )
        self._tool_registry.register(rag_tool)

        # FunctionCallAgent 在服务初始化时会捕获一次工具快照；RAG 属于后注入
        # 工具，因此注册后必须刷新绑定，否则 FC 主链路无法看到 rag_search。
        if hasattr(self, "_function_call_agent"):
            self._function_call_agent.set_tools(self._build_function_call_tools())

        # 更新 ModeRouter 的工具列表（用于 LLM Router 决策时感知可用能力）
        self._mode_router._tool_names = self._tool_adapter.list_tool_names()

        logger.info(
            "RAGService 已注入并同步到 Function Calling 工具集（知识库块数: {}）",
            rag_service.chunks_count,
        )

    async def _classify_safety(
        self,
        policy,
        text: str,
    ) -> RiskDecision:
        """两阶段医疗风险分级；关闭 LLM 确认时退化为纯关键词判断。"""
        settings = get_settings()
        if not settings.safety_risk_classifier_enabled:
            if policy.is_high_risk(text):
                return RiskDecision(
                    RiskDisposition.EMERGENCY,
                    reason="命中高风险关键词（LLM 确认已关闭）",
                    answer=policy.safety_prefix(text),
                )
            return RiskDecision(RiskDisposition.SAFE, reason="未命中高风险关键词")
        safety_llm = self._model_adapter.get_llm("safety")
        return await classify_risk(policy, text, safety_llm)

    async def classify_safety(self, domain: str, text: str) -> RiskDecision:
        """供所有聊天入口复用的统一医疗安全分流。"""
        return await self._classify_safety(get_domain_policy(domain), text)

    async def persist_exchange(
        self,
        session_id: str,
        user_text: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """统一写入一轮会话，供直连与 RAG 链路复用。"""
        await self._memory.append_turn(session_id, "user", user_text)
        if answer:
            await self._memory.append_turn(session_id, "assistant", answer, metadata=metadata)

    def _snapshot_usage(self, total_steps: int) -> dict[str, int]:
        """读取当前请求已累计的 token 用量，并附上执行步骤数。"""
        acc = _usage_ctx.get()
        usage = acc.snapshot() if acc else {}
        usage["total_steps"] = total_steps
        return usage

    async def _safety_response(
        self,
        trace_id: str,
        request: AgentChatRequest,
        session_id: str,
        user_text: str,
        answer: str,
        mode: str,
    ) -> AgentChatResponse:
        """安全分流（急诊/澄清）的统一短路响应，不进入工具调用链。"""
        await self._memory.append_turn(session_id, "user", user_text)
        await self._memory.append_turn(session_id, "assistant", answer)
        return AgentChatResponse(
            id=trace_id,
            model=request.model or get_settings().openai_model,
            content=answer,
            trace_id=trace_id,
            usage=self._snapshot_usage(0),
            agent_mode=mode,
            steps=[],
            tool_calls=[],
            degraded=False,
        )

    async def chat(
        self,
        request: AgentChatRequest,
    ) -> AgentChatResponse:
        """执行 Agent 编排对话，返回结构化响应。"""
        trace_id = str(uuid.uuid4())
        _usage_ctx.set(_UsageAccumulator())
        _rag_contexts_ctx.set([])
        session_id = request.conversation_id or trace_id
        user_text = request.messages[-1].content if request.messages else ""
        policy = get_domain_policy(request.domain)

        # 医疗安全门：关键词粗筛 + LLM 二次确认，区分「正在发作/询问知识/既往史」。
        # 仅 emergency 短路返回急诊话术；clarify 追问确认；safe 走正常流程。
        safety_decision = await self._classify_safety(policy, user_text)
        if safety_decision.disposition is RiskDisposition.EMERGENCY:
            return await self._safety_response(
                trace_id, request, session_id, user_text,
                safety_decision.answer or policy.safety_prefix(user_text),
                "safety",
            )
        if safety_decision.disposition is RiskDisposition.CLARIFY:
            return await self._safety_response(
                trace_id, request, session_id, user_text,
                safety_decision.answer or policy.clarify_notice,
                "clarify",
            )
        #    意图识别与模式路由需要历史上下文才能处理追问/指代，如"那它呢？""还有吗？"。
        #    与编排器内部使用的同一份记忆（短期 + 长期语义），读取失败则按单轮处理。
        request_history = [
            f"{message.role}: {message.content}"
            for message in request.messages[:-1]
            if message.content.strip()
        ]
        history: list[str] = list(request_history)
        try:
            memory_history = await self._memory.get_relevant(session_id, user_text, limit=6)
            for item in memory_history:
                if item not in history:
                    history.append(item)
        except Exception as exc:
            logger.warning("读取记忆上下文失败，仅使用请求历史: {}", exc)

        # 1. 意图识别（注入历史上下文）
        intent_result = await self._intent_recognizer.recognize(
            user_text,
            context={"history": history} if history else None,
        )

        # 1b. 模式选择：显式指定直接使用，否则 ModeRouter 自动路由
        _EXPLICIT_MODES = frozenset({"react", "plan_execute"})
        requested_mode = (request.agent_mode or "").strip().lower()

        if requested_mode in _EXPLICIT_MODES:
            selected_mode = requested_mode
            logger.info("使用显式指定模式: {}", selected_mode)
        else:
            decision = await self._mode_router.resolve(
                user_text,
                intent_result,
                history=history or None,
            )
            selected_mode = decision.mode
            logger.info(
                "ModeRouter 自动路由: intent={} conf={:.2f} → mode={} source={}",
                intent_result.intent, intent_result.confidence,
                selected_mode, decision.source,
            )

        intent_ctx = IntentContext(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            slots=intent_result.slots,
            preferred_mode=selected_mode,
            domain=request.domain,
        )

        # 2. 执行编排
        agent_resp = await self._orchestrator.run(
            user_input=user_text,
            session_id=session_id,
            mode=selected_mode,
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

        # 4. 转换步骤为 StepInfo（tool_calls 保留每次调用，去重交给前端展示层）
        steps: list[StepInfo] = []
        tool_calls: list[str] = []
        for s in agent_resp.steps:
            parsed = s.get("parsed") or {}
            action = s.get("action") or parsed.get("action")
            # FC 后端一个 step 可能并行调用多个工具，优先取完整列表；
            # 文本后端每个 step 只有一个 action，回退到单个 action。
            _step_tools = s.get("tools")
            if _step_tools:
                for _t in _step_tools:
                    tool_calls.append(str(_t))
            elif action:
                tool_calls.append(str(action))
            # plan_execute 子任务的工具名透传在 subtask_done 事件的 record.tools
            _record = s.get("record") or {}
            for _t in _record.get("tools") or []:
                tool_calls.append(str(_t))
            # thought: 优先 parsed（react 版），fallback 到 s（langgraph 版）
            thought = str(parsed.get("thought") or s.get("thought") or "")
            steps.append(
                StepInfo(
                    step=int(s.get("step", 0)),
                    phase=str(s.get("phase", "react")),
                    thought=thought,
                    action=str(action) if action else None,
                    action_input=s.get("action_input") or parsed.get("action_input"),
                    observation=str(s.get("observation") or ""),
                    final_answer=str(parsed.get("final_answer") or s.get("final_answer") or ""),
                    error=str(s.get("error") or ""),
                    reflection_score=s.get("reflection_score"),
                )
            )

        # 5. 构建响应（token 用量来自本次请求累计，步骤数为编排真实步数）
        usage = self._snapshot_usage(len(agent_resp.steps))

        # 6. 失败兜底：避免前端只显示"步骤 1"、回答区却一片空白
        content = agent_resp.answer
        if content:
            content = policy.with_boundary(content, user_text)
        if not content and agent_resp.error:
            content = f"Agent 执行失败：{agent_resp.error}"

        # 从反思报告中提取扩展字段
        _reflection = agent_resp.reflection
        reflection_warnings = list(agent_resp.reflection_warnings)
        hallucination_detected = bool(
            _reflection and _reflection.likely_hallucination
        )
        completeness_notes = (
            _reflection.completeness_notes if _reflection else ""
        )

        return AgentChatResponse(
            id=trace_id,
            model=request.model or get_settings().openai_model,
            content=content,
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
            tool_calls=sorted(tool_calls),
            reflection_score=(
                _reflection.quality_score if _reflection else None
            ),
            degraded=agent_resp.degraded,
            reflection_warnings=reflection_warnings,
            hallucination_detected=hallucination_detected,
            completeness_notes=completeness_notes,
            correction_applied=agent_resp.correction_applied,
            retrieved_contexts=list(_rag_contexts_ctx.get() or []),
        )
