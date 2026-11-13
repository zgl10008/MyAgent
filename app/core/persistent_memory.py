# -*- coding: utf-8 -*-
"""
统一持久化记忆管理器：Redis 热路径 + PostgreSQL 冷存储 + 内存降级。

同时满足：
- orchestrator.MemoryManager / react_agent.MemoryLike 协议
- sessions.py 路由的会话管理接口
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from app.core.memory.short_term import ShortTermMemory
from app.core.session_repository import SessionRepository
from app.models.enums import MessageRole
from app.models.schemas import Message


# ==============================================================================
# 内存版记忆管理器（最终降级兜底，无需任何外部基础设施）
# ==============================================================================


class _InMemoryMemoryManager:
    """内存版记忆管理器：用 Python dict 存储会话消息。

    同时满足 orchestrator.MemoryManager 和 react_agent.MemoryLike 协议。
    当 Redis / PostgreSQL 不可用时作为最终降级方案。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._session_meta: dict[str, dict[str, Any]] = {}

    def _ensure_meta(self, session_id: str) -> None:
        if session_id not in self._session_meta:
            self._session_meta[session_id] = {
                "title": "新对话",
                "created_at": time.time(),
                "updated_at": time.time(),
            }

    def _update_title(self, session_id: str, content: str) -> None:
        meta = self._session_meta.get(session_id)
        if meta and meta["title"] == "新对话":
            title = content.strip()[:30].replace("\n", " ")
            meta["title"] = title if title else "新对话"
            meta["updated_at"] = time.time()

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for sid, meta in self._session_meta.items():
            sessions.append({
                "id": sid,
                "title": meta["title"],
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "message_count": len(self._sessions.get(sid, [])),
            })
        sessions.sort(key=lambda s: -s["updated_at"])
        return sessions

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return self._sessions.get(session_id, [])

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._session_meta.pop(session_id, None)
            return True
        return False

    async def get_relevant(
        self, session_id: str, query: str, limit: int = 8
    ) -> list[str]:
        history = self._sessions.get(session_id, [])
        recent = history[-limit:]
        result = [f"{m['role']}: {m['content']}" for m in recent]
        logger.info(
            "[记忆-读] InMemoryMemoryManager.get_relevant session_id={} count={}",
            session_id,
            len(result),
        )
        return result

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_meta(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )
        self._session_meta[session_id]["updated_at"] = time.time()
        if role == "user":
            self._update_title(session_id, content)
        logger.info(
            "[记忆-写] InMemoryMemoryManager.append_turn session_id={} role={} content_len={}",
            session_id,
            role,
            len(content or ""),
        )

# ==============================================================================
# LLM 适配器：将 ModelRouter.acomplete 包装为 ShortTermMemory 需要的 ainvoke
# ==============================================================================


class _CompressLLMAdapter:
    """将 orchestrator 侧的 LLM（acomplete）适配为 ShortTermMemory 的压缩 LLM（ainvoke）。"""

    def __init__(self, model_router: Any) -> None:
        self._router = model_router

    async def ainvoke(self, input: str, **kwargs: Any) -> Any:
        """将摘要 prompt 发给 LLM，返回类 LangChain 结果对象。"""
        llm = self._router.get_llm("react")
        if hasattr(llm, "acomplete"):
            content = await llm.acomplete(
                [{"role": "user", "content": input}],
                temperature=0.1,
            )
        else:
            content = str(input)[:2000]

        # 返回一个简单的对象，模拟 LangChain AIMessage
        class _FakeAIMessage:
            def __init__(self, c: str) -> None:
                self.content = c

        return _FakeAIMessage(content)


# ==============================================================================
# PersistentMemoryManager
# ==============================================================================


class PersistentMemoryManager:
    """统一持久化记忆管理器。

    - Redis ShortTermMemory → 热路径（滑动窗口 + LLM 摘要压缩）
    - PostgreSQL SessionRepository → 冷存储（持久化）
    - _InMemoryMemoryManager → 最终降级兜底
    """

    def __init__(
        self,
        redis_client: Any | None,
        compress_llm: Any | None,
        pg_repo: SessionRepository,
    ) -> None:
        self._pg = pg_repo
        self._redis_ok = redis_client is not None
        self._pg_ok = True  # 先乐观标记，首次调用失败再降级
        self._pg_cooldown = 5.0  # 秒；PG 失败后的重试冷却期
        self._pg_retry_at = 0.0  # 下次允许重试 PG 的 monotonic 时间戳
        self._long_term: Any = None  # 可选：LongTermMemory（Milvus 语义召回）

        # Redis 短期记忆（滑动窗口 + 压缩）
        if self._redis_ok and compress_llm is not None:
            try:
                self._redis_stm = ShortTermMemory(
                    redis_client=redis_client,
                    llm=compress_llm,
                )
                logger.info("持久化记忆：Redis ShortTermMemory 已就绪")
            except Exception as exc:
                logger.warning("Redis ShortTermMemory 初始化失败: {}", exc)
                self._redis_ok = False
                self._redis_stm = None
        else:
            self._redis_ok = False
            self._redis_stm = None

        # 内存降级（始终可用）
        self._fallback = _InMemoryMemoryManager()

        logger.info(
            "PersistentMemoryManager 就绪 redis={} pg={}",
            self._redis_ok,
            self._pg_ok,
        )

    # ------------------------------------------------------------------
    # 长期记忆注入（可选）
    # ------------------------------------------------------------------

    def set_long_term(self, long_term: Any) -> None:
        """注入 LongTermMemory 实例，启用语义召回。"""
        self._long_term = long_term
        logger.info("长期语义记忆已注入 PersistentMemoryManager")

    # ------------------------------------------------------------------
    # PG 健康状态（可恢复降级：失败进入冷却期，冷却结束后自动重试，
    # 而非一次性永久降级。这样单个脏入参或瞬时抖动不会把整个持久化层关死。）
    # ------------------------------------------------------------------

    def _pg_available(self) -> bool:
        """PG 当前是否可尝试：健康，或冷却期已过自动恢复。"""
        if not self._pg_ok and time.monotonic() >= self._pg_retry_at:
            self._pg_ok = True
            logger.info("PG 冷却期结束，恢复对 PostgreSQL 的访问")
        return self._pg_ok

    def _pg_failed(self) -> None:
        """PG 调用失败：进入冷却期，冷却后可自动恢复。"""
        self._pg_ok = False
        self._pg_retry_at = time.monotonic() + self._pg_cooldown

    # ------------------------------------------------------------------
    # MemoryLike 协议（供 Orchestrator / ReActAgent / PlannerAgent）
    # ------------------------------------------------------------------

    async def get_relevant(
        self, session_id: str, query: str, limit: int = 8
    ) -> list[str]:
        """获取与查询相关的历史记忆片段。优先级：Redis → PG → 内存 + 长期语义。"""
        items: list[str] = []
        logger.info(
            "记忆读取开始 session_id={} limit={}",
            session_id,
            limit,
        )

        # 1) 尝试 Redis（热路径）
        if self._redis_ok and self._redis_stm is not None:
            try:
                items = await self._redis_stm.get_relevant(session_id, query, limit)
                logger.info("记忆读取 Redis session_id={} count={}", session_id, len(items))
                if items:
                    # 叠加长期语义记忆后返回
                    result = await self._append_long_term(items, session_id, query, limit)
                    logger.info("记忆读取最终结果 session_id={} count={}", session_id, len(result))
                    return result
            except Exception as exc:
                logger.warning("Redis get_relevant 失败，降级 PG: {}", exc)
                self._redis_ok = False

        # 2) 尝试 PostgreSQL
        if self._pg_available():
            try:
                items = await self._pg.get_recent_messages(session_id, limit)
                logger.info("记忆读取 PostgreSQL session_id={} count={}", session_id, len(items))
                if items:
                    result = await self._append_long_term(items, session_id, query, limit)
                    logger.info("记忆读取最终结果 session_id={} count={}", session_id, len(result))
                    return result
            except Exception as exc:
                logger.warning("PG get_recent_messages 失败，本次降级内存: {}", exc)
                self._pg_failed()

        # 3) 最终兜底
        items = await self._fallback.get_relevant(session_id, query, limit)
        logger.info("记忆读取内存降级 session_id={} count={}", session_id, len(items))
        result = await self._append_long_term(items, session_id, query, limit)
        logger.info("记忆读取最终结果 session_id={} count={}", session_id, len(result))
        return result

    async def _append_long_term(
        self,
        short_items: list[str],
        session_id: str,
        query: str,
        limit: int,
    ) -> list[str]:
        """如果启用了长期语义记忆，将召回结果插入最前面。"""
        if self._long_term is None:
            return short_items
        try:
            ltm_items = await self._long_term.recall(query, session_id, top_k=3)
            if ltm_items:
                logger.info(
                    "记忆读取 Milvus session_id={} count={}",
                    session_id,
                    len(ltm_items),
                )
                semantic = [
                    f"[历史记忆] {item.content}"
                    for item in ltm_items
                    if hasattr(item, "content")
                ]
                # 语义记忆插入最前面（重要性最高），然后接短期记忆，最后截断
                combined = semantic + short_items
                return combined[:limit]
        except Exception as exc:
            logger.warning("长期语义记忆召回失败，跳过: {}", exc)
        return short_items

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """多写：PostgreSQL（主）+ Redis（辅）+ 内存兜底 + 长期语义记忆（Milvus）。"""
        # 内存兜底始终同步写入
        await self._fallback.append_turn(session_id, role, content, metadata)
        logger.info(
            "记忆写入内存 session_id={} role={} content_len={}",
            session_id,
            role,
            len(content or ""),
        )

        # PostgreSQL 主路径
        if self._pg_available():
            try:
                await self._pg.add_message(session_id, role, content, metadata)
                logger.info("记忆写入 PostgreSQL 成功 session_id={} role={}", session_id, role)
            except Exception as exc:
                self._pg_failed()
                logger.warning("PG append_turn 失败，本次降级内存: {}", exc)

        # Redis 辅助路径
        if self._redis_ok and self._redis_stm is not None:
            try:
                await self._redis_stm.append_turn(session_id, role, content, metadata)
                logger.info("记忆写入 Redis 成功 session_id={} role={}", session_id, role)
            except Exception as exc:
                logger.warning("Redis append_turn 失败: {}", exc)

        # 长期语义记忆（Milvus）：写入旁路，失败不影响主流程。
        # 此前只有召回（_append_long_term）接通，写入（store）从未被调用；
        # 本次把每轮问答写入语义库，使"很久以前但语义相关的记忆"可被召回。
        await self._append_long_term_turn(session_id, role, content, metadata)

    async def _append_long_term_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """写入长期语义记忆：只存有内容的 user / assistant 消息，失败静默降级。"""
        if self._long_term is None:
            return
        text = (content or "").strip()
        if not text or role not in ("user", "assistant"):
            return
        try:
            await self._long_term.store(
                session_id,
                text,
                dict(metadata or {}),
            )
            logger.info(
                "记忆写入 Milvus 成功 session_id={} role={} content_len={}",
                session_id,
                role,
                len(text),
            )
        except Exception as exc:
            logger.warning("长期语义记忆写入失败，跳过: {}", exc)

    # ------------------------------------------------------------------
    # 会话管理接口（供 sessions.py 路由）
    # ------------------------------------------------------------------

    async def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话。优先级：PG → 内存。"""
        if self._pg_available():
            try:
                return await self._pg.list_sessions()
            except Exception as exc:
                logger.warning("PG list_sessions 失败，本次降级内存: {}", exc)
                self._pg_failed()
        return self._fallback.list_sessions()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话详情（元数据 + 消息列表）。优先级：PG → 内存。"""
        if self._pg_available():
            try:
                result = await self._pg.get_session(session_id)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning("PG get_session 失败，本次降级内存: {}", exc)
                self._pg_failed()
        # 内存降级：从 fallback 组装
        history = self._fallback.get_session_history(session_id)
        sessions = self._fallback.list_sessions()
        meta = next((s for s in sessions if s["id"] == session_id), None)
        if not meta:
            return None
        return {
            "id": session_id,
            "title": meta.get("title", "新对话"),
            "created_at": meta.get("created_at", 0.0),
            "updated_at": meta.get("updated_at", 0.0),
            "messages": [
                {"role": m["role"], "content": m["content"], "metadata": m.get("metadata", {})}
                for m in history
            ],
        }

    async def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """获取会话消息列表。优先级：PG → 内存。"""
        session = await self.get_session(session_id)
        if session is not None:
            return session.get("messages", [])
        return []

    async def delete_session(self, session_id: str) -> bool:
        """删除会话：PG + Redis + 内存三方同步删除。"""
        ok = self._fallback.delete_session(session_id)

        if self._pg_available():
            try:
                pg_ok = await self._pg.delete_session(session_id)
                ok = ok or pg_ok
            except Exception as exc:
                logger.warning("PG delete_session 失败: {}", exc)

        if self._redis_ok and self._redis_stm is not None:
            try:
                await self._redis_stm.delete(session_id)
                logger.info("记忆删除 Redis 成功 session_id={}", session_id)
            except Exception as exc:
                logger.warning("Redis delete_session 失败: {}", exc)

        return ok
