# -*- coding: utf-8 -*-
"""PostgreSQL 会话持久化仓库：直接使用 async_session_factory，不依赖 FastAPI Depends。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import Conversation, Message
from app.infrastructure.database import session as db_session


class SessionRepository:
    """PostgreSQL 会话持久化仓库。

    每个方法独立获取/释放 session，不依赖 HTTP 请求上下文，
    可直接被 AgentService 单例调用。
    """

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    async def _get_session() -> AsyncSession:
        """从全局工厂获取一个异步 session（通过模块引用，确保获取最新值）。"""
        factory = db_session.async_session_factory
        if factory is None:
            raise RuntimeError("数据库未初始化，请先调用 configure_session(init_engine(...))")
        return factory()

    # ------------------------------------------------------------------
    # 会话 CRUD
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str, title: str = "新对话") -> None:
        """创建新会话记录（如已存在则忽略）。"""
        try:
            sess = await self._get_session()
            async with sess:
                existing = await sess.get(Conversation, session_id)
                if existing is not None:
                    return
                conv = Conversation(
                    id=session_id,
                    title=title,
                    created_at=self._now(),
                    updated_at=self._now(),
                )
                sess.add(conv)
                await sess.commit()
        except Exception as exc:
            logger.warning("创建会话 PG 记录失败: {}", exc)

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """按 updated_at 降序返回会话摘要列表。"""
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(Conversation)
                    .order_by(Conversation.updated_at.desc())
                    .limit(limit),
                )
                rows = result.scalars().all()
                out: list[dict[str, Any]] = []
                for c in rows:
                    # 统计消息数
                    count_result = await sess.execute(
                        select(Message).where(Message.conversation_id == c.id),
                    )
                    msg_count = len(count_result.scalars().all())
                    out.append({
                        "id": c.id,
                        "title": c.title or "新对话",
                        "created_at": c.created_at.timestamp() if c.created_at else 0.0,
                        "updated_at": c.updated_at.timestamp() if c.updated_at else 0.0,
                        "message_count": msg_count,
                    })
                return out
        except Exception as exc:
            logger.exception("从 PG 列出会话失败: {}", exc)
            raise

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话详情（元数据 + 消息列表）。"""
        try:
            sess = await self._get_session()
            async with sess:
                conv = await sess.get(Conversation, session_id)
                if conv is None:
                    return None
                result = await sess.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.asc()),
                )
                msgs = result.scalars().all()
                return {
                    "id": conv.id,
                    "title": conv.title or "新对话",
                    "created_at": conv.created_at.timestamp() if conv.created_at else 0.0,
                    "updated_at": conv.updated_at.timestamp() if conv.updated_at else 0.0,
                    "messages": [
                        {
                            "role": m.role,
                            "content": m.content,
                            "metadata": m.meta or {},
                        }
                        for m in msgs
                    ],
                }
        except Exception as exc:
            logger.exception("从 PG 获取会话失败: {}", exc)
            raise

    async def update_session_title(self, session_id: str, title: str) -> None:
        """更新会话标题。"""
        try:
            sess = await self._get_session()
            async with sess:
                await sess.execute(
                    update(Conversation)
                    .where(Conversation.id == session_id)
                    .values(title=title, updated_at=self._now()),
                )
                await sess.commit()
        except Exception as exc:
            logger.warning("更新 PG 会话标题失败: {}", exc)

    async def delete_session(self, session_id: str) -> bool:
        """删除会话（级联删除消息）。"""
        try:
            sess = await self._get_session()
            async with sess:
                conv = await sess.get(Conversation, session_id)
                if conv is None:
                    return False
                await sess.delete(conv)
                await sess.commit()
                return True
        except Exception as exc:
            logger.exception("从 PG 删除会话失败: {}", exc)
            raise

    # ------------------------------------------------------------------
    # 消息 CRUD
    # ------------------------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加一条消息，自动更新会话 updated_at 与标题（首条 user 消息）。"""
        logger.info(
            "[记忆-写] PG add_message session_id={} role={} content_len={}",
            session_id,
            role,
            len(content or ""),
        )
        try:
            sess = await self._get_session()
            async with sess:
                # 确保会话存在
                conv = await sess.get(Conversation, session_id)
                if conv is None:
                    conv = Conversation(
                        id=session_id,
                        title="新对话",
                        created_at=self._now(),
                        updated_at=self._now(),
                    )
                    sess.add(conv)

                # 首条 user 消息 → 自动标题
                if conv.title == "新对话" and role == "user":
                    title = content.strip()[:30].replace("\n", " ")
                    conv.title = title if title else "新对话"

                conv.updated_at = self._now()

                msg = Message(
                    conversation_id=session_id,
                    role=role,
                    content=content,
                    meta=metadata or {},
                    created_at=self._now(),
                )
                sess.add(msg)
                await sess.commit()
                logger.info("[记忆-写] PG add_message 提交成功 session_id={}", session_id)
        except Exception as exc:
            logger.warning("写入消息到 PG 失败: {}", exc)

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取会话消息（按时间升序）。"""
        logger.info(
            "[记忆-读] PG get_messages session_id={} limit={}",
            session_id,
            limit,
        )
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.asc())
                    .limit(limit),
                )
                rows = result.scalars().all()
                items = [
                    {
                        "role": m.role,
                        "content": m.content,
                        "metadata": m.meta or {},
                    }
                    for m in rows
                ]
                logger.info(
                    "[记忆-读] PG get_messages 结果 session_id={} count={}",
                    session_id,
                    len(items),
                )
                return items
        except Exception as exc:
            logger.exception("从 PG 读取消息失败: {}", exc)
            raise

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[str]:
        """获取最近 N 条消息，格式化为 'role: content' 字符串（供 MemoryLike 使用）。"""
        logger.info(
            "[记忆-读] PG get_recent_messages session_id={} limit={}",
            session_id,
            limit,
        )
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(limit),
                )
                rows = result.scalars().all()
                rows.reverse()  # 恢复时间升序
                items = [f"{m.role}: {m.content}" for m in rows]
                logger.info(
                    "[记忆-读] PG get_recent_messages 结果 session_id={} count={}",
                    session_id,
                    len(items),
                )
                return items
        except Exception as exc:
            logger.exception("从 PG 读取最近消息失败: {}", exc)
            raise
