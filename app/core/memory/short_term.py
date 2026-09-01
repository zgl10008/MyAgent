# -*- coding: utf-8 -*-
"""短期记忆：Redis 滑动窗口 + 超量时 LLM 摘要压缩。"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import tiktoken
from loguru import logger

from app.models.enums import MessageRole
from app.models.schemas import Message


@runtime_checkable
class CompressLLMProtocol(Protocol):
    """用于摘要压缩的 LLM。"""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        ...


class ShortTermMemory:
    """短期记忆：基于 Redis 的滑动窗口 + 自动摘要压缩。"""

    def __init__(
        self,
        redis_client: Any,
        llm: Any,
        window_size: int = 20,
        max_tokens: int = 4000,
    ) -> None:
        """
        :param redis_client: ``redis.asyncio.Redis`` 实例
        :param llm: 用于摘要的模型，需实现 ``ainvoke``
        :param window_size: 最大保留消息条数（角色交替计一条）
        :param max_tokens: 触发按 token 压缩的阈值（近似）
        """
        self._redis = redis_client
        self._llm = llm
        self.window_size = window_size
        self.max_tokens = max_tokens
        self._key_prefix = "stm:session:"
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _count_tokens(self, text: str) -> int:
        """估算 token 数。"""
        if self._encoding is None:
            return max(1, len(text) // 2)  # 粗略估算：平均每 2 个字符一个 token
        return len(self._encoding.encode(text))

    def _serialize(self, msg: Message) -> str:
        payload = {
            "role": msg.role.value,
            "content": msg.content,
            "metadata": msg.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _deserialize(self, raw: str) -> Message:
        try:
            d = json.loads(raw)
            return Message(
                role=MessageRole(d["role"]),
                content=d["content"],
                metadata=d.get("metadata") or {},
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("反序列化消息失败，使用占位: {}", e)
            return Message(role=MessageRole.SYSTEM, content=raw, metadata={"error": "decode"})

    async def get_history(self, session_id: str) -> list[Message]:
        """读取会话全部消息（按时间顺序）。"""
        key = self._key(session_id)
        try:
            raw_list = await self._redis.lrange(key, 0, -1)
        except Exception as e:
            logger.exception("Redis LRANGE 失败: {}", e)
            raise RuntimeError(f"读取短期记忆失败: {e}") from e

        messages: list[Message] = []
        for raw in raw_list:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            messages.append(self._deserialize(str(raw)))
        logger.info(
            "Redis 读取短期记忆 session_id={} count={}",
            session_id,
            len(messages),
        )
        return messages

    async def delete(self, session_id: str) -> None:
        """删除会话的短期记忆键（用于删除会话时同步清理 Redis）。"""
        key = self._key(session_id)
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.exception("Redis DEL 失败: {}", e)
            raise RuntimeError(f"删除短期记忆失败: {e}") from e

    async def add_message(self, session_id: str, message: Message) -> None:
        """追加一条消息并视需要触发压缩。"""
        key = self._key(session_id)
        try:
            await self._redis.rpush(key, self._serialize(message))
            logger.info(
                "Redis 写入短期记忆 session_id={} role={} content_len={}",
                session_id,
                message.role.value,
                len(message.content or ""),
            )
        except Exception as e:
            logger.exception("Redis RPUSH 失败: {}", e)
            raise RuntimeError(f"写入短期记忆失败: {e}") from e

        await self._compress_if_needed(session_id)

    async def _compress_if_needed(self, session_id: str) -> None:
        """当消息数超过窗口或总 token 过大时，自动摘要压缩。"""
        key = self._key(session_id)
        try:
            n = await self._redis.llen(key)
        except Exception as e:
            logger.exception("Redis LLEN 失败: {}", e)
            return

        msgs = await self.get_history(session_id)
        total_tokens = sum(self._count_tokens(m.content) for m in msgs)

        if n <= self.window_size and total_tokens <= self.max_tokens:
            return

        if len(msgs) < 2:
            return

        # 保留尾部若干条，对其余做摘要；keep 不超过当前长度减一
        keep = max(2, self.window_size // 2)
        if keep >= len(msgs):
            keep = len(msgs) - 1
        to_summarize = msgs[:-keep]
        tail = msgs[-keep:]

        summary_text = await self._summarize_messages(to_summarize)
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[历史摘要]\n{summary_text}",
            metadata={"compressed_from": len(to_summarize)},
        )

        try:
            await self._redis.delete(key)
            combined = [summary_msg, *tail]
            for m in combined:
                await self._redis.rpush(key, self._serialize(m))
        except Exception as e:
            logger.exception("压缩重写 Redis 列表失败: {}", e)
            raise RuntimeError(f"记忆压缩失败: {e}") from e

        logger.info("会话 {} 已压缩，摘要 {} 条历史，保留 {} 条", session_id, len(to_summarize), len(tail))

    async def _summarize_messages(self, messages: list[Message]) -> str:
        """调用 LLM 生成摘要。"""
        lines = [f"{m.role.value}: {m.content}" for m in messages]
        prompt = (
            "请将以下对话压缩为简洁中文摘要，保留关键事实与用户意图：\n\n"
            + "\n".join(lines)
        )
        if not isinstance(self._llm, CompressLLMProtocol) and not callable(
            getattr(self._llm, "ainvoke", None)
        ):
            # 无可用 LLM 时退回截断
            return "\n".join(lines)[:2000]

        try:
            raw = await self._llm.ainvoke(prompt)
        except Exception as e:
            logger.exception("摘要 LLM 调用失败: {}", e)
            return "\n".join(lines)[:2000]

        if hasattr(raw, "content"):
            return str(getattr(raw, "content", "")).strip()
        if isinstance(raw, dict) and "content" in raw:
            return str(raw["content"]).strip()
        return str(raw).strip()

    # ------------------------------------------------------------------
    # MemoryLike 协议适配方法
    # ------------------------------------------------------------------

    async def get_relevant(
        self, session_id: str, query: str, limit: int = 8
    ) -> list[str]:
        """满足 MemoryLike / MemoryManager 协议：返回格式化历史字符串。"""
        logger.info(
            "[记忆-读] ShortTermMemory.get_relevant session_id={} limit={}",
            session_id,
            limit,
        )
        try:
            msgs = await self.get_history(session_id)
        except Exception:
            logger.exception("Redis 读取记忆失败")
            return []
        recent = msgs[-limit:]
        result = [f"{m.role.value}: {m.content}" for m in recent]
        logger.info(
            "[记忆-读] ShortTermMemory.get_relevant 结果 session_id={} count={}",
            session_id,
            len(result),
        )
        return result

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """满足 MemoryLike / MemoryManager 协议：直接入库。"""
        from app.models.enums import MessageRole as MR

        logger.info(
            "[记忆-写] ShortTermMemory.append_turn session_id={} role={} content_len={}",
            session_id,
            role,
            len(content or ""),
        )
        msg = Message(
            role=MR(role),
            content=content,
            metadata=metadata or {},
        )
        await self.add_message(session_id, msg)
