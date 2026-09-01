# -*- coding: utf-8 -*-
"""RAG 知识库检索工具：让 Agent 在 ReAct 循环中直接查询知识库。"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


# RAG 检索回调签名：async (query: str, top_k: int) -> list[RetrievalResult]
RAGSearchFunc = Callable[[str, int], Awaitable[list[Any]]]


class RAGSearchTool(BaseTool):
    """知识库检索工具：基于向量+关键词混合检索，在已上传的文档中查找相关信息。

    此工具将 RAG 从独立端点转变为 Agent 可调度的能力——
    Agent 在 ReAct 循环中自主决定何时查知识库、何时搜网络、何时两者交叉验证。
    """

    def __init__(self, search_func: RAGSearchFunc, chunks_count: Callable[[], int] | None = None) -> None:
        """
        :param search_func: 异步检索回调，签名 async (query, top_k) -> list[RetrievalResult]
        :param chunks_count: 可选，返回知识库当前分块数（用于 prompt 提示）
        """
        super().__init__()
        self.name = "rag_search"
        self.description = (
            "在已上传的文档知识库中检索相关信息。"
            "适用场景：用户询问特定领域的专业问题、需要查找已上传的文档内容时。"
            "与 web_search 的区别：rag_search 查的是用户私有文档（如医学指南、产品手册），"
            "web_search 查的是公开互联网信息。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="检索关键词或问句。建议提取核心概念作为查询词，而非整句复制。",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回的文档块数量，默认 5。信息不充分时可增加到 8-10。",
                required=False,
            ),
        ]
        self._search_func = search_func
        self._chunks_count = chunks_count

    async def execute(self, **kwargs: Any) -> str:
        """执行知识库检索并返回格式化的结果文本。"""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "错误：rag_search 需要提供 query 参数。"

        top_k = int(kwargs.get("top_k", 5))
        top_k = max(1, min(top_k, 20))  # 限制 1-20

        # 检查知识库是否为空
        if self._chunks_count is not None:
            count = self._chunks_count()
            if count == 0:
                return (
                    "知识库当前为空，没有已上传的文档。"
                    "请告知用户：知识库中暂无文档，建议上传相关文档或使用 web_search 搜索公开信息。"
                )

        try:
            results = await self._search_func(query, top_k)
        except Exception as exc:
            logger.exception("RAG 检索失败")
            return f"知识库检索失败: {exc}。请尝试使用 web_search 或直接基于已有知识回答。"

        if not results:
            return (
                f"在知识库中未找到与「{query}」直接相关的内容。"
                "建议：1) 换个关键词重试 2) 使用 web_search 搜索公开信息 3) 直接告知用户未找到相关文档。"
            )

        # 格式化检索结果
        lines = [f"知识库检索结果：「{query}」", "=" * 40]
        for i, r in enumerate(results, 1):
            content = r.content if hasattr(r, "content") else str(r)
            score = getattr(r, "score", 0.0)
            source = getattr(r, "source", "")
            score_str = f" (相关度: {score:.4f})" if score else ""
            source_str = f" [{source}]" if source else ""
            # 截断过长文本
            display = content[:500] + ("..." if len(content) > 500 else "")
            lines.append(f"\n--- 结果 {i}{source_str}{score_str} ---")
            lines.append(display)

        lines.append(f"\n{'=' * 40}")
        lines.append(f"共找到 {len(results)} 条相关文档块。请基于以上内容作答，必要时引用具体片段。")

        return "\n".join(lines)[:8000]  # 8K 上限，防止覆盖历史
