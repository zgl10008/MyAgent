# -*- coding: utf-8 -*-
"""内置网络搜索工具：SerpAPI + 代理支持。"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    """网络搜索：SerpAPI 优先。可通过 SEARCH_PROXY 环境变量配置代理。"""

    def __init__(self, api_key: str = "", timeout: float = 8.0) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = (
            "在互联网上搜索关键词并返回摘要文本（新闻、百科、最新信息等）。"
            "当你需要实时信息或最新数据时调用此工具。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词或完整问句",
                required=True,
            )
        ]
        self._api_key = api_key
        self._timeout = timeout
        import os

        self._proxy = (
            os.environ.get("SEARCH_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or None
        )

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._proxy:
            kwargs["proxy"] = self._proxy
            logger.info("搜索使用代理: {}", self._proxy)
        return httpx.AsyncClient(**kwargs)

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        # ── SerpAPI ──
        if self._api_key:
            return await self._search_serpapi(query)

        return (
            "搜索功能未配置 API Key。请在 .env 中设置 SERPAPI_API_KEY（免费注册 https://serpapi.com）。\n"
            "如果 SerpAPI 超时，可在 .env 中添加 SEARCH_PROXY=http://你的代理地址:端口。"
        )

    async def _search_serpapi(self, query: str) -> str:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self._api_key,
            "hl": "zh-cn",
            "gl": "cn",
            "num": 5,
        }
        try:
            async with self._client() as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            lines = [f"搜索结果：「{query}」", "=" * 40]

            kg = data.get("knowledge_graph")
            if kg and kg.get("title"):
                lines.append(f"\n📖 {kg['title']}")
                if kg.get("description"):
                    lines.append(f"   {kg['description'][:300]}")

            organic = data.get("organic_results", [])
            if organic:
                lines.append("")
                for i, r in enumerate(organic[:5], 1):
                    lines.append(f"\n{i}. {r.get('title', '')}")
                    if r.get("snippet"):
                        lines.append(f"   {r['snippet'][:200]}")
                    if r.get("link"):
                        lines.append(f"   🔗 {r['link']}")

            ans = data.get("answer_box") or {}
            if isinstance(ans, dict):
                ans_text = ans.get("answer") or ans.get("snippet") or ""
                if ans_text:
                    lines.append(f"\n✅ 直接答案: {ans_text[:300]}")

            logger.info("SerpAPI 完成 query={} 结果={}条", query, len(organic))
            return "\n".join(lines)[:8000]

        except httpx.ConnectTimeout:
            msg = (
                f"搜索超时：无法连接 SerpAPI。\n"
                f"如果使用了代理，请在 .env 中设置 SEARCH_PROXY。\n"
                f"没有代理的话，可访问 https://duckduckgo.com/?q={query} 手动搜索。"
            )
            logger.warning("SerpAPI 连接超时 query={}", query)
            return msg
        except httpx.HTTPStatusError as exc:
            logger.warning("SerpAPI HTTP {}: {}", exc.response.status_code, exc)
            return f"搜索服务返回错误（HTTP {exc.response.status_code}）。"
        except Exception as exc:
            logger.warning("SerpAPI 失败: {}", exc)
            return f"搜索暂时不可用。错误: {exc!s}"
