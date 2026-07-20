# -*- coding: utf-8 -*-
"""内置网络搜索工具（基于 SerpAPI）。"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    """通过 SerpAPI 执行 Google 搜索，返回结构化结果。"""

    def __init__(self, api_key: str = "", timeout: float = 15.0) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = "在互联网上搜索关键词并返回摘要文本（新闻、百科、最新信息等）。当你需要实时信息或最新数据时调用此工具。"
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

    async def execute(self, **kwargs: Any) -> str:
        """执行搜索：调用 SerpAPI 获取 Google 搜索结果。"""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        if not self._api_key:
            return "搜索功能未配置 API Key，请在 .env 中设置 SERPAPI_API_KEY。"

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self._api_key,
            "hl": "zh-cn",
            "gl": "cn",
            "num": 5,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # 提取结构化结果
            lines: list[str] = [f"搜索结果：「{query}」"]
            lines.append("=" * 40)

            # 知识图谱（如有）
            kg = data.get("knowledge_graph")
            if kg:
                title = kg.get("title", "")
                desc = kg.get("description", "")
                if title:
                    lines.append(f"\n📖 {title}")
                if desc:
                    lines.append(f"   {desc[:300]}")

            # 有机搜索结果
            organic = data.get("organic_results", [])
            if organic:
                lines.append("")
                for i, r in enumerate(organic[:5], 1):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    link = r.get("link", "")
                    lines.append(f"\n{i}. {title}")
                    if snippet:
                        lines.append(f"   {snippet[:200]}")
                    if link:
                        lines.append(f"   🔗 {link}")

            # 答案框（直接答案）
            answer = data.get("answer_box")
            if answer and isinstance(answer, dict):
                ans_text = answer.get("answer", "") or answer.get("snippet", "")
                if ans_text:
                    lines.append(f"\n✅ 直接答案: {ans_text[:300]}")

            result = "\n".join(lines)
            logger.info("web_search 完成 query={} 结果={}条", query, len(organic))
            return result[:8000]

        except httpx.HTTPStatusError as exc:
            logger.exception("web_search HTTP 错误: {}", exc)
            return f"搜索服务暂时不可用（HTTP {exc.response.status_code}）。"
        except Exception as exc:
            logger.exception("web_search 失败: {}", exc)
            return f"搜索暂时不可用，请稍后重试。错误: {exc!s}"
