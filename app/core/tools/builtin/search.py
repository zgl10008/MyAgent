# -*- coding: utf-8 -*-
"""内置网络搜索工具：百度网页搜索（国内直连，无需 API Key / 代理）。

原实现基于 SerpAPI（国内不可达、需代理且易被风控），于 2026-08-10 决定弃用，
改用百度搜索解析：免费、免 Key、国内直连，适合作为 Agent 的实时信息检索通道。
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


_BAIDU_SEARCH_URL = "https://www.baidu.com/s"
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_RESULT_RE = re.compile(
    r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h3>',
    re.S | re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_JSON_NOISE_MARKERS = ('"}],"', '"},"', '"}]')
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")


def _clean(text: str, tag_sep: str = " ") -> str:
    """去除 HTML 标签与多余空白，还原实体。"""
    text = _TAG_RE.sub(tag_sep, text)
    text = html_lib.unescape(text)
    text = _PUA_RE.sub("", text)  # 去掉百度私有区图标字符，避免编码/显示问题
    return _WS_RE.sub(" ", text).strip()


def parse_baidu_html(html_text: str) -> list[dict[str, str]]:
    """解析百度搜索结果页，返回 [{title, url, snippet}]，最多 5 条。"""
    if "百度安全验证" in html_text or "wappass" in html_text:
        raise RuntimeError("百度返回安全验证页，请降低请求频率或稍后重试")

    # 去掉内嵌 script/style/注释，避免摘要混入页面级 JSON 噪声
    html_text = _SCRIPT_STYLE_RE.sub(" ", html_text)
    html_text = _COMMENT_RE.sub(" ", html_text)

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in _RESULT_RE.finditer(html_text):
        url = html_lib.unescape(m.group(1))
        # 标题去掉标签时不留空格，避免 <em> 高亮标签把词拆开
        title = _clean(m.group(2), tag_sep="")
        if not title or not url or url in seen:
            continue
        seen.add(url)

        # 摘要：取当前标题块结束到下一个标题块之间的文本
        tail = html_text[m.end() :]
        next_h3 = tail.find("<h3", 0)
        chunk = tail if next_h3 == -1 else tail[:next_h3]
        snippet = _clean(chunk)
        for marker in _JSON_NOISE_MARKERS:
            idx = snippet.find(marker)
            if idx != -1:
                snippet = snippet[:idx]
                break
        snippet = snippet[:200]

        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= 5:
            break
    return results


class WebSearchTool(BaseTool):
    """网络搜索：百度网页搜索，国内直连、无需 API Key。"""

    def __init__(self, timeout: float = 8.0) -> None:
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
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")
        return await self._search_baidu(query)

    async def _search_baidu(self, query: str) -> str:
        params = {"wd": query, "rn": 5, "ie": "utf-8"}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=_DEFAULT_HEADERS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(_BAIDU_SEARCH_URL, params=params)
                resp.raise_for_status()
                results = parse_baidu_html(resp.text)
        except RuntimeError as exc:
            logger.warning("百度搜索被验证拦截 query={}: {}", query, exc)
            return f"百度搜索暂不可用：{exc}"
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.warning("百度搜索失败 query={}: {}", query, exc)
            return f"百度搜索失败（{exc.__class__.__name__}），请稍后重试。"
        except Exception as exc:
            logger.exception("百度搜索异常 query={}", query)
            return f"百度搜索异常: {exc!s}"

        if not results:
            return f"百度搜索未找到「{query}」的相关结果。"

        lines = [f"搜索结果：「{query}」", "=" * 40]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r['title']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append(f"   链接: {r['url']}")

        logger.info("百度搜索完成 query={} 结果={}条", query, len(results))
        return "\n".join(lines)[:8000]
