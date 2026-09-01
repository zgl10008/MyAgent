# -*- coding: utf-8 -*-
"""WebSearchTool 百度搜索解析与降级行为测试。"""

from __future__ import annotations

import pytest

from app.core.tools.builtin.search import WebSearchTool, parse_baidu_html


_SAMPLE_HTML = """<html>
<body>
<div class="result c-container" id="1">
  <h3 class="c-title t t tts-title">
    <a href="http://www.baidu.com/link?url=abc" data-click="{}">2026世界杯<em>赛程</em>公布</a>
  </h3>
  <span class="content-right_8Zs40">国际足联公布2026年世界杯完整赛程，64场比赛将在北美三国举行。</span>
</div>
<div class="result c-container" id="2">
  <h3 class="c-title">
    <a href="https://example.com/worldcup" data-click="{}">2026世界杯举办地与夺冠热门分析</a>
  </h3>
  <div class="c-abstract">美国、加拿大、墨西哥联合承办，多支强队被视为夺冠热门。</div>
</div>
</body>
</html>
"""


def test_parse_baidu_html_extracts_results():
    results = parse_baidu_html(_SAMPLE_HTML)
    assert len(results) == 2
    assert results[0]["title"] == "2026世界杯赛程公布"
    assert results[0]["url"].startswith("http://www.baidu.com/link?url=")
    assert "国际足联公布" in results[0]["snippet"]
    assert results[1]["title"] == "2026世界杯举办地与夺冠热门分析"
    assert "联合承办" in results[1]["snippet"]


def test_parse_baidu_html_detects_captcha():
    with pytest.raises(RuntimeError, match="安全验证"):
        parse_baidu_html("<html>百度安全验证，请拖动滑块完成验证</html>")


def test_parse_baidu_html_handles_empty_page():
    assert parse_baidu_html("<html><body>没有结果</body></html>") == []


@pytest.mark.asyncio
async def test_empty_query_raises():
    tool = WebSearchTool(timeout=5.0)
    with pytest.raises(ValueError):
        await tool.execute(query="   ")


@pytest.mark.asyncio
async def test_execute_formats_results(monkeypatch):
    tool = WebSearchTool(timeout=5.0)

    async def fake_search(query: str) -> str:
        return f"搜索结果：「{query}」"

    monkeypatch.setattr(tool, "_search_baidu", fake_search)
    assert await tool.execute(query="世界杯") == "搜索结果：「世界杯」"


@pytest.mark.asyncio
async def test_search_baidu_returns_friendly_error_on_failure(monkeypatch):
    tool = WebSearchTool(timeout=5.0)
    called = False

    async def fake_get(url, params=None):
        nonlocal called
        called = True
        raise httpx.ConnectTimeout("boom")

    import httpx

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None):
            return await fake_get(url, params)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())
    result = await tool._search_baidu("世界杯")
    assert called is True
    assert "百度搜索失败" in result
    assert "ConnectTimeout" in result
