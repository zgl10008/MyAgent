# -*- coding: utf-8 -*-
"""工具注册中心单元测试。"""

from __future__ import annotations

import pytest

from app.core.tools.base import BaseTool, ToolParameter
from app.core.tools.registry import ToolRegistry


class _DummyTool(BaseTool):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__()
        self.name = name
        self.description = "测试工具"
        self.parameters = [
            ToolParameter(name="q", type="string", description="查询", required=True)
        ]

    async def execute(self, **kwargs) -> str:
        return "ok"


def test_register_and_get():
    reg = ToolRegistry()
    tool = _DummyTool()
    reg.register(tool)
    assert reg.get_tool("dummy") is tool


def test_get_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get_tool("nope")


def test_get_all_tools_order():
    reg = ToolRegistry()
    a, b = _DummyTool("a"), _DummyTool("b")
    reg.register(a)
    reg.register(b)
    assert reg.get_all_tools() == [a, b]


def test_register_overwrites():
    reg = ToolRegistry()
    a, b = _DummyTool("same"), _DummyTool("same")
    reg.register(a)
    reg.register(b)
    assert reg.get_tool("same") is b


def test_tools_description_includes_name_and_params():
    reg = ToolRegistry()
    reg.register(_DummyTool())
    desc = reg.get_tools_description()
    assert "dummy" in desc
    assert "q: string" in desc


def test_empty_description():
    reg = ToolRegistry()
    assert "无可用工具" in reg.get_tools_description()
