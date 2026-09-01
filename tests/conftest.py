# -*- coding: utf-8 -*-
"""pytest 全局配置：保证测试环境稳定、离线可跑。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 必须在导入 app.* 之前设置：
# 1) .env 中 DEBUG=release 不是合法布尔值，覆盖为 false
# 2) 禁止 HuggingFace 联网，使用本地模型缓存
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_pyfunc_call(pyfuncitem):
    """轻量异步支持：将 async 测试函数用 asyncio.run 执行，无需 pytest-asyncio。"""
    testfunction = pyfuncitem.obj
    if asyncio.iscoroutinefunction(testfunction):
        funcargs = pyfuncitem.funcargs
        testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
        result = asyncio.run(testfunction(**testargs))
        pyfuncitem._result = result
        return True
    return None


class FakeRedis:
    """最小 Redis 替身：支持短期记忆测试所需的方法。"""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}

    async def rpush(self, key: str, value: str) -> int:
        self._lists.setdefault(key, []).append(value)
        return len(self._lists[key])

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        items = self._lists.get(key, [])
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    async def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    async def delete(self, key: str) -> int:
        existed = key in self._lists
        self._lists.pop(key, None)
        return 1 if existed else 0


class FakeLLM:
    """可配置返回内容的 LLM 替身。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[dict]] = []

    async def acomplete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return "Final Answer: ok"

    async def ainvoke(self, input):
        return _ContentObj("测试摘要")


class _ContentObj:
    def __init__(self, content: str) -> None:
        self.content = content
