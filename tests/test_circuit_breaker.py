# -*- coding: utf-8 -*-
"""三态熔断器单元测试：Closed → Open → Half-Open 状态机。"""

from __future__ import annotations

import asyncio

import pytest

from app.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitState


async def _ok() -> str:
    return "ok"


async def _fail() -> str:
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_initial_state_is_closed():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_success_keeps_closed_and_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_fail)
    assert cb.state == CircuitState.CLOSED
    assert await cb.call(_ok) == "ok"
    assert cb._failure_count == 0


@pytest.mark.asyncio
async def test_failure_threshold_trips_to_open():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_fail)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_rejects_calls_before_recovery_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN
    with pytest.raises(RuntimeError, match="OPEN"):
        await cb.call(_ok)


@pytest.mark.asyncio
async def test_half_open_success_closes_breaker():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN
    await asyncio.sleep(0.15)
    assert await cb.call(_ok) == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_breaker():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    await asyncio.sleep(0.15)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_attempt_limit():
    cb = CircuitBreaker(half_open_max=2)
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_attempts = 2
    with pytest.raises(RuntimeError, match="Half-Open"):
        await cb.call(_ok)


@pytest.mark.asyncio
async def test_open_stays_open_until_recovery_elapsed():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    # 手动把失败时间改为刚刚，模拟仍在恢复窗口内
    import time

    cb._last_failure_time = time.monotonic()
    with pytest.raises(RuntimeError, match="OPEN"):
        await cb.call(_ok)
    assert cb.state == CircuitState.OPEN
