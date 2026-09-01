# -*- coding: utf-8 -*-
"""Cross-Encoder 重排序器单元测试（离线，替换真实模型为假模型）。"""

from __future__ import annotations

import pytest

from app.core.rag.reranker import Reranker
from app.models.schemas import RetrievalResult


class _FakeCrossEncoder:
    """按给定分数返回的 cross-encoder 替身。"""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def predict(self, pairs):
        return self._scores


def _reranker_with_scores(scores: list[float]) -> Reranker:
    """构造一个 _load_model 被替换为假模型的 Reranker。"""
    r = Reranker()
    r._load_model = lambda: _FakeCrossEncoder(scores)  # type: ignore[method-assign]
    return r


@pytest.mark.asyncio
async def test_rerank_empty_documents_returns_empty():
    r = _reranker_with_scores([])
    assert await r.rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
async def test_rerank_top_k_zero_returns_empty():
    r = _reranker_with_scores([1.0])
    assert await r.rerank("q", [RetrievalResult(id="a", content="a")], top_k=0) == []


@pytest.mark.asyncio
async def test_rerank_reorders_and_attaches_score():
    docs = [
        RetrievalResult(id="d1", content="低", score=0.9),
        RetrievalResult(id="d2", content="高", score=0.1),
        RetrievalResult(id="d3", content="中", score=0.5),
    ]
    # cross-encoder 判定：d2 最相关、d1 次之、d3 最差
    r = _reranker_with_scores([0.6, 0.9, 0.3])

    out = await r.rerank("q", docs, top_k=2)

    assert [d.id for d in out] == ["d2", "d1"]
    assert out[0].score == 0.9
    assert out[0].metadata["rerank_score"] == 0.9
    # 原始对象不被原地修改（model_copy deep）
    assert docs[0].metadata == {}
    assert docs[0].score == 0.9
