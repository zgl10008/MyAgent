# -*- coding: utf-8 -*-
"""多路检索（BM25 / RRF 融合）单元测试。"""

from __future__ import annotations

import pytest

from app.core.rag.retriever import MultiRetriever, _BM25Index
from app.models.schemas import RetrievalResult


def test_bm25_ranks_relevant_doc_first():
    index = _BM25Index()
    index.add_document("d1", "高血压患者饮食应低盐")
    index.add_document("d2", "高血压的诊断标准是 140/90 mmHg")
    index.add_document("d3", "糖尿病需要控制血糖")
    ranked = index.search("高血压诊断标准", top_k=3)
    assert ranked[0][0] == "d2"


def test_bm25_empty_index_returns_empty():
    index = _BM25Index()
    assert index.search("高血压", 5) == []


def test_bm25_clear_resets_index():
    index = _BM25Index()
    index.add_document("d1", "高血压饮食")
    index.clear()
    assert index._N == 0
    assert index.search("高血压", 5) == []


def _make_retriever() -> MultiRetriever:
    return MultiRetriever(milvus_client=None, embedding_model=None)


def _item(rid: str, score: float) -> RetrievalResult:
    return RetrievalResult(id=rid, content=f"content-{rid}", score=score, source="vector")


def test_rrf_fuse_merges_and_boosts_overlap():
    retriever = _make_retriever()
    vec = [_item("b", 0.9), _item("a", 0.8), _item("c", 0.7)]
    kw = [_item("b", 5.0), _item("a", 4.0)]
    fused = retriever._rrf_fuse([vec, kw], top_k=3)
    assert fused[0].id == "b"  # 两路都排第 1，RRF 分数最高
    assert fused[0].source == "hybrid"
    assert {r.id for r in fused} == {"a", "b", "c"}


def test_rrf_fuse_handles_empty_lists():
    retriever = _make_retriever()
    assert retriever._rrf_fuse([[], []], top_k=5) == []
