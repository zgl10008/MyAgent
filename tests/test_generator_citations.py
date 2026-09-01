# -*- coding: utf-8 -*-
"""RAG 生成器：引用解析与 LLM 输出解析单元测试。"""

from __future__ import annotations

from app.core.rag.generator import RAGGenerator
from app.models.schemas import RetrievalResult


def _generator() -> RAGGenerator:
    return RAGGenerator(llm=None, model_name="test-model")


def _ctx(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(id=f"chunk-{i}", content=f"内容{i}" * 100, source="hybrid")
        for i in range(1, n + 1)
    ]


def test_extract_citations_maps_index_to_context():
    gen = _generator()
    contexts = _ctx(3)
    citations = gen._extract_citations("根据[2]的内容，应低盐饮食。", contexts)
    assert len(citations) == 1
    assert citations[0].index == 2
    assert citations[0].result_id == "chunk-2"


def test_extract_citations_dedup_repeated_refs():
    gen = _generator()
    citations = gen._extract_citations("见[1]和[1]和[2]", _ctx(3))
    assert len(citations) == 2


def test_extract_citations_ignores_out_of_range():
    gen = _generator()
    citations = gen._extract_citations("引用[0]和[99]", _ctx(2))
    assert citations == []


def test_parse_llm_output_variants():
    gen = _generator()
    assert gen._parse_llm_output("直接文本") == "直接文本"
    assert gen._parse_llm_output({"content": "字典文本"}) == "字典文本"
    assert gen._parse_llm_output(None) == ""
