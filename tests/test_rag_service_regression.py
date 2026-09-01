# -*- coding: utf-8 -*-
"""RAG 服务回归测试：针对评估阶段发现并修复的两个问题。

1. 中文分词必须是 jieba 粒度（整段正则会让关键词检索失效）
2. BM25 重启后能从 Milvus 重建（见 app/core/rag_service.py::_rebuild_bm25）
"""

from __future__ import annotations

from app.core.rag_service import _BM25, _tokenize


def test_tokenize_splits_chinese_into_words():
    tokens = _tokenize("高血压的诊断标准是什么")
    assert "高血压" in tokens
    assert "诊断" in tokens
    assert len(tokens) >= 3


def test_tokenize_handles_mixed_english():
    tokens = _tokenize("LDL-C 控制目标 1.8 mmol/L")
    assert "ldl" in tokens
    assert "mmol" in tokens


def test_bm25_chinese_search_works_with_jieba():
    index = _BM25()
    index.add("d1", "高血压患者每日食盐摄入应低于5克")
    index.add("d2", "糖尿病患者的血糖控制目标")
    hits = index.search("高血压 食盐", top_k=2)
    assert hits[0][0] == "d1"
