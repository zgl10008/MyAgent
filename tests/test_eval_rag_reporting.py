"""检索评测的口径回归：库外样本不能拉低可回答问题的 Recall。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_eval_rag():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_rag.py"
    spec = importlib.util.spec_from_file_location("eval_rag_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_answerable_summary_excludes_out_of_scope_queries():
    module = _load_eval_rag()
    queries = [
        module.TestQuery("可回答", {"core", "extra"}, rel_grades={"core": 2, "extra": 1}),
        module.TestQuery("库外", set()),
    ]
    metrics = {
        "hybrid": [
            module.EvalMetrics("可回答", "hybrid", recall={5: 0.5}, mrr=1.0, ndcg={5: 0.8}),
            module.EvalMetrics("库外", "hybrid", recall={5: 0.0}, mrr=0.0, ndcg={5: 0.0}),
        ]
    }
    rows = [
        {"modes": {"hybrid": {"evidence_hit5": True}}},
        {"modes": {"hybrid": {"evidence_hit5": False}}},
    ]

    result = module.RAGEvaluator._build_answerable_summary(
        object.__new__(module.RAGEvaluator), metrics, rows, queries, ["hybrid"]
    )

    assert result["answerable_count"] == 1
    assert result["out_of_scope_count"] == 1
    assert result["recall5_macro_ceiling"] == 1.0
    assert result["modes"]["hybrid"]["recall5"] == 0.5
    assert result["modes"]["hybrid"]["evidence_hit5"] == 1.0
