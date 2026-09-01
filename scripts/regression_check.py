# -*- coding: utf-8 -*-
"""CI 回归门槛：hybrid 的 Recall@5 / MRR 不得跌破上次基线（bootstrap CI 下限）。

背景（doc 7.2 回归红线）：
  每次改检索相关代码（chunker / embedding / reranker / 融合），一条命令判断"召回有没有回退"。
  红线用「上次基线的 bootstrap 95% CI 下限」——当前均值跌破该下限即报警（exit 1），
  在 CI 内浮动属正常波动，不报警。

两种模式：
  * 默认（真实门禁）：走 Milvus + bge-m3 生产检索，对 eval_queries_dev.json 跑评测，
    与 baseline.json 对比。需要本地 Milvus 数据 + bge-m3 权重（离线可加载）。
  * --standalone（CI smoke 门禁）：自包含 NGramEmbedding + 内存 BM25/RRF，无
    Milvus / sentence-transformers / LLM，只验证「检索算法 + 评测指标 + 回归对比」
    这条链路没被改坏，而非测真实 bge-m3 检索质量。与 baseline_standalone.json 对比。

使用：
  python scripts/regression_check.py --update-baseline          # 跑真实评测，写入 baseline.json
  python scripts/regression_check.py                            # 与 baseline 对比（CI 可接入）
  python scripts/regression_check.py --standalone               # CI smoke 门禁（自包含）
  python scripts/regression_check.py --standalone --update-baseline

退出码：0 = 通过（无回退）；1 = 回归（跌破红线）；2 = 基线缺失。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from eval_rag import RAGEvaluator  # noqa: E402

BASELINE_FILE = SCRIPTS_DIR / "baseline.json"
BASELINE_STANDALONE_FILE = SCRIPTS_DIR / "baseline_standalone.json"
DEFAULT_QUERIES = SCRIPTS_DIR / "eval_queries_dev.json"

# 回归门槛指标：hybrid 模式下的指标 → baseline 里的 CI 下限键
GATE_METRICS = [
    ("recall5", "recall@5"),
    ("mrr", "MRR"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extract_hybrid(result: dict[str, Any]) -> dict[str, Any]:
    """从 evaluate() 结果里取 hybrid 模式的回归门槛指标。"""
    s = result["summary"]
    if "hybrid" not in s:
        raise RuntimeError("评测结果缺少 hybrid 模式（可能知识库为空）")
    h = s["hybrid"]
    return {
        "recall5": h["recall"].get("5", 0.0),
        "recall5_ci_lo": h["recall5_ci"]["lo"],
        "mrr": h["avg_mrr"],
        "mrr_ci_lo": h["mrr_ci"]["lo"],
    }


async def run_eval(queries_path: Path) -> dict[str, Any]:
    evaluator = RAGEvaluator(include_rerank=False)  # 红线只测 hybrid，不跑 rerank，更快
    queries = evaluator.load_queries(queries_path)
    result = await evaluator.evaluate(queries, top_k=20, candidate_k=50)
    return result


def _run_standalone_eval() -> tuple[dict[str, Any], int]:
    """自包含评测（无 Milvus/sentence-transformers/LLM）。

    用 eval_rag_standalone.py 的 NGramEmbedding + 内存 BM25/RRF 对内置测试文档/查询
    跑 hybrid 检索，算 Recall@5 / MRR 的均值与 bootstrap 95% CI 下限，供 CI smoke 对比。
    """
    from eval_metrics import bootstrap_ci
    from eval_rag_standalone import (
        NGramEmbedding,
        InMemoryDocStore,
        SimpleChunker,
        TEST_DOCUMENTS,
        TEST_QUERIES,
        auto_annotate,
        compute_metrics,
    )

    embedder = NGramEmbedding(dim=256)
    chunker = SimpleChunker(chunk_size=300, chunk_overlap=50)
    store = InMemoryDocStore()

    doc_chunks: dict[str, str] = {}
    cid = 0
    for doc_text in TEST_DOCUMENTS:
        for chunk_text in chunker.chunk(doc_text):
            chunk_id = f"chunk_{cid}"
            store.add(chunk_id, chunk_text, embedder.encode(chunk_text))
            doc_chunks[chunk_id] = chunk_text
            cid += 1

    annotated: list[dict[str, Any]] = []
    for tq in TEST_QUERIES:
        relevant = {
            c for c, content in doc_chunks.items()
            if auto_annotate(tq["query"], tq["keywords"], c, content, threshold=2)
        }
        annotated.append({"query": tq["query"], "relevant_ids": relevant})

    recall5s: list[float] = []
    mrrs: list[float] = []
    for qa in annotated:
        qvec = embedder.encode(qa["query"])
        hits = store.hybrid_search(qvec, qa["query"], top_k=10)
        m = compute_metrics([r.id for r in hits], qa["relevant_ids"], [1, 3, 5, 10])
        recall5s.append(m["recall"].get(5, 0.0))
        mrrs.append(m["mrr"])

    n = len(annotated)
    r5_lo, _, _ = bootstrap_ci(recall5s)
    mrr_lo, _, _ = bootstrap_ci(mrrs)
    metrics = {
        "recall5": round(sum(recall5s) / n, 4),
        "recall5_ci_lo": round(r5_lo, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "mrr_ci_lo": round(mrr_lo, 4),
    }
    return metrics, n


def _write_baseline(
    baseline_file: Path, queries_desc: str, metrics: dict[str, Any], num_queries: int
) -> None:
    baseline = {
        "queries_file": queries_desc,
        "num_queries": num_queries,
        "updated_at": _now_iso(),
        "hybrid": metrics,
        "_说明": [
            "由 scripts/regression_check.py --update-baseline 生成。",
            "红线 = recall5_ci_lo / mrr_ci_lo（上次 bootstrap 95% CI 下限）。",
            "当前均值跌破任一红线即判回归（exit 1）。",
        ],
    }
    baseline_file.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _check(baseline: dict[str, Any], metrics: dict[str, Any], num_queries: int) -> bool:
    """对比当前指标与基线红线，返回是否通过。"""
    base_hyb = baseline.get("hybrid", {})
    base_n = baseline.get("num_queries")
    print("=" * 60)
    print("CI 回归门槛检查：hybrid Recall@5 / MRR")
    print("=" * 60)

    if base_n is not None and base_n != num_queries:
        print(
            f"[警告] 查询数变化：基线 {base_n} vs 当前 {num_queries}，"
            "阈值可能不可比（建议 --update-baseline）"
        )

    passed = True
    for key, label in GATE_METRICS:
        cur = metrics[key]
        floor = base_hyb.get(f"{key}_ci_lo", 0.0)
        base_mean = base_hyb.get(key, 0.0)
        ok = cur >= floor
        passed = passed and ok
        mark = "PASS" if ok else "FAIL"
        print(
            f"  {mark}  {label:<10} 当前={cur:.4f}  基线均值={base_mean:.4f}  "
            f"红线(CI下限)={floor:.4f}"
        )
    return passed


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索回归门槛")
    parser.add_argument("--update-baseline", action="store_true",
                        help="跑评测并写入 baseline.json（不判回归）")
    parser.add_argument("--standalone", action="store_true",
                        help="自包含 smoke 门禁（无 Milvus/sentence-transformers/LLM，CI 用）")
    parser.add_argument("--queries", type=str, default=str(DEFAULT_QUERIES),
                        help="测试查询 JSON 文件路径（仅默认真实门禁用）")
    args = parser.parse_args()

    if args.standalone:
        print("模式: standalone（自包含 smoke 门禁）")
        metrics, num_queries = _run_standalone_eval()
        baseline_file = BASELINE_STANDALONE_FILE
    else:
        queries_path = Path(args.queries)
        print(f"模式: 真实门禁 | 查询文件: {queries_path}")
        result = await run_eval(queries_path)
        metrics = _extract_hybrid(result)
        num_queries = result["config"]["num_queries"]
        baseline_file = BASELINE_FILE

    if args.update_baseline:
        queries_desc = (
            "standalone（内置 TEST_DOCUMENTS/TEST_QUERIES）"
            if args.standalone
            else str(args.queries)
        )
        _write_baseline(baseline_file, queries_desc, metrics, num_queries)
        print(f"\n已写入基线: {baseline_file.name}")
        print(f"  hybrid Recall@5 = {metrics['recall5']:.4f} "
              f"(CI 下限 {metrics['recall5_ci_lo']:.4f})")
        print(f"  hybrid MRR      = {metrics['mrr']:.4f} "
              f"(CI 下限 {metrics['mrr_ci_lo']:.4f})")
        return

    if not baseline_file.exists():
        print(f"[错误] 基线不存在: {baseline_file.name}")
        print("       请先运行 --update-baseline 生成基线。")
        sys.exit(2)

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    passed = _check(baseline, metrics, num_queries)

    print("-" * 60)
    print("结论:", "✅ 通过（无回退）" if passed else "❌ 回归（跌破红线）")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
