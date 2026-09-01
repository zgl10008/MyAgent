# -*- coding: utf-8 -*-
"""
Cross-Encoder 精排 A/B 对比评测：rerank 开 vs 关。

对比对象：混合检索（BM25 + 向量 + RRF）在「开启 / 关闭」cross-encoder 精排两种
配置下，对同一批测试查询的检索质量差异。

使用方式：
  python scripts/eval_rerank.py

依赖：
  - 知识库已入库（Milvus 中有 chunk）
  - scripts/eval_queries.json 存在（含人工/自动标注的 relevant_ids）
  - cross-encoder 模型已缓存（首次运行会走 HF 镜像下载）

输出：scripts/eval_report_rerank.md + 控制台对比表
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from app.config import get_settings  # noqa: E402
from app.core.rag.reranker import Reranker  # noqa: E402
from app.core.rag_service import RAGService  # noqa: E402
import eval_rag  # noqa: E402  复用 compute_all_metrics / TestQuery / load_queries

EVAL_K_VALUES = [1, 3, 5, 10]
TOP_K = 20  # 最终返回/报告的检索结果数
CANDIDATE_K = 50  # 精排前召回的候选集大小（需 > TOP_K，才能体现精排的价值）
QUERIES_FILE = SCRIPTS_DIR / "eval_queries.json"
REPORT_FILE = SCRIPTS_DIR / "eval_report_rerank.md"


def enable_rerank(rag: RAGService) -> None:
    """在已构造的 RAGService 实例上原地开启精排（复用同一套 embedding + 向量库）。

    关键：candidate_k 必须大于 top_k。两阶段检索的价值在于「廉价方法多召回 →
    cross-encoder 精排取 top」，若 candidate_k == top_k，精排只能重排、无法提升召回。
    """
    settings = get_settings()
    rag._rerank_enabled = True
    rag._rerank_failed = False
    rag._rerank_candidate_k = max(CANDIDATE_K, int(settings.rerank_candidate_k))
    rag._reranker = Reranker(
        model_name=settings.rerank_model,
        device=settings.rerank_device or None,
    )


async def eval_hybrid(rag: RAGService, queries: list[Any], top_k: int):
    """对每条查询跑 hybrid 检索，返回 (逐查询指标, 耗时列表 ms)。"""
    metrics: list[Any] = []
    latencies: list[float] = []
    for tq in queries:
        t0 = time.perf_counter()
        hits = await rag.search(tq.query, top_k=top_k, mode="hybrid")
        latencies.append((time.perf_counter() - t0) * 1000)
        m = eval_rag.compute_all_metrics(
            [r.id for r in hits], tq.relevant_ids, EVAL_K_VALUES
        )
        m.query = tq.query
        m.mode = "hybrid"
        metrics.append(m)
    return metrics, latencies


def summarize(metrics: list[Any], latencies: list[float]) -> dict[str, Any]:
    """汇总平均指标 + 平均耗时。"""
    n = len(metrics)
    s: dict[str, Any] = {
        "count": n,
        "avg_mrr": round(sum(m.mrr for m in metrics) / n, 4),
        "avg_latency_ms": round(sum(latencies) / n, 1),
    }
    for metric in ("recall", "precision", "ndcg"):
        s[metric] = {}
        for k in EVAL_K_VALUES:
            s[metric][str(k)] = round(
                sum(getattr(m, metric).get(k, 0.0) for m in metrics) / n, 4
            )
    return s


def build_markdown(
    base: dict[str, Any],
    rerank: dict[str, Any],
    config: dict[str, Any],
    per_query: list[dict[str, Any]],
) -> str:
    """生成对比报告。"""
    lines: list[str] = []
    lines.append("# Cross-Encoder 精排 A/B 对比评估报告\n")
    lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**知识库块数**: {config['total_chunks']}")
    lines.append(f"**测试查询数**: {config['num_queries']}")
    lines.append(f"**检索模式**: hybrid（BM25 + 向量 + RRF），top_k={TOP_K}")
    lines.append(f"**精排模型**: {config['rerank_model']}，候选集 candidate_k={config['candidate_k']}\n")

    lines.append("## 一、汇总对比（hybrid 基线 vs hybrid+精排）\n")
    for metric in ("recall", "precision", "ndcg"):
        title = {"recall": "Recall@K（召回率）", "precision": "Precision@K（精确率）",
                 "ndcg": "NDCG@K（排序质量）"}[metric]
        lines.append(f"### {title} — 越高越好\n")
        lines.append("| 配置 | " + " | ".join(f"K={k}" for k in EVAL_K_VALUES) + " |")
        lines.append("|------|" + "|".join(["------"] * len(EVAL_K_VALUES)) + "|")
        lines.append("| 关闭精排 | " + " | ".join(
            f"{base[metric][str(k)]:.4f}" for k in EVAL_K_VALUES) + " |")
        lines.append("| **开启精排** | " + " | ".join(
            f"{rerank[metric][str(k)]:.4f}" for k in EVAL_K_VALUES) + " |")
        deltas = []
        for k in EVAL_K_VALUES:
            b = base[metric][str(k)]
            r = rerank[metric][str(k)]
            pct = (r - b) / b * 100 if b > 0 else 0.0
            deltas.append(f"{pct:+.1f}%")
        lines.append("| 变化幅度 | " + " | ".join(deltas) + " |")
        lines.append("")

    lines.append("### MRR 与耗时\n")
    lines.append("| 配置 | MRR | 平均耗时(ms) |")
    lines.append("|------|-----|-------------|")
    lines.append(f"| 关闭精排 | {base['avg_mrr']:.4f} | {base['avg_latency_ms']:.1f} |")
    lines.append(f"| **开启精排** | {rerank['avg_mrr']:.4f} | {rerank['avg_latency_ms']:.1f} |")
    mrr_pct = ((rerank["avg_mrr"] - base["avg_mrr"]) / base["avg_mrr"] * 100
               if base["avg_mrr"] > 0 else 0.0)
    lines.append(f"| 变化幅度 | **{mrr_pct:+.1f}%** | +{rerank['avg_latency_ms'] - base['avg_latency_ms']:.1f} ms |")
    lines.append("")

    # 逐查询：列出精排后 Top-5 命中数变化的查询
    lines.append("## 二、逐查询 Top-5 命中数对比（仅列出有变化的查询）\n")
    lines.append("| Query | 关闭精排命中 | 开启精排命中 |")
    lines.append("|-------|-------------|-------------|")
    changed = 0
    for row in per_query:
        if row["base_hits"] != row["rerank_hits"]:
            changed += 1
            lines.append(
                f"| {row['query'][:40]} | {row['base_hits']} | **{row['rerank_hits']}** |"
            )
    if changed == 0:
        lines.append("| （无变化） | - | - |")
    lines.append(f"\n共 {changed}/{config['num_queries']} 条查询 Top-5 命中数发生变化。")

    # 简历结论
    lines.append("\n## 三、简历可引用结论\n")
    lines.append(
        f"- 在两阶段检索中加入 cross-encoder 精排后，MRR 从 {base['avg_mrr']:.4f} "
        f"{'提升' if mrr_pct >= 0 else '变化'}至 {rerank['avg_mrr']:.4f}（{mrr_pct:+.1f}%）"
    )
    p5 = (rerank["precision"]["5"] - base["precision"]["5"]) / base["precision"]["5"] * 100 \
        if base["precision"]["5"] > 0 else 0.0
    lines.append(
        f"- Precision@5 从 {base['precision']['5']:.4f} "
        f"{'提升' if p5 >= 0 else '变化'}至 {rerank['precision']['5']:.4f}（{p5:+.1f}%）"
    )
    lines.append(
        f"- 代价：单查询平均增加 {rerank['avg_latency_ms'] - base['avg_latency_ms']:.0f} ms "
        f"（cross-encoder 逐对打分），实际部署按需开关"
    )

    return "\n".join(lines) + "\n"


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Encoder 精排 A/B 对比评测")
    parser.add_argument("--queries", type=str, default=None,
                        help="测试查询 JSON 文件路径（默认 scripts/eval_queries.json）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告路径（默认 scripts/eval_report_rerank.md）")
    args = parser.parse_args()
    queries_path = Path(args.queries) if args.queries else QUERIES_FILE
    report_path = Path(args.output) if args.output else REPORT_FILE

    print("=" * 64)
    print("Cross-Encoder 精排 A/B 对比评测")
    print("=" * 64)

    rag = RAGService()
    total_chunks = rag.chunks_count
    if total_chunks == 0:
        print("[错误] 知识库为空，请先入库文档。")
        sys.exit(1)

    evaluator = eval_rag.RAGEvaluator.__new__(eval_rag.RAGEvaluator)  # 仅复用 load_queries
    queries = evaluator.load_queries(queries_path)

    settings = get_settings()
    print(f"知识库块数: {total_chunks}")
    print(f"测试查询数: {len(queries)}")
    print(f"精排模型: {settings.rerank_model}")
    print(f"候选集 candidate_k: {max(CANDIDATE_K, int(settings.rerank_candidate_k))}")
    print("-" * 64)

    # 基线：关闭精排
    print("[1/2] 评估基线（关闭精排）...")
    rag._rerank_enabled = False
    base_metrics, base_lat = await eval_hybrid(rag, queries, TOP_K)
    base = summarize(base_metrics, base_lat)

    # 开启精排（先 warm-up 一次，避免首次模型加载计入耗时）
    print("[2/2] 开启精排并评估...")
    enable_rerank(rag)
    warm_query = queries[0].query
    print(f"  预热精排模型（首次调用加载模型）: {warm_query[:30]}...")
    await rag.search(warm_query, top_k=TOP_K, mode="hybrid")
    rerank_metrics, rerank_lat = await eval_hybrid(rag, queries, TOP_K)
    rerank = summarize(rerank_metrics, rerank_lat)

    # 打印控制台对比
    print("\n── 控制台对比（hybrid 基线 vs hybrid+精排）──")
    for metric, label in (("recall", "Recall"), ("precision", "Precision"), ("ndcg", "NDCG")):
        print(f"  {label}: 基线 " + " ".join(f"{base[metric][str(k)]:.3f}" for k in EVAL_K_VALUES)
              + "  →  精排 " + " ".join(f"{rerank[metric][str(k)]:.3f}" for k in EVAL_K_VALUES))
    print(f"  MRR:   基线 {base['avg_mrr']:.4f}  →  精排 {rerank['avg_mrr']:.4f}")
    print(f"  耗时:  基线 {base['avg_latency_ms']:.0f} ms  →  精排 {rerank['avg_latency_ms']:.0f} ms")

    # 重新计算逐查询命中数（干净实现）
    per_query = []
    for tq, bm, rm in zip(queries, base_metrics, rerank_metrics):
        per_query.append({
            "query": tq.query,
            "base_hits": _top5_hits_for_metrics(bm, tq),
            "rerank_hits": _top5_hits_for_metrics(rm, tq),
        })

    report = build_markdown(
        base, rerank,
        {
            "total_chunks": total_chunks,
            "num_queries": len(queries),
            "rerank_model": settings.rerank_model,
            "candidate_k": max(CANDIDATE_K, int(settings.rerank_candidate_k)),
        },
        per_query,
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {report_path}")


def _top5_hits_for_metrics(metrics: Any, tq: Any) -> int:
    """从 EvalMetrics 推导 Top-5 命中数（precision@5 * 5）。"""
    p5 = metrics.precision.get(5, 0.0)
    return round(p5 * 5)


if __name__ == "__main__":
    asyncio.run(main())
