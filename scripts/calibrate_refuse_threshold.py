# -*- coding: utf-8 -*-
"""
库外拒答门控阈值校准

背景：生成层评测发现 12 条「反例」（库外查询）全部被硬答、faithfulness 仅 ~0.34，
系统对知识库外的问题不拒答、伴随幻觉。本脚本在生产检索路径上收集每条 query 的
top-1 精排相关度（cross-encoder 分，[0,1] 越大越相关），按「领域内 / 反例」两组
扫阈值网格，找出能最大化分离度的拒答阈值，供 config.rag_refuse_threshold 使用。

口径：
  - 分数取自 RAGService.search(query, top_k=5, mode="hybrid", rerank=True) 的
    hits[0].score（rerank 开启时是 cross-encoder 分；与 chat() 里拒答门控同一口径）。
  - 反例（type=="反例"）= 应拒答（正类）；其余类型 = 领域内应回答（负类）。
  - 判定：score < threshold → 拒答。据此算反例拒答率（recall）、领域内误拒率、
    F1，并打印推荐阈值。

注意：RAGService 走 Milvus Lite（milvus.db 单写者锁），本脚本必须与其它
Milvus 脚本串行运行，不能并发。

使用：
  python scripts/calibrate_refuse_threshold.py --queries scripts/eval_queries_full.json     # 扫阈值网格
  python scripts/calibrate_refuse_threshold.py --verify-refuse 0.40                        # 端到端验证反例全拒答
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rag_service import RAGService  # noqa: E402

QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries_full.json"


def load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["queries"] if isinstance(data, dict) and "queries" in data else data)


async def verify_refuse(
    queries: list[dict[str, Any]], rag: RAGService, threshold: float
) -> None:
    """端到端验证拒答门控：阈值下 chat() 对反例是否全部返回拒答（不触发 LLM）。"""
    rag._rag_refuse_threshold = threshold
    fanli = [q for q in queries if q.get("type") == "反例"]
    refused = 0
    print(f"验证拒答门控 threshold={threshold}（反例 {len(fanli)} 条）")
    for i, q in enumerate(fanli, 1):
        resp = await rag.chat(q["query"], chat_history=None, mode="hybrid", top_k=5)
        is_refused = (resp.answer or "").startswith("抱歉")
        refused += 1 if is_refused else 0
        tag = "拒答" if is_refused else "硬答✗"
        print(f"  [{i}/{len(fanli)}] {tag}  {q['query'][:44]}", flush=True)
    print(f"\n反例拒答 {refused}/{len(fanli)}（{refused / len(fanli) * 100:.0f}%）")


async def collect_scores(
    queries: list[dict[str, Any]], rag: RAGService
) -> list[tuple[str, str, float | None]]:
    """逐条检索，返回 [(query, type, top1_score)]。"""
    rows: list[tuple[str, str, float | None]] = []
    for i, q in enumerate(queries, 1):
        qtype = q.get("type", "")
        try:
            hits = await rag.search(q["query"], top_k=5, mode="hybrid", rerank=True)
            top1 = hits[0].score if hits else None
        except Exception as e:
            print(f"  [{i}/{len(queries)}] 检索失败 {q['query'][:30]}: {e}", flush=True)
            top1 = None
        rows.append((q["query"], qtype, top1))
        print(f"  [{i}/{len(queries)}] {qtype} score={top1}  {q['query'][:40]}", flush=True)
    return rows


def evaluate_threshold(rows: list[tuple[str, str, float | None]], thr: float) -> dict[str, Any]:
    """给定阈值，返回二分类混淆指标。正类=反例（应拒答）。"""
    tp = fp = tn = fn = 0
    for _, qtype, s in rows:
        refuse = (s is None) or (s < thr)  # 无命中视为拒答
        if qtype == "反例":
            if refuse:
                tp += 1
            else:
                fn += 1
        else:
            if refuse:
                fp += 1
            else:
                tn += 1
    pos = tp + fn
    neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "反例拒答率(recall)": round(recall, 4),
        "领域内误拒率(FPR)": round(fp / neg, 4) if neg else 0.0,
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="库外拒答门控阈值校准")
    parser.add_argument("--queries", type=str, default=str(QUERIES_FILE))
    parser.add_argument(
        "--verify-refuse", type=float, default=None,
        help="端到端验证：用该阈值在 chat() 上确认反例全部拒答（仅反例、不触发 LLM）",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    queries = load_queries(queries_path)
    print(f"查询数: {len(queries)}")

    rag = RAGService()
    if rag.chunks_count == 0:
        print("[错误] 知识库为空，请先入库。")
        sys.exit(1)
    print(f"知识库块数: {rag.chunks_count}")

    if args.verify_refuse is not None:
        asyncio.run(verify_refuse(queries, rag, args.verify_refuse))
        sys.exit(0)

    rows = asyncio.run(collect_scores(queries, rag))

    # 分组分数分布
    fanli = [s for _, t, s in rows if t == "反例" and s is not None]
    indom = [s for _, t, s in rows if t != "反例" and s is not None]
    print("\n===== top-1 精排分分布 =====")
    if fanli:
        print(f"反例({len(fanli)}):   min={min(fanli):.4f}  max={max(fanli):.4f}  "
              f"mean={sum(fanli)/len(fanli):.4f}")
    if indom:
        print(f"领域内({len(indom)}): min={min(indom):.4f}  max={max(indom):.4f}  "
              f"mean={sum(indom)/len(indom):.4f}")

    # 无命中（score=None）的条数
    none_fanli = sum(1 for _, t, s in rows if t == "反例" and s is None)
    none_indom = sum(1 for _, t, s in rows if t != "反例" and s is None)
    print(f"无命中(score=None): 反例 {none_fanli} / 领域内 {none_indom}")

    # 扫阈值网格
    all_scores = sorted({s for _, _, s in rows if s is not None})
    if not all_scores:
        print("[错误] 没有任何检索结果，无法校准。")
        sys.exit(1)
    grid = [round(min(all_scores) + i * (max(all_scores) - min(all_scores)) / 99, 4)
            for i in range(100)]
    results = [evaluate_threshold(rows, t) for t in grid]
    best = max(results, key=lambda r: r["f1"])

    print("\n===== 阈值扫描（前若干 + 最佳）=====")
    print(f"{'阈值':>8} {'反例拒答率':>10} {'领域内误拒率':>12} {'precision':>10} {'F1':>8}")
    # 打印有代表性的若干点：最佳附近 + 端点
    shown = sorted(
        set([best["threshold"]]
            + [results[0]["threshold"], results[-1]["threshold"]]
            + [r["threshold"] for r in results if abs(r["f1"] - best["f1"]) < 1e-6]),
        key=lambda t: t,
    )
    # 额外补几个等距点便于观察
    for idx in range(0, len(results), 20):
        shown.append(results[idx]["threshold"])
    for r in sorted([x for x in results if x["threshold"] in shown], key=lambda x: x["threshold"]):
        print(f"{r['threshold']:>8.4f} {r['反例拒答率(recall)']:>10.4f} "
              f"{r['领域内误拒率(FPR)']:>12.4f} {r['precision']:>10.4f} {r['f1']:>8.4f}")

    print("\n===== 推荐阈值 =====")
    print(f"RAG_REFUSE_THRESHOLD = {best['threshold']}")
    print(f"  反例拒答率 {best['反例拒答率(recall)']*100:.1f}%  领域内误拒率 "
          f"{best['领域内误拒率(FPR)']*100:.1f}%  F1={best['f1']}")
    print(f"  混淆矩阵: TP(反例拒答)={best['tp']} FP(领域内误拒)={best['fp']} "
          f"TN(领域内答)={best['tn']} FN(反例漏拒)={best['fn']}")


if __name__ == "__main__":
    main()
