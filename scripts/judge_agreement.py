# -*- coding: utf-8 -*-
"""
LLM-as-judge 一致性校验（judge vs 独立标注 的 Cohen's Kappa）

背景：生成层评测的数字（faithfulness 0.53 / answer relevancy 0.83 等）来自 LLM-as-judge，
缺一道「judge 打分是否可信」的锚。本脚本按类型分层抽 20%（20 条），对每条：
  1. 走生产路径 rag.chat() 拿到完整答案 + 检索上下文 + [n] 引用
  2. 复用 eval_generation.py 的 4 个 judge 函数打分
  3. 把全文（answer 全文 + contexts 全文 + 四项 judge 分）落成 JSON 供独立标注
随后读独立标注（judge_gold.json，二值 faithfulness 金标），算：
  原始一致率、Cohen's Kappa、混淆矩阵（faithfulness 以 >=0.5 二值化）。

两步用法（Milvus 单写者锁，串行跑）：
  python scripts/judge_agreement.py --sample            # 生成 judge_sample.json
  （人工/二次标注 judge_sample.json → 写 judge_gold.json）
  python scripts/judge_agreement.py --agreement         # 算 Kappa

金标 judge_gold.json 结构：
  {"labels": [{"query": "...", "faithful": true, "note": "..."}]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings  # noqa: E402
from app.core.rag_service import RAGService, _get_embedding_model  # noqa: E402
from eval_generation import (  # noqa: E402
    _build_router,
    judge_answer_relevancy,
    judge_citations,
    judge_context_relevancy,
    judge_faithfulness,
    load_queries,
)

QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries_full.json"
SAMPLE_FILE = PROJECT_ROOT / "scripts" / "judge_sample.json"
GOLD_FILE = PROJECT_ROOT / "scripts" / "judge_gold.json"

# 按类型分层抽 20%（20/99 ≈ 20%）
TARGET_COUNTS = {"事实型": 5, "多跳": 5, "边界": 4, "对比型": 3, "反例": 2, "反馈": 1}
SEED = 42
FAITH_BIN_THRESHOLD = 0.5  # faithfulness >= 0.5 视为「忠实」


def sample_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按类型分层随机抽样（固定 seed，可复现）。"""
    rng = random.Random(SEED)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in queries:
        by_type.setdefault(q.get("type") or "未分类", []).append(q)

    picked: list[dict[str, Any]] = []
    for tname, want in TARGET_COUNTS.items():
        pool = by_type.get(tname, [])
        n = min(want, len(pool))
        picked.extend(rng.sample(pool, n))
        if n < want:
            print(f"  [警告] 类型 {tname} 只有 {len(pool)} 条，抽 {n} 条")
    return picked


async def run_sample(
    queries: list[dict[str, Any]],
    rag: RAGService,
    router: Any,
    embed_model: Any,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(queries, 1):
        query = item["query"]
        qtype = item.get("type", "")
        try:
            resp = await rag.chat(query, chat_history=None, mode=mode, top_k=top_k)
            answer = resp.answer or ""
            contexts = resp.raw_contexts or []
            cite_indices = [c.index for c in (resp.citations or [])]
        except Exception as e:
            print(f"  [{i}/{len(queries)}] 生成失败 {query[:30]}: {e}", flush=True)
            rows.append({"query": query, "type": qtype, "error": str(e)})
            continue

        faith = await judge_faithfulness(router, answer, contexts)
        ctx_rel = await judge_context_relevancy(router, query, contexts)
        ans_rel = await judge_answer_relevancy(router, query, answer, embed_model)
        cit_acc = await judge_citations(router, answer, contexts, cite_indices)

        rows.append({
            "query": query,
            "type": qtype,
            "answer": answer,
            "contexts": [
                {"id": c.id, "content": c.content, "score": round(c.score, 6)}
                for c in contexts
            ],
            "cite_indices": cite_indices,
            "faithfulness": faith,
            "answer_relevancy": ans_rel,
            "context_relevancy": ctx_rel,
            "citation_accuracy": cit_acc,
        })
        print(f"  [{i}/{len(queries)}] {qtype} faith={faith}  {query[:40]}", flush=True)
    return rows


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float, dict[str, int]]:
    """二值 Cohen's Kappa。返回 (po, kappa, 混淆矩阵)。"""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, {}
    n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    po = (n00 + n11) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0
    return po, kappa, {"judge=0/gold=0": n00, "judge=0/gold=1": n01,
                       "judge=1/gold=0": n10, "judge=1/gold=1": n11}


def compute_agreement(sample: list[dict[str, Any]], gold: list[dict[str, Any]]) -> None:
    gold_map = {g["query"].strip(): g.get("faithful") for g in gold}
    judge_bin: list[int] = []
    gold_bin: list[int] = []
    missing = 0
    for r in sample:
        if r.get("error") or r.get("query") is None:
            continue
        g = gold_map.get(r["query"].strip())
        if g is None:
            missing += 1
            continue
        f = r.get("faithfulness")
        judge_bin.append(1 if (f is not None and f >= FAITH_BIN_THRESHOLD) else 0)
        gold_bin.append(1 if g else 0)

    po, kappa, cm = cohen_kappa(judge_bin, gold_bin)
    print("\n===== judge vs 独立标注 一致性 =====")
    print(f"有效样本: {len(judge_bin)}  (缺金标 {missing} 条)")
    print(f"原始一致率 (accuracy): {po:.4f}")
    print(f"Cohen's Kappa:          {kappa:.4f}")
    print(f"混淆矩阵 (judge 行 × gold 列，1=忠实 0=不忠实): {cm}")
    if kappa < 0.0:
        verdict = "无一致性（比随机还差）"
    elif kappa < 0.2:
        verdict = "极低一致性"
    elif kappa < 0.4:
        verdict = "弱一致性"
    elif kappa < 0.6:
        verdict = "中等一致性"
    elif kappa < 0.8:
        verdict = "较强一致性"
    else:
        verdict = "几乎完全一致"
    print(f"解读: {verdict}（Kappa >= 0.6 视为 judge 可信）")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge 一致性校验")
    parser.add_argument("--sample", action="store_true", help="抽样+打分，落 judge_sample.json")
    parser.add_argument("--agreement", action="store_true", help="读 sample+gold，算 Kappa")
    parser.add_argument("--queries", type=str, default=str(QUERIES_FILE))
    parser.add_argument("--mode", type=str, default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.sample and not args.agreement:
        parser.error("请指定 --sample 或 --agreement")

    queries = load_queries(Path(args.queries))

    if args.sample:
        settings = get_settings()
        rag = RAGService()
        if rag.chunks_count == 0:
            print("[错误] 知识库为空。")
            sys.exit(1)
        picked = sample_queries(queries)
        print(f"分层抽样: {len(picked)} 条  |  mode={args.mode}  top_k={args.top_k}")
        router = _build_router(settings)
        embed_model = _get_embedding_model()
        rows = asyncio.run(run_sample(
            picked, rag, router, embed_model, args.mode, args.top_k
        ))
        out = {
            "meta": {
                "sampled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_queries_in_file": len(queries),
                "sampled": len(picked),
                "mode": args.mode,
                "top_k": args.top_k,
                "seed": SEED,
                "faith_bin_threshold": FAITH_BIN_THRESHOLD,
            },
            "rows": rows,
        }
        SAMPLE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {SAMPLE_FILE.name}，请据此完成独立标注 → {GOLD_FILE.name}，再跑 --agreement")

    if args.agreement:
        if not SAMPLE_FILE.exists():
            print(f"[错误] 缺少 {SAMPLE_FILE.name}，先跑 --sample")
            sys.exit(1)
        if not GOLD_FILE.exists():
            print(f"[错误] 缺少 {GOLD_FILE.name}，请先独立标注")
            sys.exit(1)
        sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["rows"]
        gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["labels"]
        compute_agreement(sample, gold)


if __name__ == "__main__":
    main()
