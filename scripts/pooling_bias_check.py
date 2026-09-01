# -*- coding: utf-8 -*-
"""
量化 pooling bias（候选池漏检）+ 标注一致性（Cohen's Kappa）。

背景（doc 3.3 坑①）：
  评测标注的候选池是「现有检索系统捞出来的」，池外可能还有真正相关的文档没进池，
  标注者永远标不到 → 漏检被系统性漏掉 → Recall 虚高。本脚本用「全库通读」做基准，
  直接对比「池内标注」和「全库标注」的相关集合，量化池外漏了多少。

两个子功能：
  1. 默认（--mode pool）：对每条 query
       - 池内相关 = 候选池（三路召回 Top-20 并集）里 LLM 判的相关块
       - 全库相关 = 把 916 块分批复读、LLM 判的相关块（短截断，够判相关性）
       输出池外漏检率 = |全库 - 池内| / |全库|。
  2. --mode kappa：两个独立 judge（标准 / 严格）对同一候选池打 0/1，
       用 Cohen's Kappa 量化标注一致性（<0.6 说明标准不清）。

使用：
  python scripts/pooling_bias_check.py --limit 5
  python scripts/pooling_bias_check.py --mode kappa --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from eval_metrics import cohens_kappa
from reannotate_queries import (  # noqa: E402
    CHUNKS_FILE,
    V2_QUERIES_FILE,
    InMemoryRetriever,
    _build_router,
    _extract_json,
    llm_judge,
    load_chunks,
    load_queries,
)

MAX_TOKENS = 6000
MAX_CONCURRENCY = 4
BATCH_SIZE = 50            # 全库分批读，每批 50 块
CONTENT_TRUNC = 120        # 全库扫描用短截断（够判相关性即可）

JUDGE_FULLSCAN = (
    "你是医学信息相关性标注员。下面是从知识库抽取的文档块（编号从 1 开始）。"
    "判断哪些块能回答用户问题（提供问题所需的事实/数据/定义/建议）。\n"
    "只输出 JSON：{\"relevant\": [2, 5]}（相关块的编号；没有则 []）"
)

JUDGE_STRICT = (
    "你是医学信息相关性标注员（严格口径）。只有「直接回答用户问题核心、能作为最终答案依据」"
    "的文档块才算相关；只泛泛提及主题或提供背景的判为不相关。\n"
    "只输出 JSON：{\"relevant\": [2, 5]}（相关块编号；没有则 []）"
)


async def _judge(router: Any, system: str, user: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(messages, temperature=0.0, max_tokens=MAX_TOKENS)
            data = _extract_json(resp.content)
            if data is None:
                raise ValueError(f"无法解析 JSON: {resp.content[:120]!r}")
            return data
        except Exception as e:
            print(f"    [重试 {attempt + 1}] {e}")
            await asyncio.sleep(1.0)
    return {}


async def fullscan_relevant(
    router: Any, query: str, chunks: list[dict[str, Any]]
) -> set[str]:
    """全库分批复读，返回相关 chunk id 集合。"""
    ids = [c["id"] for c in chunks]
    relevant: set[str] = set()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        lines = [
            f"[{start + i + 1}] {c['content'][:CONTENT_TRUNC]}"
            for i, c in enumerate(batch)
        ]
        user = (
            f"用户问题：{query}\n\n候选文档块：\n" + "\n".join(lines)
            + "\n\n判断哪些块相关，输出 JSON。"
        )
        data = await _judge(router, JUDGE_FULLSCAN, user)
        for idx in data.get("relevant", []):
            if isinstance(idx, int) and start + 1 <= idx <= start + len(batch):
                relevant.add(ids[idx - 1])
    return relevant


async def pool_mode(
    router: Any,
    retriever: InMemoryRetriever,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> None:
    print("=" * 60)
    print("Pooling bias 检查：池内标注 vs 全库通读基准")
    print("=" * 60)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        async with sem:
            pool = retriever.candidate_pool(query)
            pool_rel = await llm_judge(router, query, pool)
        full_rel = await fullscan_relevant(router, query, chunks)
        missed = full_rel - pool_rel
        rate = len(missed) / len(full_rel) if full_rel else 0.0
        print(
            f"[{idx}/{len(queries)}] {query[:28]}  池内={len(pool_rel)} "
            f"全库={len(full_rel)} 池外漏检={len(missed)} 漏检率={rate:.2f}",
            flush=True,
        )
        return {
            "query": query,
            "pool_relevant": sorted(pool_rel),
            "full_relevant": sorted(full_rel),
            "missed": sorted(missed),
            "miss_rate": round(rate, 4),
        }

    rows = [await one(i, q) for i, q in enumerate(queries, 1)]
    avg = sum(r["miss_rate"] for r in rows) / len(rows) if rows else 0.0
    print("\n" + "-" * 60)
    print(f"平均池外漏检率 = {avg:.2%}")
    print("漏检率越高 → pooling bias 越严重，当前 Recall 越可能虚高。")
    print("建议：漏检率 > 20% 的 query 应全库重标；整体偏高则考虑扩候选池（加多路召回/提高 Top-K）。")


async def kappa_mode(
    router: Any,
    retriever: InMemoryRetriever,
    queries: list[dict[str, Any]],
) -> None:
    print("=" * 60)
    print("标注一致性检查：标准 judge vs 严格 judge（Cohen's Kappa）")
    print("=" * 60)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def one(idx: int, item: dict[str, Any]) -> tuple[str, float]:
        query = item["query"]
        async with sem:
            pool = retriever.candidate_pool(query)
            std = await llm_judge(router, query, pool)  # 标准口径（复用 reannotate 的 judge）

        # 严格口径 judge：同一候选池
        ids = list(pool.keys())
        lines = [f"[{i}] {pool[cid][:400]}" for i, cid in enumerate(ids)]
        user = (
            f"用户问题：{query}\n\n候选文档块：\n" + "\n".join(lines)
            + "\n\n判断哪些块相关（严格口径），输出 JSON。"
        )
        data = await _judge(router, JUDGE_STRICT, user)
        strict = {
            ids[i]
            for i in data.get("relevant", [])
            if isinstance(i, int) and 0 <= i < len(ids)
        }

        # 两个标注者对同一批 pool item 打 0/1
        a = {cid: (1 if cid in std else 0) for cid in ids}
        b = {cid: (1 if cid in strict else 0) for cid in ids}
        kappa = cohens_kappa(a, b, categories=[0, 1])
        print(
            f"[{idx}/{len(queries)}] {query[:28]}  标准={len(std)} 严格={len(strict)} "
            f"Kappa={kappa:.3f}",
            flush=True,
        )
        return query, kappa

    rows = [await one(i, q) for i, q in enumerate(queries, 1)]
    avg = sum(k for _, k in rows) / len(rows) if rows else 0.0
    print("\n" + "-" * 60)
    print(f"平均 Cohen's Kappa = {avg:.3f}")
    print("解读：<0.6 一致性不足（标注标准模糊，评测数字不可信）；≥0.6 可用；≥0.8 良好。")


async def main() -> None:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="pooling bias / Kappa 一致性检查")
    parser.add_argument("--mode", choices=["pool", "kappa"], default="pool")
    parser.add_argument("--limit", type=int, default=5, help="只检查前 N 条 query")
    args = parser.parse_args()

    settings = get_settings()
    print("加载 doc_chunks_full.json ...")
    chunks = load_chunks(CHUNKS_FILE)
    print("构建内存检索（bge-m3 向量 + BM25）...")
    retriever = InMemoryRetriever(chunks)
    print(f"知识库块数: {retriever.count}")

    router = _build_router(settings)
    items, _meta = load_queries(V2_QUERIES_FILE)
    items = items[: args.limit]
    print(f"检查查询数: {len(items)}")

    if args.mode == "kappa":
        await kappa_mode(router, retriever, items)
    else:
        await pool_mode(router, retriever, chunks, items)


if __name__ == "__main__":
    asyncio.run(main())
