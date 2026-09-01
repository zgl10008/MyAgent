# -*- coding: utf-8 -*-
"""
用 LLM-as-judge 做「分级相关度」标注（0/1/2），替代二元标注。

背景：
  二元相关（相关/不相关）只能算 Recall/Precision/MRR；分级相关（0/1/2）才能
  让 NDCG 体现排序质量——把「高度相关」的 chunk 排前面比「部分相关」更该加分。
  本脚本在 reannotate_queries.py 的候选池之上，让 LLM 对池内每个块打 0/1/2 分。

输出：scripts/eval_queries.graded.json
  - 每个 query 增加 "relevance": {"chunk_id": 2}（2=高度相关，1=部分相关）
  - "relevant_ids" 仍保留 = grade > 0 的 id 集合（供 Recall/Precision/MRR 用）
  - 沿用 v2 的 type 标签

使用：
  python scripts/reannotate_graded.py
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

from reannotate_queries import (  # noqa: E402
    CHUNKS_FILE,
    V2_QUERIES_FILE,
    InMemoryRetriever,
    _build_router,
    _extract_json,
    load_chunks,
    load_queries,
)

OUTPUT_FILE = SCRIPTS_DIR / "eval_queries.graded.json"
CONTENT_TRUNC = 400          # 送入 LLM 的每个 chunk 截断字符数
MAX_TOKENS = 6000            # 推理模型思维链 + JSON
MAX_CONCURRENCY = 4

JUDGE_SYSTEM = (
    "你是医学信息相关性分级标注员。对每个候选文档块，判断它能否回答用户问题，"
    "给出分级相关度：\n"
    "  2 = 高度相关：直接回答问题的核心事实/数据/定义/结论\n"
    "  1 = 部分相关：提供背景或部分信息，但不完整、不能单独回答\n"
    "  0 = 不相关：与问题无关或提供不了实质信息\n"
    "只输出一个 JSON：{\"grades\": {\"0\": 2, \"3\": 1}}\n"
    "键为候选块编号（字符串），值为 0/1/2；不相关的块可省略（默认 0）。"
)


def _build_prompt(query: str, pool: dict[str, str]) -> tuple[str, list[str]]:
    ids = list(pool.keys())
    lines = [f"[{i}] {pool[cid][:CONTENT_TRUNC]}" for i, cid in enumerate(ids)]
    user = (
        f"用户问题：{query}\n\n"
        f"候选文档块：\n" + "\n".join(lines) + "\n\n"
        "请对每个块给出 0/1/2 分级相关度，输出 JSON。"
    )
    return user, ids


async def graded_judge(
    router: Any, query: str, pool: dict[str, str]
) -> dict[str, int]:
    """返回 {chunk_id: grade}（grade ∈ {1, 2}，0 的剔除）。"""
    if not pool:
        return {}
    user, ids = _build_prompt(query, pool)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(messages, temperature=0.0, max_tokens=MAX_TOKENS)
            data = _extract_json(resp.content)
            if data is None:
                raise ValueError(f"无法解析 JSON: {resp.content[:120]!r}")
            raw = data.get("grades", {})
            grades: dict[str, int] = {}
            for key, g in raw.items():
                idx = int(key)
                grade = int(g)
                if 0 <= idx < len(ids) and grade in (1, 2):
                    grades[ids[idx]] = grade
            return grades
        except Exception as e:
            print(f"    [重试 {attempt + 1}] 分级判断失败: {e}")
            await asyncio.sleep(1.0)
    return {}


async def main() -> None:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="分级标注（LLM-as-judge 0/1/2）")
    parser.add_argument("--queries", type=str, default=str(V2_QUERIES_FILE),
                        help="输入查询 JSON 文件路径（默认 eval_queries.v2.json）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help="输出分级标注 JSON 文件路径")
    parser.add_argument("--limit", type=int, default=0, help="只标注前 N 条（0=全部）")
    args = parser.parse_args()

    settings = get_settings()
    print("加载 doc_chunks_full.json ...")
    chunks = load_chunks(CHUNKS_FILE)
    if not chunks:
        print(f"[错误] 无有效 chunk: {CHUNKS_FILE}")
        sys.exit(1)

    print("构建内存检索（bge-m3 向量 + BM25）...")
    retriever = InMemoryRetriever(chunks)
    print(f"知识库块数: {retriever.count}")

    router = _build_router(settings)
    items, meta = load_queries(Path(args.queries))
    if args.limit > 0:
        items = items[: args.limit]
    print(f"测试查询数: {len(items)}")
    print("-" * 60)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def grade_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        async with sem:
            pool = retriever.candidate_pool(query)
            grades = await graded_judge(router, query, pool)

        new_item = dict(item)
        new_item["relevance"] = grades
        new_item["relevant_ids"] = sorted(grades.keys())
        print(
            f"[{idx}/{len(items)}] {query[:30]}  候选={len(pool)}  相关(≥1)={len(grades)}",
            flush=True,
        )
        return new_item

    new_items = []
    for i, item in enumerate(items, 1):
        new_items.append(await grade_one(i, item))

    avg_rel = sum(len(it["relevant_ids"]) for it in new_items) / len(new_items)
    avg_high = sum(
        1 for it in new_items for g in it["relevance"].values() if g == 2
    ) / len(new_items)
    print("\n" + "=" * 60)
    print(f"分级标注完成：平均相关块 {avg_rel:.2f}，其中高度相关(2) 平均 {avg_high:.2f}")

    out = dict(meta)
    out["_说明"] = [
        "由 scripts/reannotate_graded.py 用 LLM-as-judge 分级标注生成（0/1/2）。",
        "relevance = {chunk_id: grade}，grade 1=部分相关 2=高度相关；relevant_ids = grade>0 集合。",
        "分级只影响 NDCG 增益权重；Recall/Precision/MRR 只关心相关与否。",
    ]
    out["queries"] = new_items
    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写出: {out_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
