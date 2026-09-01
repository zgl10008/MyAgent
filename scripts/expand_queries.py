# -*- coding: utf-8 -*-
"""
扩充评测查询集 + 拆 dev/test。

背景（doc 3.1 + 6.4）：
  当前 29 条查询类型失衡（18 事实型 / 10 多跳 / 1 边界 / 0 对比 / 0 反例），
  且一套数据既调参又评估会 overfit。本脚本：
    1) 用 LLM 生成缺的类型（对比型 / 多跳 / 边界 / 反例）的新查询；
    2) 对新查询走与 v2 相同的候选池 + 分级标注，产出带 relevance 的标注；
    3) 合并后拆 dev（调参）/ test（最终评估，只碰一次）。

输出：
  scripts/eval_queries_dev.json   （分级标注集，作为调参集）
  scripts/eval_queries_test.json  （新增查询，作为最终评估集）
  scripts/eval_queries_expanded.json（合并全集，含 type + relevance）

使用：
  python scripts/expand_queries.py --num 12    # 生成并标注 12 条新查询（冒烟）
  python scripts/expand_queries.py             # 目标补齐到 100+（较长）
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

from reannotate_graded import graded_judge  # noqa: E402
from reannotate_queries import (  # noqa: E402
    CHUNKS_FILE,
    InMemoryRetriever,
    _build_router,
    _extract_json,
    load_chunks,
    load_queries,
)

# dev 以「分级标注」为准（P2 的规范 ground truth），不是旧的二元 v2。
GRADED_QUERIES_FILE = SCRIPTS_DIR / "eval_queries.graded.json"

OUT_DEV = SCRIPTS_DIR / "eval_queries_dev.json"
OUT_TEST = SCRIPTS_DIR / "eval_queries_test.json"
OUT_EXPANDED = SCRIPTS_DIR / "eval_queries_expanded.json"
MAX_TOKENS = 6000
MAX_CONCURRENCY = 4

# 缺的类型 → 生成提示（要求：真实问法、可评估、口语化多样）
GEN_PROMPTS = {
    "对比型": (
        "你是心血管慢病测试集设计者。生成「对比型」用户提问：需要跨多段文档对比两种"
        "药物/两种疾病/两个指标才能回答（如「XX 和 YY 的用药差异」「A 与 B 哪个更适合 C」）。"
        "领域限于高血压/糖尿病/冠心病/心衰/血脂。只输出 JSON：{\"queries\": [\"...\"]}"
    ),
    "多跳": (
        "你是心血管慢病测试集设计者。生成「多跳/推理型」用户提问：需要拼接多个条件或"
        "多个 chunk 才能回答（如「合并 X 和 Y 的患者首选什么药」「在 Z 情况下如何调整 A」）。"
        "领域限于高血压/糖尿病/冠心病/心衰/血脂。只输出 JSON：{\"queries\": [\"...\"]}"
    ),
    "边界": (
        "你是心血管慢病测试集设计者。生成「边界/歧义型」用户提问：症状描述含糊、一词多病、"
        "或带极端词（如「胸口疼是怎么回事」「所有高血压都要吃药吗」），考排序质量。"
        "只输出 JSON：{\"queries\": [\"...\"]}"
    ),
    "反例": (
        "你是心血管慢病测试集设计者。生成「反例/库外型」用户提问：问的是心血管知识库"
        "大概率没有的内容（如其他系统疾病「帕金森病如何治疗」、否定句、或口语化闲聊），"
        "用于测试系统能否诚实说「我不知道」而非强行检索。只输出 JSON：{\"queries\": [\"...\"]}"
    ),
}


async def generate_queries(
    router: Any, qtype: str, num: int
) -> list[str]:
    prompt = GEN_PROMPTS[qtype] + f"\n请生成 {num} 条。"
    messages = [
        {"role": "system", "content": "你只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(messages, temperature=0.8, max_tokens=MAX_TOKENS)
            data = _extract_json(resp.content)
            qs = data.get("queries", []) if data else []
            qs = [q for q in qs if isinstance(q, str) and q.strip()]
            if qs:
                return qs[:num]
        except Exception as e:
            print(f"    [生成重试 {attempt + 1}] {e}")
            await asyncio.sleep(1.0)
    return []


async def main() -> None:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="扩充查询集 + 拆 dev/test")
    parser.add_argument("--num", type=int, default=12, help="生成的新查询总数")
    parser.add_argument("--types", type=str, default="对比型,多跳,边界,反例",
                        help="逗号分隔的生成类型")
    args = parser.parse_args()

    settings = get_settings()
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    per_type = max(1, args.num // len(types))

    print("加载 doc_chunks_full.json ...")
    chunks = load_chunks(CHUNKS_FILE)
    print("构建内存检索 ...")
    retriever = InMemoryRetriever(chunks)
    print(f"知识库块数: {retriever.count}")

    router = _build_router(settings)

    # 1) 生成新查询
    print(f"生成目标: 每类 {per_type} 条，类型 = {types}")
    new_raw: list[dict[str, Any]] = []
    for t in types:
        qs = await generate_queries(router, t, per_type)
        for q in qs:
            new_raw.append({"query": q, "type": t})
        print(f"  [{t}] 生成 {len(qs)} 条")
    print(f"新查询合计: {len(new_raw)}")

    # 2) 对新查询分级标注（候选池 + graded judge）
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def annotate(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        async with sem:
            pool = retriever.candidate_pool(query)
            grades = await graded_judge(router, query, pool)
        item["relevance"] = grades
        item["relevant_ids"] = sorted(grades.keys())
        item["relevant_keywords"] = []
        print(f"  [{idx}/{len(new_raw)}] {query[:28]}  相关={len(grades)}", flush=True)
        return item

    new_items = []
    for i, item in enumerate(new_raw, 1):
        new_items.append(await annotate(i, item))

    # 3) 合并 + 拆 dev/test（分级标注集 → dev；新增 → test，只碰一次）
    old_items, meta = load_queries(GRADED_QUERIES_FILE)
    dev = dict(meta); dev["queries"] = old_items
    test = dict(meta); test["queries"] = new_items
    expanded = dict(meta); expanded["queries"] = old_items + new_items

    OUT_DEV.write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TEST.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_EXPANDED.write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"dev（分级标注集）: {len(old_items)} 条  -> {OUT_DEV.name}")
    print(f"test（新增）: {len(new_items)} 条  -> {OUT_TEST.name}")
    print(f"合并全集: {len(old_items) + len(new_items)} 条  -> {OUT_EXPANDED.name}")
    print("说明：test 集只用于最终评估、只碰一次；调参只在 dev 上做，避免 overfit。")


if __name__ == "__main__":
    asyncio.run(main())
