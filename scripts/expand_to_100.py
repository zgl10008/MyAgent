# -*- coding: utf-8 -*-
"""
扩充评测查询集到 100+，并做类型均衡的 dev/test 分层拆分。

背景（doc 3.1 + 6.4，RAG完整评测方案）：
  当前 unique query 仅 36 条（事实18 / 多跳11 / 边界2 / 对比1 / 反例1 / 反馈3），
  类型严重失衡（对比/边界/反例几乎为空），且样本量不足以支撑严格显著性。
  本脚本：
    1) 收集现有全部 unique query（expanded 33 + feedback 分级 3）；
    2) 按目标分布计算各类型缺口；
    3) 用 LLM 生成缺口 query（5 类，含事实型，分批生成去重）；
    4) 对新 query 走候选池 + 分级标注（复用 reannotate_graded.graded_judge）；
    5) 合并去重后按类型分层抽样拆 dev（~80%）/ test（~20%，feedback 固定进 test）。

输出：
  scripts/eval_queries_full.json   合并全集（含 type + relevance）
  scripts/eval_queries_dev.json    调参集
  scripts/eval_queries_test.json   最终评估集（只碰一次）

使用：
  python scripts/expand_to_100.py --limit 5   # 冒烟：只生成 5 条
  python scripts/expand_to_100.py             # 全量：按缺口补齐到 100+
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from reannotate_queries import (  # noqa: E402
    CHUNKS_FILE,
    InMemoryRetriever,
    _build_router,
    _extract_json,
    load_chunks,
    load_queries,
)
from reannotate_graded import graded_judge  # noqa: E402

EXPANDED_FILE = SCRIPTS_DIR / "eval_queries_expanded.json"
FEEDBACK_GRADED_FILE = SCRIPTS_DIR / "eval_queries.feedback.graded.json"

OUT_FULL = SCRIPTS_DIR / "eval_queries_full.json"
OUT_DEV = SCRIPTS_DIR / "eval_queries_dev.json"
OUT_TEST = SCRIPTS_DIR / "eval_queries_test.json"

MAX_TOKENS = 6000
MAX_CONCURRENCY = 4
GEN_BATCH = 6                # 每类型单次生成条数（分批，避免 LLM 一次生成太多不稳）
TEST_RATIO = 0.2             # test 占比
SEED = 42

# 目标类型分布（feedback 不生成，保持现有 3 条回流）。合计 ≈ 24+24+16+16+16+3 = 99。
TARGET_COUNTS = {"事实型": 24, "多跳": 24, "对比型": 16, "边界": 16, "反例": 16}

GEN_PROMPTS = {
    "事实型": (
        "你是心血管慢病测试集设计者。生成「事实型」用户提问：单点知识抽取，答案集中在"
        "单一文档块（如「高血压诊断标准是多少」「某类降压药的禁忌症」「某指标的正常范围」）。"
        "领域限于高血压/糖尿病/冠心病/心衰/血脂。只输出 JSON：{\"queries\": [\"...\"]}"
    ),
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


async def generate_queries(router: Any, qtype: str, num: int) -> list[str]:
    prompt = GEN_PROMPTS[qtype] + f"\n请生成 {num} 条，彼此不重复。"
    messages = [
        {"role": "system", "content": "你只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(messages, temperature=0.8, max_tokens=MAX_TOKENS)
            data = _extract_json(resp.content)
            qs = data.get("queries", []) if data else []
            qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
            if qs:
                return qs[:num]
        except Exception as e:
            print(f"    [生成重试 {attempt + 1}] {e}")
            await asyncio.sleep(1.0)
    return []


def collect_existing() -> list[dict[str, Any]]:
    """合并现有全部 unique query（expanded + feedback 分级），按 query 文本去重。"""
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for f in (EXPANDED_FILE, FEEDBACK_GRADED_FILE):
        if not f.exists():
            print(f"  [跳过] 不存在 {f.name}")
            continue
        qs, _ = load_queries(f)
        for q in qs:
            key = q["query"].strip()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(q)
    return items


def stratified_split(
    items: list[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 type 分层抽样：feedback 固定进 test，其余每类 ~test_ratio 进 test。"""
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_type[it.get("type") or "未分类"].append(it)

    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for tname, group in by_type.items():
        rng.shuffle(group)
        if tname == "反馈":
            # 负反馈回流本就属于最终评估集
            test.extend(group)
            continue
        n_test = max(1, round(len(group) * test_ratio)) if len(group) >= 2 else 0
        n_test = min(n_test, len(group) - 1) if len(group) > 1 else 0
        test.extend(group[:n_test])
        dev.extend(group[n_test:])
    return dev, test


async def main() -> None:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="扩量到 100+ 并分层拆 dev/test")
    parser.add_argument("--limit", type=int, default=0,
                        help="只生成前 N 条新查询（0=按缺口全量，冒烟用）")
    args = parser.parse_args()

    settings = get_settings()
    print("加载 doc_chunks_full.json ...")
    chunks = load_chunks(CHUNKS_FILE)
    print("构建内存检索（bge-m3 + BM25 + RRF）...")
    retriever = InMemoryRetriever(chunks)
    print(f"知识库块数: {retriever.count}")
    router = _build_router(settings)

    # 1) 收集现有 + 算缺口
    existing = collect_existing()
    existing_counts = Counter(it.get("type") or "未分类" for it in existing)
    print(f"\n现有 unique query: {len(existing)}  {dict(existing_counts)}")

    gaps: dict[str, int] = {}
    for t, target in TARGET_COUNTS.items():
        gaps[t] = max(0, target - existing_counts.get(t, 0))
    gaps = {t: g for t, g in sorted(gaps.items(), key=lambda kv: -kv[1]) if g > 0}
    total_gap = sum(gaps.values())
    print(f"各类型缺口: {gaps}  （合计 {total_gap}）")

    # 2) 分批生成（去重）+ 3) 分级标注
    seen = {it["query"].strip() for it in existing}
    new_items: list[dict[str, Any]] = []
    generated = 0
    for t, gap in gaps.items():
        while gap > 0 and (args.limit == 0 or generated < args.limit):
            batch = min(GEN_BATCH, gap)
            if args.limit > 0:
                batch = min(batch, args.limit - generated)
            qs = await generate_queries(router, t, batch)
            added = 0
            for q in qs:
                key = q.strip()
                if key in seen:
                    continue
                seen.add(key)
                new_items.append({"query": key, "type": t})
                added += 1
            gap -= added
            generated += added
            print(f"  [{t}] 生成 {len(qs)} → 去重后 +{added}，剩余缺口 {gap}")
            if added == 0:
                # 连续去重为 0，避免死循环
                break
    print(f"\n新生成合计: {len(new_items)}")

    if not new_items:
        print("[提示] 无新 query（可能 limit=0 且无缺口），直接拆分现有集。")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def annotate(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        async with sem:
            pool = retriever.candidate_pool(query)
            grades = await graded_judge(router, query, pool)
        item["relevance"] = grades
        item["relevant_ids"] = sorted(grades.keys())
        print(f"  [{idx}/{len(new_items)}] {query[:28]}  相关={len(grades)}", flush=True)
        return item

    annotated = []
    for i, item in enumerate(new_items, 1):
        annotated.append(await annotate(i, item))

    # 空标注兜底：分级 judge 偶发返回空 content（推理模型思维链耗光 token 预算），
    # 对事实/多跳/对比/边界型这是标注失败而非真相（反例型空标注可能是正确结论，
    # 但重试一次确认也无害）。对空标注项最多再重试 2 轮。
    for retry_round in range(1, 3):
        empties = [it for it in annotated if not it["relevance"]]
        if not empties:
            break
        print(f"\n[兜底重试 第{retry_round}轮] 空标注 {len(empties)} 条：")
        for i, it in enumerate(empties, 1):
            await annotate(i, it)
    n_empty = sum(1 for it in annotated if not it["relevance"])
    print(f"\n空标注残留: {n_empty}/{len(annotated)}")
    if n_empty:
        for it in annotated:
            if not it["relevance"]:
                print(f"    [仍空] {it['type']}: {it['query'][:40]}")

    # 4) 合并 + 分层拆分
    all_items = existing + annotated
    dev, test = stratified_split(all_items, TEST_RATIO, SEED)

    meta = {"_说明": [
        "由 scripts/expand_to_100.py 生成：LLM 扩量 + 分级标注 + 类型均衡分层拆分。",
        "relevance = {chunk_id: grade}，grade 1=部分相关 2=高度相关；relevant_ids = grade>0 集合。",
        "dev=调参集；test=最终评估集（只碰一次）；feedback 固定进 test。",
        f"目标分布 {TARGET_COUNTS}；test_ratio={TEST_RATIO}；seed={SEED}。",
    ]}

    def dump(path: Path, qs: list[dict[str, Any]]) -> None:
        out = dict(meta)
        out["queries"] = qs
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    dump(OUT_FULL, all_items)
    dump(OUT_DEV, dev)
    dump(OUT_TEST, test)

    def counts(qs: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(q.get("type") or "未分类" for q in qs))

    print("\n" + "=" * 60)
    print(f"合并全集: {len(all_items)} 条  {counts(all_items)}  -> {OUT_FULL.name}")
    print(f"dev（调参）: {len(dev)} 条  {counts(dev)}  -> {OUT_DEV.name}")
    print(f"test（最终）: {len(test)} 条  {counts(test)}  -> {OUT_TEST.name}")
    print("说明：test 只碰一次；调参只在 dev 上做，避免 overfit。")


if __name__ == "__main__":
    asyncio.run(main())
