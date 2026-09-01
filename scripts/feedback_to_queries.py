# -*- coding: utf-8 -*-
"""在线负反馈回流：把 data/feedback.jsonl 里的负反馈 query 转成待标注测试集。

背景（doc 4.3 在线隐式负反馈）：
  离线评测测不到的"真实失败"来自在线：用户点踩、打分低、或明确说「不对/答非所问」。
  这些 query 回流到测试集，人工/LLM 补标注后进 eval，形成"在线失败 → 离线回归"闭环。

本脚本只做「提取 + 去重 + 导出」，不做标注——导出的 query 的 relevant_ids 为空，
需后续用 reannotate_graded.py（或人工）标注后再进 eval。

负反馈判定（满足任一）：
  - helpful == false
  - rating <= 2
  - comment 含否定关键词（不对/错/答非所问/没回答/不准确/胡说/乱编）

使用：
  python scripts/feedback_to_queries.py                     # 读取 data/feedback.jsonl
  python scripts/feedback_to_queries.py --file data/feedback.jsonl

输出：scripts/eval_queries.feedback.json（待标注，带 _source="feedback" 与 comment 溯源）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_FEEDBACK = PROJECT_ROOT / "data" / "feedback.jsonl"
OUTPUT_FILE = SCRIPTS_DIR / "eval_queries.feedback.json"

# 覆盖现有查询集，避免重复回流
EXISTING_QUERY_FILES = [
    "eval_queries.v2.json",
    "eval_queries.graded.json",
    "eval_queries_dev.json",
    "eval_queries_test.json",
    "eval_queries_expanded.json",
    "eval_queries.json",
]

NEG_KEYWORDS = ["不对", "错了", "错误", "答非所问", "没回答", "没找到", "不准确",
                "胡说", "乱编", "瞎说", "不相关", "没用"]


def _is_negative(rec: dict[str, Any]) -> bool:
    if rec.get("helpful") is False:
        return True
    rating = rec.get("rating")
    if rating is not None and int(rating) <= 2:
        return True
    comment = str(rec.get("comment") or "")
    return any(kw in comment for kw in NEG_KEYWORDS)


def _norm(q: str) -> str:
    return q.strip().lower().replace(" ", "")


def _load_existing_queries() -> set[str]:
    seen: set[str] = set()
    for name in EXISTING_QUERY_FILES:
        path = SCRIPTS_DIR / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data["queries"] if isinstance(data, dict) else data
            for item in items:
                q = item.get("query")
                if q:
                    seen.add(_norm(q))
        except Exception:
            continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="负反馈回流成待标注测试集")
    parser.add_argument("--file", type=str, default=str(DEFAULT_FEEDBACK),
                        help="反馈 JSONL 文件路径")
    args = parser.parse_args()

    fb_path = Path(args.file)
    if not fb_path.exists():
        print(f"[提示] 反馈文件不存在（尚无负反馈）: {fb_path}")
        print("       可先 POST /rag/feedback 采集，或手动造几条测试。")
        return

    records = []
    with open(fb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    negative = [r for r in records if _is_negative(r)]
    print(f"反馈总数: {len(records)}  负反馈: {len(negative)}")

    seen = _load_existing_queries()
    fresh: list[dict[str, Any]] = []
    for r in negative:
        q = str(r.get("query") or "").strip()
        if not q:
            continue
        if _norm(q) in seen:
            continue
        seen.add(_norm(q))
        fresh.append({
            "query": q,
            "type": "反馈",
            "relevant_ids": [],
            "relevant_keywords": [],
            "_source": "feedback",
            "_comment": r.get("comment"),
            "_feedback_ts": r.get("ts"),
        })

    out = {
        "_说明": [
            "由 scripts/feedback_to_queries.py 从 data/feedback.jsonl 提取负反馈生成。",
            "relevant_ids 为空，需用 reannotate_graded.py（或人工）标注后再进 eval。",
            "_source=feedback 用于溯源：这是在线真实失败，不是离线设计的 happy path。",
        ],
        "queries": fresh,
    }
    OUTPUT_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 60)
    print(f"新回流 query: {len(fresh)} 条（已去重，跳过 {len(negative) - len(fresh)} 条已覆盖）")
    print(f"输出: {OUTPUT_FILE.name}")
    if fresh:
        print("下一步: 用 reannotate_graded.py 对这批 query 补分级标注后，"
              "合并进 eval_queries_test.json 做最终评估。")


if __name__ == "__main__":
    main()
