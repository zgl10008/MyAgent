# -*- coding: utf-8 -*-
"""
清理「反例」类型标注：把扩量时误标为反例的领域内/跨域问题改回正确类型，
并把库外查询的伪相关标注（引用误命中）清空，使反例 = 纯「应判空」集合。

背景（doc 7.1 口径说明）：
  expand_to_100.py 生成反例时，LLM 把 4 条领域内/跨域问题标成了反例：
    - 3 条高血压领域内（停药判断 / 饮食谣言 / ST段抬高鉴别）→ 边界型
    - 1 条跨域（阿尔茨海默 ↔ 心脏病，KB 真有房颤→痴呆 chunk）→ 多跳
  另有 3 条「帕金森病怎么治」是真正库外（神经退行性疾病），但 LLM judge 因
  候选池里参考文献条目出现「帕金森」字样而给了伪相关（grade 1/2）→ 清空。

改动（按 query 原文精确匹配，只改 type / relevance，不改 query 文本与集合成员）：
  RELABEL:       type 改为正确类型
  ZERO_RELEVANCE: relevance={} 且 relevant_ids=[]（保留 type=反例）

输出：就地更新 eval_queries_full.json / dev / test 三个文件，打印新分布。

使用：
  python scripts/relabel_fanli.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

FILES = [
    SCRIPTS_DIR / "eval_queries_full.json",
    SCRIPTS_DIR / "eval_queries_dev.json",
    SCRIPTS_DIR / "eval_queries_test.json",
]

RELABEL = {
    "高血压是不是不用吃药了？我血压正常了。": "边界",
    "高血压患者可以吃榴莲吗？还是说榴莲能降血压？": "边界",
    "心电图ST段抬高就一定是心肌梗死吗？如果不是，还可能是什么？": "边界",
    "我奶奶得了阿尔茨海默症，和心脏病有关系吗？": "多跳",
}

ZERO_RELEVANCE = {
    "帕金森病怎么治？",
    "帕金森病如何治疗？",
    "帕金森病怎么治疗？",
}


def process(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = {k: v for k, v in data.items() if k != "queries"}
    queries = data["queries"]

    n_relabel = n_zero = 0
    for q in queries:
        qtext = q["query"].strip()
        if qtext in RELABEL:
            q["type"] = RELABEL[qtext]
            n_relabel += 1
        if qtext in ZERO_RELEVANCE:
            q["relevance"] = {}
            q["relevant_ids"] = []
            n_zero += 1

    out = dict(meta)
    out["queries"] = queries
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(q.get("type") or "未分类" for q in queries)
    fanli = [q for q in queries if q.get("type") == "反例"]
    fanli_nonempty = sum(1 for q in fanli if q.get("relevance"))
    print(f"{path.name}: 改标 {n_relabel} / 清空 {n_zero}  | {dict(counts)}")
    print(f"    反例 {len(fanli)} 条，其中仍带相关标注 {fanli_nonempty} 条")


def main() -> None:
    for f in FILES:
        if not f.exists():
            print(f"[跳过] 不存在 {f.name}")
            continue
        process(f)


if __name__ == "__main__":
    main()
