# -*- coding: utf-8 -*-
"""把当前 Milvus 知识库导出为 doc_chunks_full.json（id/content/source）。

用途：
  知识库通过 Web UI / API 重新入库或扩充后，Milvus 里的 chunk id 会整体变更，
  而 scripts/doc_chunks_full.json 与评测标注（eval_queries.v2.json 的 relevant_ids）
  仍是旧 id，导致评测全 0。用本脚本从 milvus.db 重新导出当前 chunk，
  再重跑 reannotate_queries.py 对齐标注即可恢复评测有效性。

使用：
  D:/Code/MyAgent/ai-agent-interview-guide/aiAgent/Scripts/python.exe scripts/dump_kb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import MilvusClient  # noqa: E402

DB_PATH = str(PROJECT_ROOT / "milvus.db")
COLLECTION = "rag_chunks"
OUTPUT = Path(__file__).resolve().parent / "doc_chunks_full.json"


def main() -> int:
    client = MilvusClient(DB_PATH)
    if not client.has_collection(COLLECTION):
        print(f"[错误] 集合不存在: {COLLECTION} (db={DB_PATH})", file=sys.stderr)
        return 1

    # 重启/空闲后集合可能处于 released 状态，需先 load 再查询。
    client.load_collection(COLLECTION)

    # 与 rag_service._MilvusRAGStore._rebuild_bm25 相同的查询方式，确保字段名一致。
    rows = client.query(
        collection_name=COLLECTION,
        filter="id != ''",
        output_fields=["id", "content", "source"],
        limit=10000,
    )
    chunks = [
        {"id": r["id"], "content": r.get("content", ""), "source": r.get("source", "")}
        for r in rows
    ]
    OUTPUT.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已导出 {len(chunks)} 个 chunk -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
