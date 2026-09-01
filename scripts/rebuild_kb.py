# -*- coding: utf-8 -*-
"""从 doc_chunks_full.json 重建 Milvus 知识库（保留原始 chunk id）。

用法:
  python scripts/rebuild_kb.py

说明: doc_chunks_full.json 是 445 个文档块的源数据（id/content/source）。
      scripts/eval_queries.json 的 relevant_ids 引用这些原始 id，因此重建时必须
      保留原 id，否则评测标注会失效。知识库非空时默认跳过，避免误覆盖。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rag_service import RAGService  # noqa: E402

CHUNKS_FILE = Path(__file__).resolve().parent / "doc_chunks_full.json"
BATCH_SIZE = 100


async def main() -> None:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"源数据: {len(chunks)} 个文档块 ({CHUNKS_FILE.name})")

    rag = RAGService()
    print(f"当前知识库块数: {rag.chunks_count}")

    if rag.chunks_count > 0:
        print("知识库非空，跳过重建（如需重建请先清空 milvus.db）。")
        return

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        part = chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in part]
        embeddings = await rag._encode_batch(texts)
        items = [
            (c["id"], c["content"], emb, c["source"])
            for c, emb in zip(part, embeddings)
        ]
        if rag._use_milvus:
            rag._store.add_batch(items)
        else:
            for did, content, emb, src in items:
                rag._store.add(did, content, emb, src)
        total += len(items)
        print(f"  已入库 {total}/{len(chunks)}")

    print(f"重建完成: {rag.chunks_count} 个文档块")


if __name__ == "__main__":
    asyncio.run(main())
