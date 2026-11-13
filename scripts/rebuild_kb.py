# -*- coding: utf-8 -*-
"""从 doc_chunks_full.json 重建 Milvus 知识库（保留原始 chunk id）。

用法:
  python scripts/rebuild_kb.py

说明: doc_chunks_full.json 是 445 个文档块的源数据（id/content/source）。
      scripts/eval_queries.json 的 relevant_ids 引用这些原始 id，因此重建时必须
      保留原 id，否则评测标注会失效。知识库非空时默认跳过，避免误覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNKS_FILE = Path(__file__).resolve().parent / "doc_chunks_full.json"


async def main(args: argparse.Namespace) -> None:
    # 必须在导入 RAGService 前设定，Settings 会缓存配置。
    os.environ["MILVUS_DB_PATH"] = args.db_path
    from app.core.rag_service import RAGService  # noqa: E402

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"源数据: {len(chunks)} 个文档块 ({CHUNKS_FILE.name})")

    rag = RAGService()
    # 在线服务为稳定性固定为单线程；离线构库可显式提高 CPU 并行度以缩短评测时间。
    if args.torch_threads:
        try:
            import torch

            torch.set_num_threads(args.torch_threads)
            print(f"离线嵌入线程数: {args.torch_threads}")
        except Exception as exc:
            print(f"[警告] 无法设置 Torch 线程数，沿用默认值: {exc}")
    print(f"当前知识库块数: {rag.chunks_count}")

    if rag.chunks_count > 0:
        print("知识库非空，跳过重建（如需重建请先清空 milvus.db）。")
        return

    total = 0
    for i in range(0, len(chunks), args.batch_size):
        part = chunks[i : i + args.batch_size]
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
        # Milvus Lite 在 Windows 下对“插入后立即 flush”会发生 manifest 临时文件重命名冲突。
        # insert 本身同步可见，故这里用 count 校验写入进度；进程退出后的独立读取再验证持久化。
        total += len(items)
        persisted = rag.chunks_count
        if persisted < total:
            raise RuntimeError(f"第 {total} 块写入后校验失败：库中仅 {persisted} 块")
        print(f"  已入库 {total}/{len(chunks)}（已持久化 {persisted}）", flush=True)

    expected_ids = {str(c["id"]) for c in chunks}
    if rag._use_milvus:
        rows = rag._store._client.query(  # 离线校验需要读取精确主键。
            collection_name="rag_chunks", filter="id != ''", output_fields=["id"], limit=len(chunks) + 1
        )
        actual_ids = {str(row["id"]) for row in rows}
        missing = expected_ids - actual_ids
        if missing:
            raise RuntimeError(f"重建完整性校验失败：缺少 {len(missing)} 个原始 chunk ID")
    print(f"重建完成: {rag.chunks_count} 个文档块；chunk ID 校验通过")
    # Milvus Lite 的 gRPC 客户端会保留后台线程；显式关闭保证 CLI 正常退出并释放数据库锁。
    if rag._use_milvus:
        close = getattr(rag._store._client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按原始 chunk ID 重建独立 Milvus Lite 评测库")
    parser.add_argument("--db-path", default="data/milvus_eval_repro.db")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="离线嵌入的 Torch CPU 线程数；0 表示不覆盖运行时设置")
    parser.add_argument("--max-chunks", type=int, default=None, help="仅用于构库冒烟验证")
    parsed_args = parser.parse_args()
    if parsed_args.batch_size < 1 or parsed_args.torch_threads < 0:
        parser.error("--batch-size 必须大于 0")
    asyncio.run(main(parsed_args))
