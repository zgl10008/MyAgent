# -*- coding: utf-8 -*-
"""
用 LLM-as-judge 重新标注 eval_queries 的相关文档，修复评测集标注质量问题。

背景：
  当前 eval_queries.json 的 relevant_ids 由 eval_full_kb.py 的「关键词+语义 top-10」
  自动标注生成，存在两类问题：
    1) 45%（13/29）查询的 relevant_ids 为空 → 这些查询对每个指标都贡献 0；
    2) 其余查询平均只标 ~1.55 个相关块 → 答案跨块时 recall 被系统性低估。
  这属于「标注偏差(bias)」，不是「样本量(variance)」问题，扩集无法解决。

方法（与依赖 Milvus 的第一版不同，本版自包含、离线、可复现）：
  数据源 = scripts/doc_chunks_full.json（445 个 chunk 的纯 JSON，已核对其 id
  与 eval_queries.json 的 relevant_ids 100% 对齐）。
  对每条查询，用 bge-m3 向量 + BM25 在 445 个 chunk 上做三路召回（vector/keyword/
  hybrid-RRF）取并集作为候选池，再由 LLM 一次性判断候选池中哪些块真正相关。

局限（诚实声明）：
  候选池来自检索结果，因此「检索完全没召回的真正相关块」不会进入候选池，
  标注对这类块是盲的。这是检索派生物候选池的通病，代价可接受。

输出：scripts/eval_queries.v2.json（不覆盖原文件，便于对比）。

使用：
  D:/Code/MyAgent/ai-agent-interview-guide/aiAgent/Scripts/python.exe scripts/reannotate_queries.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.core.rag_service import _BM25, _get_embedding_model
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter

CHUNKS_FILE = PROJECT_ROOT / "scripts" / "doc_chunks_full.json"
QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "scripts" / "eval_queries.v2.json"
# 带 type 标签的 P1 产物（分级标注/一致性/扩量都应以它为准，而不是旧的无 type 文件）
V2_QUERIES_FILE = PROJECT_ROOT / "scripts" / "eval_queries.v2.json"

CANDIDATE_TOP_K = 20          # 每种检索模式的候选召回数
CONTENT_TRUNC = 400           # 送入 LLM 判断的每个 chunk 截断字符数
MAX_CONCURRENCY = 4           # LLM 判断并发数
# 注意：deepseek-v4-flash 是「推理模型」，回复 = reasoning_content(思维链) + content(最终答案)。
# max_tokens 同时计入两者，设太小会导致模型把额度全花在思维链上、content 为空。
# 判断任务思维链较长，需给足额度（最终 JSON 本身很短）。
MAX_TOKENS = 6000             # LLM 单次回复上限
RRF_K = 60                    # RRF 融合常数（与 rag_service 一致）

JUDGE_SYSTEM = (
    "你是医学信息相关性标注员。判断哪些候选文档块能够回答用户的问题。\n"
    "只输出一个 JSON 对象，不要输出任何其他内容，格式：\n"
    '{"relevant_ids": [0, 3, 5]}\n'
    "其中 relevant_ids 是「与问题相关、能提供回答所需信息」的文档块编号列表，"
    "对应下方候选块的 [编号]（编号从 0 开始）。\n"
    "判定标准：\n"
    "  - 相关：文档块包含能直接回答该问题的事实、数据、定义或建议。\n"
    "  - 不相关：与问题主题无关、只是泛泛提及、或提供不了实质信息。\n"
    "如果没有任何块相关，输出 {\"relevant_ids\": []}。"
)


class InMemoryRetriever:
    """基于 doc_chunks_full.json 的纯内存检索：bge-m3 向量 + BM25 三路召回。"""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._embeddings: np.ndarray | None = None

        model = _get_embedding_model()
        bm25 = _BM25()
        for c in chunks:
            cid = c.get("id")
            content = c.get("content", "") or ""
            if not cid or not content:
                continue
            self._ids.append(cid)
            self._contents.append(content)
            bm25.add(cid, content)

        contents = self._contents
        vecs = model.encode(
            contents, normalize_embeddings=True, show_progress_bar=True
        )
        self._embeddings = np.asarray(vecs, dtype=np.float32)
        self._bm25 = bm25

    @property
    def count(self) -> int:
        return len(self._ids)

    def _vector_top(self, query: str, top_k: int) -> list[tuple[str, float]]:
        model = _get_embedding_model()
        qvec = np.asarray(
            model.encode(query, normalize_embeddings=True), dtype=np.float32
        )
        scores = self._embeddings @ qvec  # 已归一化 → 点积即余弦
        order = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def _keyword_top(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return [(cid, s) for cid, s, _ in self._bm25.search(query, top_k)]

    def _hybrid_top(self, query: str, top_k: int) -> list[tuple[str, float]]:
        v = self._vector_top(query, top_k * 2)
        k = self._keyword_top(query, top_k * 2)
        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(v):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(k):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        merged = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        return merged

    def candidate_pool(self, query: str) -> dict[str, str]:
        """三路召回 Top-N 并集，返回 {chunk_id: content}。"""
        seen: dict[str, str] = {}
        idx = {cid: i for i, cid in enumerate(self._ids)}
        for hits in (
            self._vector_top(query, CANDIDATE_TOP_K),
            self._keyword_top(query, CANDIDATE_TOP_K),
            self._hybrid_top(query, CANDIDATE_TOP_K),
        ):
            for cid, _ in hits:
                seen.setdefault(cid, self._contents[idx[cid]])
        return seen


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中鲁棒地提取 JSON 对象（容忍 markdown 围栏/前后缀文本）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def load_queries(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["queries"] if isinstance(data, dict) and "queries" in data else data
    meta = {k: v for k, v in data.items() if k != "queries"} if isinstance(data, dict) else {}
    return list(items), meta


def load_chunks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("chunks", [])


def _build_router(settings: Any) -> ModelRouter:
    return ModelRouter([
        ModelConfig(
            model_id=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
    ])


def _build_prompt(query: str, pool: dict[str, str]) -> tuple[str, list[str]]:
    ids = list(pool.keys())
    lines = [f"[{i}] {pool[cid][:CONTENT_TRUNC]}" for i, cid in enumerate(ids)]
    user = (
        f"用户问题：{query}\n\n"
        f"候选文档块：\n" + "\n".join(lines) + "\n\n"
        "请判断哪些块相关，输出 JSON。"
    )
    return user, ids


async def llm_judge(
    router: ModelRouter, query: str, pool: dict[str, str]
) -> set[str]:
    """LLM 一次性判断候选池中哪些块相关，返回相关 chunk id 集合。"""
    if not pool:
        return set()
    user, ids = _build_prompt(query, pool)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    for attempt in range(3):
        try:
            resp = await router.chat(
                messages, temperature=0.0, max_tokens=MAX_TOKENS
            )
            data = _extract_json(resp.content)
            if data is None:
                raise ValueError(f"无法解析 JSON: {resp.content[:120]!r}")
            idxs = data.get("relevant_ids", [])
            return {
                ids[i]
                for i in idxs
                if isinstance(i, int) and 0 <= i < len(ids)
            }
        except Exception as e:
            print(f"    [重试 {attempt + 1}] LLM 判断失败: {e}")
            await asyncio.sleep(1.0)
    return set()


async def main() -> None:
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

    items, meta = load_queries(QUERIES_FILE)
    print(f"测试查询数: {len(items)}")
    print("-" * 60)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def reannotate_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        existing = set(item.get("relevant_ids", []))
        async with sem:
            pool = retriever.candidate_pool(query)
            judged = await llm_judge(router, query, pool)

        # 护栏：LLM 判断为空但原标注非空时，保留原标注（避免因 LLM 偶发失败抹掉标签）
        new_ids = list(judged) if judged else list(existing)

        print(
            f"[{idx}/{len(items)}] {query[:40]}  "
            f"候选={len(pool)}  原={len(existing)}  新={len(new_ids)}",
            flush=True,
        )
        return {
            "query": query,
            "relevant_ids": new_ids,
            "relevant_keywords": item.get("relevant_keywords", []),
            "_reannotated": True,
            "_orig_relevant_ids": sorted(existing),
        }

    new_items = []
    for i, item in enumerate(items, 1):
        new_items.append(await reannotate_one(i, item))

    # 汇总
    n_empty_before = sum(1 for it in items if not it.get("relevant_ids"))
    n_empty_after = sum(1 for it in new_items if not it.get("relevant_ids"))
    avg_before = sum(len(it.get("relevant_ids", [])) for it in items) / len(items)
    avg_after = sum(len(it["relevant_ids"]) for it in new_items) / len(items)

    print("\n" + "=" * 60)
    print("重标注完成")
    print(f"  空标注查询: {n_empty_before} → {n_empty_after}")
    print(f"  平均相关块: {avg_before:.2f} → {avg_after:.2f}")

    out = dict(meta)
    out["_说明"] = [
        "由 scripts/reannotate_queries.py 用 LLM-as-judge 重标注生成。",
        "数据源 = doc_chunks_full.json；候选池 = bge-m3 向量/BM25/RRF 三路召回 Top-%d 并集。" % CANDIDATE_TOP_K,
    ]
    out["queries"] = new_items
    OUTPUT_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写出: {OUTPUT_FILE.name}（原文件未改动）")


if __name__ == "__main__":
    asyncio.run(main())
