# -*- coding: utf-8 -*-
"""
RAG 服务：基于 Milvus Lite（嵌入式向量库）+ sentence-transformers 真实语义嵌入。

- 向量检索：Milvus Lite 本地文件持久化，COSINE 相似度
- 关键词检索：BM25（Okapi）内存索引
- 混合检索：RRF（倒数排名融合）

若 Milvus 不可用，自动降级为纯内存版（字符 n-gram 嵌入 + list 暴力搜索）。
"""

from __future__ import annotations

import math
import re
import time
import uuid
from typing import Any

from loguru import logger

from app.config import get_settings
from app.core.rag.generator import RAGGenerator
from app.etl.chunker import DocumentChunker
from app.etl.parser import DocumentParser
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.models.schemas import Citation, RAGResponse, RetrievalResult

# ==============================================================================
# 嵌入模型
# ==============================================================================

_EMBEDDING_MODEL: Any = None

# 设置 HuggingFace 镜像环境变量（必须在 import sentence_transformers 之前）
import os

_settings = get_settings()
_hf_endpoint = getattr(_settings, "hf_endpoint", "") or ""
if _hf_endpoint and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
    logger.info("已设置 HF_ENDPOINT={}", _hf_endpoint)


def _get_embedding_model() -> Any:
    """懒加载 sentence-transformers 模型（首次约 0.5s，模型已缓存）。"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("已加载 embedding 模型: all-MiniLM-L6-v2 (384-dim)")
    return _EMBEDDING_MODEL


# ==============================================================================
# BM25 关键词索引
# ==============================================================================


def _tokenize(text: str) -> list[str]:
    """简单中英文分词：按字母-数字-中文连续段切分。"""
    return [t.lower() for t in re.findall(r"[\w一-鿿㐀-䶿]+", text) if t]


class _BM25:
    """内存 BM25 (Okapi) 关键词索引，用于 RAG 关键词检索。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}
        self._N: int = 0
        self._idf: dict[str, float] = {}

    def add(self, doc_id: str, text: str) -> None:
        """添加文档并更新 IDF。"""
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self._ids.append(doc_id)
        self._contents.append(text)
        self._doc_freqs.append(tf)
        self._doc_lens.append(len(tokens))
        self._N += 1
        self._avgdl = sum(self._doc_lens) / self._N
        # 重建 IDF
        self._idf = {}
        for t, df in self._df.items():
            self._idf[t] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def remove(self, doc_ids: set[str]) -> None:
        """移除指定文档（耗时操作，仅删除少量文档时使用）。"""
        keep_idx = [i for i, did in enumerate(self._ids) if did not in doc_ids]
        if len(keep_idx) == len(self._ids):
            return
        self._ids = [self._ids[i] for i in keep_idx]
        self._contents = [self._contents[i] for i in keep_idx]
        self._doc_freqs = [self._doc_freqs[i] for i in keep_idx]
        self._doc_lens = [self._doc_lens[i] for i in keep_idx]
        # 全量重建统计
        self._df.clear()
        for tf in self._doc_freqs:
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._N = len(self._ids)
        self._avgdl = sum(self._doc_lens) / self._N if self._N else 0.0
        self._idf.clear()
        for t, df in self._df.items():
            self._idf[t] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        """返回 [(doc_id, bm25_score, content), ...] 降序。"""
        if self._N == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: list[tuple[str, float, str]] = []
        k1, b = self.k1, self.b
        avgdl = self._avgdl or 1.0
        for i in range(self._N):
            tf = self._doc_freqs[i]
            dl = self._doc_lens[i]
            s = 0.0
            for t in q_terms:
                idf = self._idf.get(t, 0.0)
                if idf == 0.0:
                    continue
                f = tf.get(t, 0)
                if f == 0:
                    continue
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scores.append((self._ids[i], s, self._contents[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ==============================================================================
# Milvus Lite 向量库（主实现）
# ==============================================================================


class _MilvusRAGStore:
    """基于 Milvus Lite 的文档块存储：向量检索（COSINE 相似度）。"""

    def __init__(self, db_path: str, dim: int = 384) -> None:
        import os

        # 钉死 pymilvus 内置环境变量，防止被 .env 中的值干扰导致 import 崩溃
        if "MILVUS_URI" not in os.environ:
            os.environ["MILVUS_URI"] = ""

        from pymilvus import MilvusClient

        self._client = MilvusClient(db_path)
        self._collection = "rag_chunks"
        self._dim = dim
        self._bm25 = _BM25()
        self._chunk_to_source: dict[str, str] = {}  # chunk_id → source

        # 确保集合存在（id 字段为 varchar 类型，支持自定义 UUID）
        if not self._client.has_collection(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                dimension=self._dim,
                metric_type="COSINE",
                id_type="string",
                max_length=64,
            )
            logger.info(
                "已创建 Milvus 集合 [{}] dim={} metric=COSINE db={}",
                self._collection,
                self._dim,
                db_path,
            )
            # 新建集合后加载到内存
            self._client.load_collection(self._collection)
        else:
            # 已有集合也需要确保加载（重启后可能处于 released 状态）
            try:
                self._client.load_collection(self._collection)
            except Exception:
                pass

    def _ensure_loaded(self) -> None:
        """确保集合已加载到内存（Milvus Lite 在重启或空闲后会 released）。"""
        try:
            self._client.load_collection(self._collection)
        except Exception:
            pass  # 已加载状态下重复 load 不会报错

    @property
    def count(self) -> int:
        try:
            self._ensure_loaded()
            stats = self._client.get_collection_stats(self._collection)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    # ---- 写入 ----

    def add(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        source: str = "",
    ) -> None:
        """插入一条文档块到 Milvus 和 BM25。"""
        self._client.insert(
            collection_name=self._collection,
            data=[{
                "id": doc_id,
                "vector": embedding,
                "content": content,
                "source": source,
            }],
        )
        self._bm25.add(doc_id, content)
        self._chunk_to_source[doc_id] = source

    def add_batch(
        self,
        items: list[tuple[str, str, list[float], str]],
    ) -> None:
        """批量插入 [(doc_id, content, embedding, source), ...]。"""
        if not items:
            return
        data = [
            {"id": did, "vector": vec, "content": text, "source": src}
            for did, text, vec, src in items
        ]
        self._client.insert(collection_name=self._collection, data=data)
        for did, text, _, src in items:
            self._bm25.add(did, text)
            self._chunk_to_source[did] = src

    # ---- 删除 ----

    def remove_by_source(self, source: str) -> int:
        """移除指定来源的所有分块，返回移除数量。"""
        # 1. 找属于该 source 的 chunk id
        chunk_ids = [
            cid for cid, src in self._chunk_to_source.items() if src == source
        ]
        if not chunk_ids:
            return 0
        # 2. 从 Milvus 删除
        ids_to_delete = set(chunk_ids)
        expr = "id in [" + ", ".join(f'"{cid}"' for cid in chunk_ids) + "]"
        try:
            self._client.delete(collection_name=self._collection, filter=expr)
        except Exception as e:
            logger.warning("Milvus delete 失败: {}", e)
        # 3. 从 BM25 删除
        self._bm25.remove(ids_to_delete)
        # 4. 清理映射
        for cid in chunk_ids:
            self._chunk_to_source.pop(cid, None)
        logger.info("已从 Milvus 移除 {} 个分块 (source={})", len(chunk_ids), source)
        return len(chunk_ids)

    # ---- 向量检索 ----

    def vector_search(self, query_vec: list[float], top_k: int) -> list[RetrievalResult]:
        """Milvus 向量检索（COSINE 相似度），分数归一化为越大越好。"""
        self._ensure_loaded()
        if self.count == 0:
            return []
        try:
            raw = self._client.search(
                collection_name=self._collection,
                data=[query_vec],
                limit=top_k,
                output_fields=["content", "source"],
            )
        except Exception as e:
            logger.exception("Milvus search 失败")
            raise RuntimeError(f"向量检索失败: {e}") from e

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(raw[0] if raw else []):
            ent = hit.get("entity", {}) or {}
            score = float(hit.get("distance", 0.0))
            results.append(
                RetrievalResult(
                    id=str(hit.get("id", "")),
                    content=str(ent.get("content") or ""),
                    score=score,
                    metadata={"rank": rank},
                    source="vector",
                )
            )
        return results

    # ---- 关键词检索 ----

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25 关键词检索。"""
        ranked = self._bm25.search(query, top_k)
        results: list[RetrievalResult] = []
        for doc_id, score, content in ranked:
            results.append(
                RetrievalResult(
                    id=doc_id,
                    content=content,
                    score=score,
                    metadata={},
                    source="keyword",
                )
            )
        return results

    # ---- 混合检索 (RRF) ----

    def hybrid_search(self, query_vec: list[float], query_text: str, top_k: int) -> list[RetrievalResult]:
        """向量 + 关键词 → RRF 融合。"""
        vec_results = self.vector_search(query_vec, top_k * 2)
        kw_results = self.keyword_search(query_text, top_k * 2)

        # RRF = 1 / (k + rank + 1), k=60
        rrf_k = 60
        scores: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(vec_results):
            scores[r.id] = 1.0 / (rrf_k + rank + 1)
            best[r.id] = r

        for rank, r in enumerate(kw_results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if r.id not in best:
                best[r.id] = r

        merged_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        merged: list[RetrievalResult] = []
        for rid in merged_ids:
            item = best[rid]
            item.score = scores[rid]
            item.source = "hybrid"
            merged.append(item)
        return merged

    # ---- 文档来源 ----

    def list_sources(self) -> set[str]:
        return set(self._chunk_to_source.values())


# ==============================================================================
# 纯内存降级实现（Milvus 不可用时自动切换）
# ==============================================================================


class _NGramEmbedding:
    """字符 n-gram 哈希嵌入（256 维，无需外部模型）。"""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text = text.lower().strip()
        if not text:
            return vec
        for i in range(len(text)):
            for n in (2, 3):
                if i + n <= len(text):
                    h = hash(text[i : i + n]) % self._dim
                    vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class _InMemoryDocStore:
    """内存文档库（降级方案）：Python list + 暴力搜索。"""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._bm25 = _BM25()

    @property
    def count(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, content: str, embedding: list[float], source: str = "") -> None:
        self._docs.append({
            "id": doc_id, "content": content, "embedding": embedding,
            "source": source,
        })
        self._bm25.add(doc_id, content)

    def remove_by_source(self, source: str) -> int:
        ids = {d["id"] for d in self._docs if d.get("source") == source}
        if not ids:
            return 0
        self._docs = [d for d in self._docs if d.get("source") != source]
        self._bm25.remove(ids)
        return len(ids)

    def vector_search(self, query_vec: list[float], top_k: int) -> list[RetrievalResult]:
        if not self._docs:
            return []
        scored = []
        for d in self._docs:
            dot = sum(a * b for a, b in zip(query_vec, d["embedding"]))
            scored.append((dot, d))
        scored.sort(key=lambda x: -x[0])
        return [
            RetrievalResult(
                id=d["id"], content=d["content"], score=s,
                metadata={"rank": i}, source="vector",
            )
            for i, (s, d) in enumerate(scored[:top_k])
        ]

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        ranked = self._bm25.search(query, top_k)
        return [
            RetrievalResult(id=did, content=content, score=s, metadata={}, source="keyword")
            for did, s, content in ranked
        ]

    def hybrid_search(self, query_vec: list[float], query_text: str, top_k: int) -> list[RetrievalResult]:
        v = self.vector_search(query_vec, top_k * 2)
        k = self.keyword_search(query_text, top_k * 2)
        rrf_k = 60
        scores: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}
        for rank, r in enumerate(v):
            scores[r.id] = 1.0 / (rrf_k + rank + 1)
            best[r.id] = r
        for rank, r in enumerate(k):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if r.id not in best:
                best[r.id] = r
        merged_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            RetrievalResult(
                id=best[rid].id, content=best[rid].content,
                score=scores[rid], source="hybrid", metadata={},
            )
            for rid in merged_ids
        ]

    def list_sources(self) -> set[str]:
        return {d.get("source", "") for d in self._docs if d.get("source")}


# ==============================================================================
# RAG 服务 — 顶层接口
# ==============================================================================


class RAGService:
    """RAG 服务：文档摄入 → 检索（向量/关键词/混合）→ 生成。

    优先使用 Milvus Lite + sentence-transformers 语义嵌入；
    若 Milvus 不可用则自动降级为纯内存版。
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
        self._parser = DocumentParser()
        self._documents: dict[str, dict[str, Any]] = {}  # filename → meta

        # 初始化向量存储
        self._db_path = settings.milvus_db_path or "milvus.db"
        self._use_milvus = False
        self._store: _MilvusRAGStore | _InMemoryDocStore
        self._embed: _NGramEmbedding | Any

        # 尝试加载真实 embedding 模型
        try:
            self._embed = _get_embedding_model()
            embed_dim = 384
            logger.info("RAG 使用 sentence-transformers (384-dim 语义向量)")
        except Exception as e:
            logger.warning("无法加载 sentence-transformers，降级为字符 n-gram: {}", e)
            self._embed = _NGramEmbedding()
            embed_dim = 256

        # 尝试使用 Milvus Lite
        try:
            self._store = _MilvusRAGStore(self._db_path, dim=embed_dim)
            self._use_milvus = True
            logger.info("RAG 使用 Milvus Lite: {}", self._db_path)
        except Exception as e:
            logger.warning("无法初始化 Milvus，降级为内存版: {}", e)
            self._store = _InMemoryDocStore()
            self._use_milvus = False

        # LLM
        model_cfg = ModelConfig(
            model_id=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
        self._model_router = ModelRouter([model_cfg])
        self._model_name = settings.openai_model
        self._generator = RAGGenerator(
            llm=self._make_llm_adapter(),
            model_name=self._model_name,
        )

        logger.info(
            "RAGService 初始化完成: store={}, embed_dim={}, chunks={}",
            "Milvus" if self._use_milvus else "InMemory",
            embed_dim,
            self._store.count,
        )

    def _make_llm_adapter(self) -> Any:
        class _LLMAdapter:
            def __init__(self, router: ModelRouter) -> None:
                self._r = router

            async def ainvoke(self, messages: list[dict], **kwargs: Any) -> str:
                resp = await self._r.chat(messages, **kwargs)
                return resp.content

        return _LLMAdapter(self._model_router)

    @property
    def store_type(self) -> str:
        return "Milvus" if self._use_milvus else "InMemory"

    # ------------------------------------------------------------------
    # 嵌入辅助
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list[float]:
        """把文本编码为向量。"""
        if isinstance(self._embed, _NGramEmbedding):
            return self._embed.embed_query(text)
        import asyncio
        return self._embed.encode(text, normalize_embeddings=True).tolist()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码。"""
        if isinstance(self._embed, _NGramEmbedding):
            return [self._embed.embed_query(t) for t in texts]
        import asyncio
        vecs = self._embed.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    # ------------------------------------------------------------------
    # 文档摄入
    # ------------------------------------------------------------------

    async def ingest_text(self, text: str, source: str = "manual") -> int:
        """直接摄入文本（分块后入库）。"""
        if not text.strip():
            return 0
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0
        embeddings = self._encode_batch(chunks)
        items = [
            (str(uuid.uuid4()), c, emb, source)
            for c, emb in zip(chunks, embeddings)
        ]
        if self._use_milvus:
            self._store.add_batch(items)
        else:
            for did, c, emb, src in items:
                self._store.add(did, c, emb, src)
        logger.info("RAG 摄入文本: {} 字符 → {} 块", len(text), len(chunks))
        return len(chunks)

    async def ingest_file(
        self,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
    ) -> int:
        """摄入文件（PDF/TXT 等）。"""
        parsed = self._parser.parse_bytes(data, filename, mime_type)
        if not parsed.text.strip():
            logger.warning("文件解析结果为空: {}", filename)
            return 0
        chunks = self._chunker.chunk(parsed.text)
        if not chunks:
            return 0
        embeddings = self._encode_batch(chunks)
        items = [
            (str(uuid.uuid4()), c, emb, filename)
            for c, emb in zip(chunks, embeddings)
        ]
        if self._use_milvus:
            self._store.add_batch(items)
        else:
            for did, c, emb, src in items:
                self._store.add(did, c, emb, src)
        self._documents[filename] = {
            "filename": filename,
            "mime_type": mime_type or parsed.mime_type,
            "chunk_count": len(chunks),
            "char_count": len(parsed.text),
            "uploaded_at": time.time(),
        }
        logger.info("RAG 摄入文件: {} → {} 块", filename, len(chunks))
        return len(chunks)

    def list_documents(self) -> list[dict[str, Any]]:
        docs = list(self._documents.values())
        docs.sort(key=lambda d: -d["uploaded_at"])
        return docs

    def delete_document(self, filename: str) -> bool:
        if filename not in self._documents:
            return False
        n = self._store.remove_by_source(filename)
        del self._documents[filename]
        logger.info("RAG 删除文档: {} (移除 {} 块)", filename, n)
        return True

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[RetrievalResult]:
        """执行检索（支持 vector / keyword / hybrid）。"""
        if self._store.count == 0:
            return []

        mode = mode.lower().strip()
        if mode == "vector":
            q_vec = self._encode(query)
            return self._store.vector_search(q_vec, top_k)
        elif mode == "keyword":
            return self._store.keyword_search(query, top_k)
        else:
            q_vec = self._encode(query)
            return self._store.hybrid_search(q_vec, query, top_k)

    # ------------------------------------------------------------------
    # RAG 对话
    # ------------------------------------------------------------------

    async def chat(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> RAGResponse:
        """检索 → 生成（含引用标注）。"""
        if self._store.count == 0:
            return RAGResponse(
                answer="知识库为空，请先上传文档。",
                citations=[],
                raw_contexts=[],
                model=self._model_name,
            )

        contexts = await self.search(query, top_k=top_k, mode=mode)
        response = await self._generator.generate(
            query=query,
            contexts=contexts,
            chat_history=chat_history or [],
        )
        return response

    @property
    def chunks_count(self) -> int:
        return self._store.count
