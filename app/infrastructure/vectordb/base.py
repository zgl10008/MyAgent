"""向量存储抽象：隔离 RAG 业务层与具体向量数据库。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models.schemas import RetrievalResult


VectorItem = tuple[str, str, list[float], str]


@runtime_checkable
class VectorStore(Protocol):
    """RAG 所需的最小向量存储边界。

    具体实现可以是 Milvus、pgvector 或测试用内存存储；业务层只依赖这些操作。
    """

    @property
    def count(self) -> int:
        ...

    def add(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        source: str = "",
    ) -> None:
        ...

    def add_batch(self, items: list[VectorItem]) -> None:
        ...

    def vector_search(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        ...

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        ...

    def hybrid_search(
        self,
        query_vector: list[float],
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...

    def remove_by_source(self, source: str) -> int:
        ...
