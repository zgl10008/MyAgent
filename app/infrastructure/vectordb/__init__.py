"""向量数据库封装与可替换存储抽象。"""

from app.infrastructure.vectordb.base import VectorStore
from app.infrastructure.vectordb.milvus_client import MilvusManager

__all__ = ["MilvusManager", "VectorStore"]
