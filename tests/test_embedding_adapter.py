from __future__ import annotations

from app.core.rag_service import SentenceTransformerEmbeddingAdapter


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def encode(self, text: str, normalize_embeddings: bool = False):
        self.calls.append((text, normalize_embeddings))
        return _Vector([0.1, 0.2, 0.3])


class _Vector(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


def test_sentence_transformer_adapter_uses_normalized_query_embedding():
    model = _FakeModel()
    adapter = SentenceTransformerEmbeddingAdapter(model)

    result = adapter.embed_query("血压管理")

    assert result == [0.1, 0.2, 0.3]
    assert model.calls == [("血压管理", True)]
    assert all(isinstance(value, float) for value in result)
