from app.infrastructure.vectordb.base import VectorStore


def test_vector_store_protocol_is_runtime_checkable():
    class FakeStore:
        count = 0

        def add(self, doc_id, content, embedding, source=""):
            pass

        def add_batch(self, items):
            pass

        def vector_search(self, query_vector, top_k):
            return []

        def keyword_search(self, query, top_k):
            return []

        def hybrid_search(self, query_vector, query, top_k):
            return []

        def remove_by_source(self, source):
            return 0

    assert isinstance(FakeStore(), VectorStore)
