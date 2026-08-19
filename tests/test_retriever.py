"""Unit tests for embedding and FAISS vector retrieval."""

import tempfile
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.pipeline.schemas import Chunk
from src.retrieval.embedder import BaseEmbedder, MultilingualE5Embedder
from src.retrieval.index import FAISSIndexManager
from src.retrieval.retriever import Retriever


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for fast unit tests without downloading models."""

    def __init__(self, dim: int = 16):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_queries(self, queries):
        if not queries:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = []
        for q in queries:
            v = np.zeros(self._dim, dtype=np.float32)
            idx = hash(q) % self._dim
            v[idx] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def embed_passages(self, passages, batch_size=64):
        if not passages:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = []
        for p in passages:
            v = np.zeros(self._dim, dtype=np.float32)
            idx = hash(p) % self._dim
            v[idx] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


def test_faiss_index_manager_add_and_search():
    dim = 8
    index_mgr = FAISSIndexManager(dimension=dim, index_type="FlatIP")
    assert index_mgr.total_vectors == 0

    chunks = [
        Chunk(chunk_id="c0", doc_id="d0", text="भारत की राजधानी नई दिल्ली है।"),
        Chunk(chunk_id="c1", doc_id="d1", text="सौर ऊर्जा के लाभ।"),
    ]
    # Unit orthogonal vectors
    embeddings = np.zeros((2, dim), dtype=np.float32)
    embeddings[0, 0] = 1.0  # Vector for c0
    embeddings[1, 1] = 1.0  # Vector for c1

    index_mgr.add_chunks(chunks, embeddings)
    assert index_mgr.total_vectors == 2

    # Query matching c0 exactly
    query_vec = np.zeros((1, dim), dtype=np.float32)
    query_vec[0, 0] = 1.0

    results = index_mgr.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].passage_id == "c0"
    assert pytest.approx(results[0].score, 0.001) == 1.0
    assert results[0].rank == 1
    assert results[1].passage_id == "c1"
    assert pytest.approx(results[1].score, 0.001) == 0.0


def test_faiss_index_save_and_load():
    dim = 4
    index_mgr = FAISSIndexManager(dimension=dim, index_type="FlatIP")
    chunks = [Chunk(chunk_id="c_test", doc_id="d_test", text="परीक्षण पाठ")]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    index_mgr.add_chunks(chunks, embeddings)

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = f"{tmpdir}/index.bin"
        meta_path = f"{tmpdir}/meta.json"

        index_mgr.save(idx_path, meta_path)

        new_mgr = FAISSIndexManager(dimension=dim, index_type="FlatIP")
        new_mgr.load(idx_path, meta_path)

        assert new_mgr.total_vectors == 1
        query_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        res = new_mgr.search(query_vec, top_k=1)
        assert len(res) == 1
        assert res[0].passage_id == "c_test"


def test_retriever_coordination():
    embedder = MockEmbedder(dim=8)
    index_mgr = FAISSIndexManager(dimension=8, index_type="FlatIP")
    retriever = Retriever(embedder=embedder, index_manager=index_mgr, top_k=3)

    # Search on empty index
    res, embed_ms, search_ms = retriever.retrieve("परीक्षण")
    assert res.is_empty is True
    assert embed_ms >= 0.0
    assert search_ms >= 0.0

    # Search on populated index
    chunks = [Chunk(chunk_id="p1", doc_id="d1", text="परीक्षण")]
    emb = embedder.embed_passages(["परीक्षण"])
    index_mgr.add_chunks(chunks, emb)

    res, embed_ms, search_ms = retriever.retrieve("परीक्षण")
    assert res.is_empty is False
    assert len(res.passages) == 1
    assert res.top_score >= 0.99


def test_embedder_caching_and_lru():
    mock_model_instance = MagicMock()
    mock_model_instance.get_embedding_dimension.return_value = 8
    mock_model_instance.get_sentence_embedding_dimension.return_value = 8
    encode_call_count = [0]

    def fake_encode(texts, **kwargs):
        encode_call_count[0] += len(texts)
        return np.ones((len(texts), 8), dtype=np.float32)

    mock_model_instance.encode.side_effect = fake_encode

    embedder = MultilingualE5Embedder(
        model_name_or_path="test-model",
        device="cpu",
        max_cache_size=3,
        model_instance=mock_model_instance,
    )

    # 1. First query call -> encodes 1
    q1 = embedder.embed_queries(["भारत"])
    assert q1.shape == (1, 8)
    assert encode_call_count[0] == 1

    # 2. Second query call with same query -> hits cache, encode count remains 1
    q1_again = embedder.embed_queries(["भारत"])
    assert np.allclose(q1, q1_again)
    assert encode_call_count[0] == 1

    # 3. New query -> encodes 1 more
    q2 = embedder.embed_queries(["दिल्ली"])
    assert encode_call_count[0] == 2

    # 4. Clear cache and re-query -> re-encodes
    embedder.clear_cache()
    q1_re = embedder.embed_queries(["भारत"])
    assert encode_call_count[0] == 3
