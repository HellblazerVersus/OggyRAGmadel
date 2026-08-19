"""In-process FAISS Vector Index Manager for low-latency similarity search."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
from src.pipeline.schemas import Chunk, RetrievedPassage
from src.utils.logging import logger


class FAISSIndexManager:
    """Manages an in-process local FAISS index with passage metadata tracking."""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "FlatIP",
        hnsw_m: int = 32,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.index: Optional[faiss.Index] = None
        self.metadata_store: List[Dict[str, Any]] = []
        self._init_index()

    def _init_index(self) -> None:
        """Initializes empty FAISS index."""
        if self.index_type in ("FlatIP", "flat_ip", "cosine"):
            # Inner product index for L2-normalized embeddings (equivalent to cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type in ("HNSW", "HNSWFlat", "hnsw"):
            # Fast Approximate Nearest Neighbor graph index
            self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            logger.warning(f"Unrecognized index_type {self.index_type}, falling back to IndexFlatIP")
            self.index = faiss.IndexFlatIP(self.dimension)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Adds text chunks and their corresponding embeddings into the FAISS index."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if len(chunks) == 0:
            return

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        for chunk in chunks:
            self.metadata_store.append(chunk.model_dump())

        logger.info(f"Added {len(chunks)} vectors to index. Total indexed: {self.total_vectors}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievedPassage]:
        """Search top-K nearest neighbors for a query embedding."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty FAISS index.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results: List[RetrievedPassage] = []
        for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1 or idx >= len(self.metadata_store):
                continue
            meta = self.metadata_store[idx]
            passage = RetrievedPassage(
                passage_id=meta.get("chunk_id", f"idx_{idx}"),
                text=meta.get("text", ""),
                score=float(score),
                rank=rank_idx,
                metadata=meta,
            )
            results.append(passage)

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist FAISS index binary and metadata JSON to disk."""
        idx_path = Path(index_path)
        meta_path = Path(metadata_path)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving FAISS index to {idx_path} and metadata to {meta_path}")
        faiss.write_index(self.index, str(idx_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load persisted FAISS index binary and metadata from disk."""
        idx_path = Path(index_path)
        meta_path = Path(metadata_path)

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index or metadata file not found at {idx_path} or {meta_path}"
            )

        logger.info(f"Loading FAISS index from {idx_path}...")
        self.index = faiss.read_index(str(idx_path))
        self.dimension = self.index.d

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)

        logger.info(
            f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.metadata_store)} metadata entries."
        )
