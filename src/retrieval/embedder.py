"""Embedding model wrappers for multilingual retrieval."""

from abc import ABC, abstractmethod
from collections import OrderedDict
import threading
from typing import List, Optional, Union
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from src.utils.logging import logger


class BaseEmbedder(ABC):
    """Abstract interface for dense retrieval embedding models."""

    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed a list of search queries. Returns float32 numpy array of shape (N, D)."""
        pass

    @abstractmethod
    def embed_passages(self, passages: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of text passages. Returns float32 numpy array of shape (N, D)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension D."""
        pass


class MultilingualE5Embedder(BaseEmbedder):
    """Wrapper for intfloat/multilingual-e5-base dense retriever model.
    
    Adheres to E5 specification by prepending 'query: ' and 'passage: ' prefixes,
    and returns L2-normalized embeddings for fast cosine similarity via inner product.
    Includes in-memory LRU query caching and FP16 tensor acceleration.
    """

    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        max_cache_size: int = 4096,
        warmup: bool = False,
        model_instance: Optional[object] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.max_cache_size = max_cache_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        logger.info(f"Initializing MultilingualE5Embedder ({self.model_name_or_path}) on {self.device}")

        if model_instance is not None:
            self.model = model_instance
        else:
            from sentence_transformers import SentenceTransformer
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            try:
                self.model = SentenceTransformer(
                    self.model_name_or_path,
                    device=self.device,
                    model_kwargs=model_kwargs if model_kwargs else None,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize with model_kwargs ({e}). Retrying without model_kwargs...")
                self.model = SentenceTransformer(
                    self.model_name_or_path,
                    device=self.device,
                )

        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = int(self.model.get_embedding_dimension())
        else:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Prime PyTorch / CUDA execution graph if requested
        if warmup:
            try:
                with torch.inference_mode():
                    self.model.encode(["query: warmup"], show_progress_bar=False, normalize_embeddings=self.normalize_embeddings)
            except Exception:
                pass

    @property
    def dimension(self) -> int:
        return self._dim

    def clear_cache(self) -> None:
        """Clears the internal query embedding LRU cache."""
        with self._cache_lock:
            self._query_cache.clear()

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed queries prepending 'query: ' prefix with LRU caching."""
        if not queries:
            return np.empty((0, self._dim), dtype=np.float32)

        results = [None] * len(queries)
        uncached_indices = []
        uncached_texts = []

        with self._cache_lock:
            for idx, q in enumerate(queries):
                cleaned_q = q.strip()
                if cleaned_q in self._query_cache:
                    # Move to end for LRU policy
                    self._query_cache.move_to_end(cleaned_q)
                    results[idx] = self._query_cache[cleaned_q]
                else:
                    uncached_indices.append(idx)
                    prefixed = f"query: {cleaned_q}" if not cleaned_q.startswith("query:") else cleaned_q
                    uncached_texts.append(prefixed)

        if uncached_texts:
            with torch.inference_mode():
                new_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=len(uncached_texts),
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                ).astype(np.float32)

            with self._cache_lock:
                for idx_pos, orig_idx in enumerate(uncached_indices):
                    vec = new_embeddings[idx_pos]
                    results[orig_idx] = vec
                    cleaned_q = queries[orig_idx].strip()
                    self._query_cache[cleaned_q] = vec
                    if len(self._query_cache) > self.max_cache_size:
                        self._query_cache.popitem(last=False)

        return np.array(results, dtype=np.float32)

    def embed_passages(self, passages: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed passages prepending 'passage: ' prefix."""
        if not passages:
            return np.empty((0, self._dim), dtype=np.float32)

        prefixed_passages = [
            f"passage: {p.strip()}" if not p.strip().startswith("passage:") else p.strip()
            for p in passages
        ]

        with torch.inference_mode():
            embeddings = self.model.encode(
                prefixed_passages,
                batch_size=batch_size,
                show_progress_bar=len(passages) > 100,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
        return embeddings.astype(np.float32)


def get_embedder(
    model_name: str = "intfloat/multilingual-e5-base",
    device: Optional[str] = None,
) -> BaseEmbedder:
    """Factory helper to obtain an embedder instance."""
    return MultilingualE5Embedder(model_name_or_path=model_name, device=device)
