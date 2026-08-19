"""Retriever coordinating query embedding and in-process vector search."""

import time
from typing import List, Optional, Tuple
import numpy as np
from src.pipeline.schemas import RetrievalResult, RetrievedPassage
from src.retrieval.embedder import BaseEmbedder
from src.retrieval.index import FAISSIndexManager
from src.utils.logging import logger


class Retriever:
    """End-to-end vector retriever combining embedding generation and FAISS search."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        index_manager: FAISSIndexManager,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.index_manager = index_manager
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[RetrievalResult, float, float]:
        """Retrieve relevant passages for a query.
        
        Returns:
            Tuple of (RetrievalResult, embed_time_ms, search_time_ms)
        """
        k = top_k or self.top_k
        query_cleaned = query.strip()

        if not query_cleaned:
            return (
                RetrievalResult(
                    query=query,
                    passages=[],
                    top_score=0.0,
                    is_empty=True,
                ),
                0.0,
                0.0,
            )

        # 1. Embed Query
        t0 = time.perf_counter_ns()
        query_vec = self.embedder.embed_queries([query_cleaned])
        embed_time_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # 2. FAISS Vector Search
        t1 = time.perf_counter_ns()
        passages: List[RetrievedPassage] = self.index_manager.search(query_vec, top_k=k)
        search_time_ms = (time.perf_counter_ns() - t1) / 1_000_000.0

        top_score = passages[0].score if passages else 0.0
        is_empty = len(passages) == 0

        result = RetrievalResult(
            query=query,
            passages=passages,
            top_score=top_score,
            is_empty=is_empty,
        )

        return result, embed_time_ms, search_time_ms
