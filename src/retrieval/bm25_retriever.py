"""Sparse Retriever using BM25 for low-memory environments."""

import json
import time
from typing import List, Optional, Tuple
from pathlib import Path
from rank_bm25 import BM25Okapi

from src.pipeline.schemas import RetrievalResult, RetrievedPassage
from src.utils.logging import logger


class BM25Retriever:
    """Sparse retriever using BM25 keyword matching."""

    def __init__(self, metadata_path: str = "data/processed/passage_metadata.json", top_k: int = 5):
        self.top_k = top_k
        self.passages = []
        self.bm25 = None
        
        logger.info(f"Loading BM25 index from {metadata_path}...")
        self._load_passages(metadata_path)
        self._build_index()

    def _load_passages(self, metadata_path: str):
        path = Path(metadata_path)
        if not path.exists():
            logger.warning(f"Metadata file {metadata_path} not found.")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.passages.append(json.loads(line))
                    except Exception:
                        pass
        logger.info(f"Loaded {len(self.passages)} passages for BM25.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for Hindi/English (split by space and punctuation)."""
        import re
        # Split on whitespace and common punctuation including Devanagari danda
        tokens = re.split(r'[\s,.!?।॥\(\)\[\]"\'\-]+', text)
        return [t.lower() for t in tokens if t.strip()]

    def _build_index(self):
        if not self.passages:
            logger.warning("No passages to build BM25 index.")
            self.bm25 = None
            return

        logger.info("Tokenizing corpus for BM25...")
        tokenized_corpus = [self._tokenize(p.get("text", "")) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[RetrievalResult, float, float]:
        """Retrieve relevant passages for a query using BM25."""
        k = top_k or self.top_k
        query_cleaned = query.strip()

        if not query_cleaned or not self.bm25:
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

        # 1. Embed Query (Tokenization for BM25)
        t0 = time.perf_counter_ns()
        tokenized_query = self._tokenize(query_cleaned)
        embed_time_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # 2. Vector Search (BM25 scoring)
        t1 = time.perf_counter_ns()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
        
        passages: List[RetrievedPassage] = []
        for rank, idx in enumerate(top_indices):
            score = float(doc_scores[idx])
            # Normalize BM25 score roughly to 0-1 for compatibility with guardrails (BM25 scores can be >1)
            # A simple heuristic: clamp to 1.0 or divide by a max expected value
            normalized_score = min(score / 15.0, 1.0) if score > 0 else 0.0
            
            p = self.passages[idx]
            passages.append(
                RetrievedPassage(
                    passage_id=p.get("passage_id", str(idx)),
                    text=p.get("text", ""),
                    score=normalized_score,
                    rank=rank + 1,
                    metadata=p.get("metadata", {}),
                )
            )
            
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
