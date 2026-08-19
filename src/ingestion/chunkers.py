"""Engineered chunking strategies for Indic and multilingual text corpora.

Strategies:
1. FixedWindowChunker - Word-based sliding window with overlap
2. SentenceBoundaryChunker - Sentence-aware Indic/Latin punctuation chunker
3. SemanticChunker - Groups sentences by character n-gram similarity
4. RecursiveChunker - Multi-level separator splitting (paragraphs → sentences → words)
5. MetadataAwareChunker - Preserves passage metadata as chunk prefix context
6. HybridChunker - Multi-strategy with deduplication
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.pipeline.schemas import Chunk, RawPassage


class BaseChunker(ABC):
    """Abstract base class establishing the contract for all chunking strategies."""

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split a raw text string into structured Chunk objects."""
        pass

    def chunk_passages(self, passages: List[RawPassage]) -> List[Chunk]:
        """Batch-chunks a collection of RawPassage objects."""
        all_chunks: List[Chunk] = []
        for passage in passages:
            chunks = self.chunk_text(
                text=passage.text,
                doc_id=passage.passage_id,
                metadata=passage.metadata,
            )
            all_chunks.extend(chunks)
        return all_chunks


class FixedWindowChunker(BaseChunker):
    """Fixed-size word/token window chunker with configurable sliding overlap."""

    def __init__(self, window_size: int = 128, overlap: int = 32):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than window_size ({window_size})"
            )
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        text = text.strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        # If document is shorter than window, return single chunk
        if len(words) <= self.window_size:
            return [
                Chunk(
                    chunk_id=f"{doc_id}_c0",
                    doc_id=doc_id,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    token_count=len(words),
                    metadata={**meta, "strategy": "fixed_window", "chunk_index": 0},
                )
            ]

        chunks: List[Chunk] = []
        chunk_idx = 0

        for start_idx in range(0, len(words), self.stride):
            end_idx = min(start_idx + self.window_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_str = " ".join(chunk_words)

            # Approximate char offsets
            char_start = text.find(chunk_words[0]) if chunk_words else 0
            char_end = char_start + len(chunk_str)

            chunk = Chunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                text=chunk_str,
                char_start=max(0, char_start),
                char_end=min(len(text), char_end),
                token_count=len(chunk_words),
                metadata={
                    **meta,
                    "strategy": "fixed_window",
                    "chunk_index": chunk_idx,
                    "window_size": self.window_size,
                    "overlap": self.overlap,
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_idx >= len(words):
                break

        return chunks


class SentenceBoundaryChunker(BaseChunker):
    """Sentence-boundary chunker supporting Devanagari (danda '।', '॥') and Latin punctuation.
    
    Packs whole sentences into chunks up to max_tokens with sentence-level overlap,
    ensuring semantic continuity and avoiding mid-sentence cuts.
    """

    # Regex matching Hindi/Devanagari danda, double danda, period, exclamation, question mark, or newlines
    SENTENCE_SPLIT_REGEX = re.compile(r"([^।॥\.\?\!\n]+[।॥\.\?\!\n]*)")

    def __init__(self, max_tokens: int = 150, overlap_sentences: int = 1):
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if overlap_sentences < 0:
            raise ValueError(f"overlap_sentences cannot be negative, got {overlap_sentences}")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences respecting Indic and standard punctuation."""
        matches = self.SENTENCE_SPLIT_REGEX.findall(text)
        sentences = [s.strip() for s in matches if s.strip()]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 0
        i = 0

        while i < len(sentences):
            current_sentences: List[str] = []
            current_token_count = 0
            j = i

            while j < len(sentences):
                sent = sentences[j]
                sent_tokens = len(sent.split())
                if current_sentences and (current_token_count + sent_tokens > self.max_tokens):
                    break
                current_sentences.append(sent)
                current_token_count += sent_tokens
                j += 1

            chunk_str = " ".join(current_sentences)
            char_start = text.find(current_sentences[0]) if current_sentences else 0
            char_end = char_start + len(chunk_str)

            chunk = Chunk(
                chunk_id=f"{doc_id}_s{chunk_idx}",
                doc_id=doc_id,
                text=chunk_str,
                char_start=max(0, char_start),
                char_end=min(len(text), char_end),
                token_count=current_token_count,
                metadata={
                    **meta,
                    "strategy": "sentence_boundary",
                    "chunk_index": chunk_idx,
                    "num_sentences": len(current_sentences),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Advance by step size (j - overlap)
            if j >= len(sentences):
                break
            step = max(1, (j - i) - self.overlap_sentences)
            i += step

        return chunks


class SemanticChunker(BaseChunker):
    """Groups sentences by semantic similarity using character n-gram overlap (Jaccard).
    
    Adjacent sentences with high character n-gram overlap are grouped together.
    When similarity drops below a threshold, a new chunk boundary is created.
    This avoids needing an embedding model at chunking time.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        ngram_size: int = 3,
        max_tokens: int = 200,
    ):
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.max_tokens = max_tokens
        self._sentence_splitter = SentenceBoundaryChunker(max_tokens=max_tokens, overlap_sentences=0)

    def _get_ngrams(self, text: str) -> set:
        """Extracts character n-grams from text."""
        text = text.lower()
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Computes Jaccard similarity between character n-gram sets of two strings."""
        set1 = self._get_ngrams(s1)
        set2 = self._get_ngrams(s2)
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        text = text.strip()
        if not text:
            return []

        sentences = self._sentence_splitter._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_group: List[str] = [sentences[0]]
        current_tokens = len(sentences[0].split())
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_tokens = len(sent.split())
            sim = self._jaccard_similarity(sentences[i - 1], sent)

            if sim < self.threshold or (current_tokens + sent_tokens > self.max_tokens):
                # Emit current group as a chunk
                chunk_str = " ".join(current_group)
                char_start = text.find(current_group[0]) if current_group else 0
                char_end = char_start + len(chunk_str)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_sem{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_str,
                    char_start=max(0, char_start),
                    char_end=min(len(text), char_end),
                    token_count=current_tokens,
                    metadata={
                        **meta,
                        "strategy": "semantic",
                        "chunk_index": chunk_idx,
                        "num_sentences": len(current_group),
                    },
                ))
                chunk_idx += 1
                current_group = [sent]
                current_tokens = sent_tokens
            else:
                current_group.append(sent)
                current_tokens += sent_tokens

        # Emit final group
        if current_group:
            chunk_str = " ".join(current_group)
            char_start = text.find(current_group[0]) if current_group else 0
            char_end = char_start + len(chunk_str)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_sem{chunk_idx}",
                doc_id=doc_id,
                text=chunk_str,
                char_start=max(0, char_start),
                char_end=min(len(text), char_end),
                token_count=current_tokens,
                metadata={
                    **meta,
                    "strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "num_sentences": len(current_group),
                },
            ))

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively splits text using multiple separator levels.
    
    Tries separators in order: paragraphs (\\n\\n) → line breaks (\\n) →
    Devanagari danda (।) → period (.) → space → character-level.
    Similar to LangChain's RecursiveCharacterTextSplitter approach.
    """

    def __init__(self, max_tokens: int = 128, overlap: int = 20):
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "।", ".", " "]

    def _split_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text by trying separators in order until each piece fits."""
        if not text.strip():
            return []
        if len(text.split()) <= self.max_tokens:
            return [text]
        if not separators:
            return [text]

        sep = separators[0]
        parts = text.split(sep)
        result: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Re-append separator if it's a sentence ender
            if sep in ("।", "."):
                part = part + sep
            if len(part.split()) <= self.max_tokens:
                result.append(part)
            else:
                result.extend(self._split_recursively(part, separators[1:]))

        return result

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        text = text.strip()
        if not text:
            return []

        splits = self._split_recursively(text, self.separators)
        if not splits:
            return []

        # Merge small splits into chunks respecting max_tokens
        chunks: List[Chunk] = []
        chunk_idx = 0
        i = 0

        while i < len(splits):
            current_parts: List[str] = []
            current_tokens = 0
            j = i

            while j < len(splits):
                part_tokens = len(splits[j].split())
                if current_parts and current_tokens + part_tokens > self.max_tokens:
                    break
                current_parts.append(splits[j])
                current_tokens += part_tokens
                j += 1

            chunk_str = " ".join(current_parts).strip()
            if chunk_str:
                char_start = text.find(chunk_str[:50])
                char_start = max(0, char_start) if char_start >= 0 else 0
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_rec{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_str,
                    char_start=char_start,
                    char_end=min(len(text), char_start + len(chunk_str)),
                    token_count=len(chunk_str.split()),
                    metadata={
                        **meta,
                        "strategy": "recursive",
                        "chunk_index": chunk_idx,
                    },
                ))
                chunk_idx += 1

            # Advance with overlap
            if j <= i:
                i += 1
            else:
                # Calculate overlap step-back
                overlap_tokens = 0
                step_back = 0
                while j - 1 - step_back > i and overlap_tokens < self.overlap:
                    overlap_tokens += len(splits[j - 1 - step_back].split())
                    step_back += 1
                i = max(i + 1, j - step_back)

        return chunks


class MetadataAwareChunker(BaseChunker):
    """Preserves passage-level metadata by injecting provenance context as prefix.
    
    Wraps any base chunker and prepends metadata (passage_id, language, source)
    to each chunk, helping the retriever understand chunk provenance during search.
    """

    def __init__(self, base_chunker: Optional[BaseChunker] = None):
        self.base_chunker = base_chunker or SentenceBoundaryChunker(max_tokens=150)

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}
        chunks = self.base_chunker.chunk_text(text, doc_id, meta)

        # Build metadata prefix
        prefix_parts = []
        if doc_id:
            prefix_parts.append(f"Source: {doc_id}")
        if "language" in meta:
            prefix_parts.append(f"Language: {meta['language']}")
        if "dataset" in meta:
            prefix_parts.append(f"Dataset: {meta['dataset']}")

        prefix = " | ".join(prefix_parts)
        if prefix:
            prefix += "\n"

        for chunk in chunks:
            chunk.text = prefix + chunk.text
            chunk.token_count += len(prefix.split())
            chunk.metadata["strategy"] = "metadata_aware"
            chunk.metadata["has_metadata_prefix"] = True

        return chunks


class HybridChunker(BaseChunker):
    """Applies multiple chunking strategies and deduplicates for maximum coverage.
    
    Runs SentenceBoundary + FixedWindow strategies, then deduplicates
    by exact text match to keep unique chunks with the best coverage
    of the source document.
    """

    def __init__(self, max_tokens: int = 150, window_size: int = 128, overlap: int = 32):
        self.sentence_chunker = SentenceBoundaryChunker(max_tokens=max_tokens)
        self.fixed_chunker = FixedWindowChunker(window_size=window_size, overlap=overlap)

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = metadata or {}

        # Run both strategies
        sent_chunks = self.sentence_chunker.chunk_text(text, doc_id, meta)
        fixed_chunks = self.fixed_chunker.chunk_text(text, doc_id, meta)

        # Deduplicate by text content
        seen_texts: set = set()
        final_chunks: List[Chunk] = []

        # Prefer sentence-boundary chunks first (higher quality)
        for chunk in sent_chunks + fixed_chunks:
            normalized = chunk.text.strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                final_chunks.append(chunk)

        # Re-index
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_id = f"{doc_id}_hyb{i}"
            chunk.metadata["strategy"] = "hybrid"
            chunk.metadata["chunk_index"] = i

        return final_chunks


def get_chunker(strategy: str = "sentence", **kwargs) -> BaseChunker:
    """Factory function to instantiate chunkers by strategy name.
    
    Supported strategies: 'fixed', 'sentence', 'semantic', 'recursive',
    'metadata_aware', 'hybrid'
    """
    strat = strategy.lower()
    if strat in ("fixed", "fixed_window"):
        return FixedWindowChunker(
            window_size=kwargs.get("window_size", 128),
            overlap=kwargs.get("overlap", 32),
        )
    elif strat in ("sentence", "sentence_boundary"):
        return SentenceBoundaryChunker(
            max_tokens=kwargs.get("max_tokens", 150),
            overlap_sentences=kwargs.get("overlap_sentences", 1),
        )
    elif strat == "semantic":
        return SemanticChunker(
            threshold=kwargs.get("threshold", 0.1),
            ngram_size=kwargs.get("ngram_size", 3),
            max_tokens=kwargs.get("max_tokens", 200),
        )
    elif strat == "recursive":
        return RecursiveChunker(
            max_tokens=kwargs.get("max_tokens", 128),
            overlap=kwargs.get("overlap", 20),
        )
    elif strat in ("metadata_aware", "metadata"):
        base_strategy = kwargs.pop("base_strategy", "sentence")
        base_chunker = get_chunker(base_strategy, **kwargs)
        return MetadataAwareChunker(base_chunker=base_chunker)
    elif strat == "hybrid":
        return HybridChunker(
            max_tokens=kwargs.get("max_tokens", 150),
            window_size=kwargs.get("window_size", 128),
            overlap=kwargs.get("overlap", 32),
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Supported: 'fixed', 'sentence', 'semantic', 'recursive', 'metadata_aware', 'hybrid'"
        )
