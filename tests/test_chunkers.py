"""Unit tests for all 6 chunking strategies."""

import pytest
from src.ingestion.chunkers import (
    BaseChunker,
    FixedWindowChunker,
    HybridChunker,
    MetadataAwareChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceBoundaryChunker,
    get_chunker,
)
from src.pipeline.schemas import Chunk, RawPassage


def test_fixed_window_chunker_basic():
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    chunker = FixedWindowChunker(window_size=5, overlap=2)
    chunks = chunker.chunk_text(text, doc_id="doc_1")
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "doc_1_c0"
    assert chunks[0].text == "word1 word2 word3 word4 word5"
    assert chunks[1].text == "word4 word5 word6 word7 word8"
    assert chunks[2].text == "word7 word8 word9 word10"
    assert chunks[0].metadata["strategy"] == "fixed_window"


def test_fixed_window_chunker_short_text():
    text = "छोटा वाक्य"
    chunker = FixedWindowChunker(window_size=10, overlap=2)
    chunks = chunker.chunk_text(text, doc_id="doc_short")
    assert len(chunks) == 1
    assert chunks[0].text == "छोटा वाक्य"
    assert chunks[0].token_count == 2


def test_fixed_window_invalid_params():
    with pytest.raises(ValueError):
        FixedWindowChunker(window_size=10, overlap=10)
    with pytest.raises(ValueError):
        FixedWindowChunker(window_size=0, overlap=0)


def test_sentence_boundary_chunker_hindi_danda():
    text = "भारत एक विशाल देश है। इसकी राजधानी नई दिल्ली है। यहाँ कई भाषाएँ बोली जाती हैं।"
    chunker = SentenceBoundaryChunker(max_tokens=10, overlap_sentences=1)
    chunks = chunker.chunk_text(text, doc_id="doc_hi")
    assert len(chunks) >= 2
    for c in chunks:
        assert "।" in c.text
        assert c.metadata["strategy"] == "sentence_boundary"


def test_sentence_boundary_chunker_multilingual():
    text = "Sentence one. Sentence two! Sentence three? Sentence four."
    chunker = SentenceBoundaryChunker(max_tokens=5, overlap_sentences=1)
    chunks = chunker.chunk_text(text, doc_id="doc_en")
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc_en_s0"


def test_semantic_chunker():
    text = "सौर ऊर्जा सूर्य से मिलती है। सोलर पैनल बिजली बनाते हैं। क्रिकेट भारत में लोकप्रिय खेल है।"
    chunker = SemanticChunker(threshold=0.1, ngram_size=3, max_tokens=20)
    chunks = chunker.chunk_text(text, doc_id="doc_sem")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["strategy"] == "semantic"


def test_recursive_chunker():
    text = "पहला पैराग्राफ यहाँ है। इसमें कुछ जानकारी है।\n\nदूसरा पैराग्राफ यहाँ है। इसमें अन्य विवरण हैं।"
    chunker = RecursiveChunker(max_tokens=10, overlap=2)
    chunks = chunker.chunk_text(text, doc_id="doc_rec")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["strategy"] == "recursive"


def test_metadata_aware_chunker():
    text = "यह एक परीक्षण वाक्य है।"
    chunker = MetadataAwareChunker(base_chunker=SentenceBoundaryChunker(max_tokens=50))
    chunks = chunker.chunk_text(text, doc_id="doc_meta_1", metadata={"language": "hi", "dataset": "msmarco"})
    assert len(chunks) == 1
    assert "Source: doc_meta_1" in chunks[0].text
    assert "Language: hi" in chunks[0].text
    assert chunks[0].metadata["strategy"] == "metadata_aware"


def test_hybrid_chunker():
    text = "वाक्य एक। वाक्य दो। वाक्य तीन। वाक्य चार।"
    chunker = HybridChunker(max_tokens=10)
    chunks = chunker.chunk_text(text, doc_id="doc_hyb")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["strategy"] == "hybrid"


def test_chunker_empty_input():
    fixed = FixedWindowChunker(window_size=10, overlap=2)
    sentence = SentenceBoundaryChunker(max_tokens=10, overlap_sentences=1)
    semantic = SemanticChunker()
    recursive = RecursiveChunker()

    assert fixed.chunk_text("") == []
    assert fixed.chunk_text("   ") == []
    assert sentence.chunk_text("") == []
    assert semantic.chunk_text("") == []
    assert recursive.chunk_text("") == []


def test_chunk_passages_batch():
    passages = [
        RawPassage(passage_id="p1", text="पहला वाक्य। दूसरा वाक्य।", language="hi"),
        RawPassage(passage_id="p2", text="तीसरा वाक्य। चौथा वाक्य।", language="hi"),
    ]
    chunker = get_chunker("sentence", max_tokens=100)
    chunks = chunker.chunk_passages(passages)
    assert len(chunks) == 2
    assert chunks[0].doc_id == "p1"
    assert chunks[1].doc_id == "p2"


def test_get_chunker_factory():
    assert isinstance(get_chunker("fixed", window_size=50, overlap=10), FixedWindowChunker)
    assert isinstance(get_chunker("sentence", max_tokens=50), SentenceBoundaryChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    assert isinstance(get_chunker("metadata_aware"), MetadataAwareChunker)
    assert isinstance(get_chunker("hybrid"), HybridChunker)

    with pytest.raises(ValueError):
        get_chunker("unknown_strategy")
