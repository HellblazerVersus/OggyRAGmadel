"""End-to-end tests for the complete Voice-Enabled RAG pipeline."""

import tempfile
import numpy as np
import pytest
from src.generation.generator import MockGenerator
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.safety import CompositeGuardrail
from src.pipeline.harness import RobustExecutionHarness, CircuitBreakerOpenException
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.schemas import AudioInputRequest, Chunk, TextInputRequest
from src.retrieval.index import FAISSIndexManager
from src.retrieval.retriever import Retriever
from src.stt.transcriber import MockTranscriber
from tests.test_retriever import MockEmbedder


@pytest.fixture
def populated_pipeline():
    dim = 8
    embedder = MockEmbedder(dim=dim)
    index_mgr = FAISSIndexManager(dimension=dim, index_type="FlatIP")

    # Add knowledge base chunk
    chunks = [
        Chunk(
            chunk_id="kb_india",
            doc_id="d1",
            text="भारत की राजधानी नई दिल्ली है।",
        )
    ]
    # In MockEmbedder, text determines embedding
    emb = embedder.embed_passages([c.text for c in chunks])
    index_mgr.add_chunks(chunks, emb)

    retriever = Retriever(embedder=embedder, index_manager=index_mgr, top_k=3)
    guardrail = ConfidenceGuardrail(min_confidence_threshold=0.5)
    generator = MockGenerator(model_name="mock-test-llm")
    transcriber = MockTranscriber(fixed_text="भारत की राजधानी नई दिल्ली है।")

    pipeline = RAGPipeline(
        transcriber=transcriber,
        retriever=retriever,
        guardrail=guardrail,
        generator=generator,
        harness=RobustExecutionHarness(),
        default_language="hi",
    )
    return pipeline


def test_e2e_voice_pipeline_success(populated_pipeline):
    audio_req = AudioInputRequest(audio_path="dummy_path.wav", language="hi")
    resp = populated_pipeline.process_voice(audio_req)

    assert resp.query == "भारत की राजधानी नई दिल्ली है।"
    assert resp.is_refusal is False
    assert "नई दिल्ली" in resp.answer
    assert resp.confidence_score >= 0.5
    assert len(resp.retrieved_passages) == 1
    assert resp.latencies.stt_ms >= 0.0
    assert resp.latencies.retrieval_leg_total_ms >= 0.0
    assert resp.latencies.total_pipeline_ms >= 0.0


def test_e2e_text_pipeline_success(populated_pipeline):
    text_req = TextInputRequest(query="भारत की राजधानी नई दिल्ली है।", language="hi")
    resp = populated_pipeline.process_text(text_req)

    assert resp.is_refusal is False
    assert "नई दिल्ली" in resp.answer
    assert resp.latencies.stt_ms == 0.0  # STT bypassed for text


def test_e2e_guardrail_abstain(populated_pipeline):
    # Set impossible threshold to force refusal
    populated_pipeline.guardrail.min_confidence_threshold = 2.0
    text_req = TextInputRequest(query="अनजान प्रश्न", language="hi")
    resp = populated_pipeline.process_text(text_req)

    assert resp.is_refusal is True
    assert "पर्याप्त जानकारी उपलब्ध नहीं है" in resp.answer
    assert resp.latencies.generation_ms == 0.0  # Generator skipped on refusal


def test_e2e_composite_guardrail_safety_refusal(populated_pipeline):
    # Replace guardrail with CompositeGuardrail
    composite = CompositeGuardrail(confidence_threshold=0.5)
    populated_pipeline.guardrail = composite

    # Unsafe query should be blocked pre-retrieval
    text_req = TextInputRequest(query="मुझे बम बनाने की विधि बताओ", language="hi")
    resp = populated_pipeline.process_text(text_req)

    assert resp.is_refusal is True
    assert "सुरक्षा दिशानिर्देशों" in resp.answer
    assert len(resp.retrieved_passages) == 0


def test_harness_retry_and_circuit_breaker():
    harness = RobustExecutionHarness(
        max_retries=2,
        backoff_factor=1.1,
        enable_circuit_breaker=True,
        circuit_failure_threshold=2,
    )

    call_count = [0]

    def failing_func():
        call_count[0] += 1
        raise ValueError("Simulated network blip")

    # Should attempt 3 times (1 initial + 2 retries) then fail
    with pytest.raises(ValueError):
        harness.execute_with_retry(failing_func, max_retries=2)

    assert call_count[0] == 3
