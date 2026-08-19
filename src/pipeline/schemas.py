"""Pydantic schemas and typed data models for the Voice-Enabled RAG pipeline."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a discrete text chunk produced by a chunking strategy."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Text content of the chunk")
    char_start: int = Field(default=0, description="Starting character index in parent doc")
    char_end: int = Field(default=0, description="Ending character index in parent doc")
    token_count: int = Field(default=0, description="Approximate or exact token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary document metadata")


class RawPassage(BaseModel):
    """Raw passage loaded from dataset."""
    passage_id: str = Field(..., description="Unique passage ID from MSMARCO")
    text: str = Field(..., description="Passage body text")
    language: str = Field(default="hi", description="ISO language code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class AudioInputRequest(BaseModel):
    """Request payload containing voice input."""
    audio_path: Optional[str] = Field(None, description="Path to audio file on disk")
    audio_bytes: Optional[bytes] = Field(None, description="Raw audio bytes in WAV/MP3 format")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    language: Optional[str] = Field(default="hi", description="Expected spoken language hint")


class TextInputRequest(BaseModel):
    """Request payload containing direct text query."""
    query: str = Field(..., description="Text query")
    language: Optional[str] = Field(default="hi", description="Query language code")


class STTResult(BaseModel):
    """Result of the Speech-To-Text transcription stage."""
    transcribed_text: str = Field(..., description="Recognized speech text")
    detected_language: str = Field(default="hi", description="Detected or specified language")
    duration_seconds: float = Field(default=0.0, description="Audio duration in seconds")
    avg_logprob: Optional[float] = Field(default=None, description="Model confidence log-probability")


class RetrievedPassage(BaseModel):
    """A single retrieved passage with vector similarity score."""
    passage_id: str = Field(..., description="Identifier of the retrieved passage/chunk")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Similarity score (cosine similarity / inner product)")
    rank: int = Field(..., description="Rank in retrieval results (1-indexed)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")


class RetrievalResult(BaseModel):
    """Aggregated retrieval stage results."""
    query: str = Field(..., description="Query used for vector search")
    passages: List[RetrievedPassage] = Field(default_factory=list, description="Top-K retrieved passages")
    top_score: float = Field(default=0.0, description="Highest similarity score among retrieved passages")
    is_empty: bool = Field(default=False, description="True if no passages were found")


class GuardrailResult(BaseModel):
    """Result of confidence & groundedness evaluation."""
    passed: bool = Field(..., description="True if retrieval confidence meets quality threshold")
    is_refusal: bool = Field(..., description="True if system should abstain and refuse to answer")
    confidence_score: float = Field(..., description="Calculated retrieval confidence score")
    reason: str = Field(..., description="Explanation of guardrail decision")
    refusal_message: Optional[str] = Field(None, description="Pre-composed refusal message if abstaining")


class GenerationResult(BaseModel):
    """Result of the LLM generation stage."""
    answer: str = Field(..., description="Generated answer text")
    prompt_used: str = Field(default="", description="Final prompt sent to LLM")
    model_name: str = Field(default="mock", description="Name of generator model")
    finish_reason: str = Field(default="stop", description="Generation termination reason")


class StageLatencyBreakdown(BaseModel):
    """Microsecond-accurate breakdown of latencies across pipeline stages."""
    stt_ms: float = Field(default=0.0, description="Speech-To-Text stage latency in ms")
    embed_ms: float = Field(default=0.0, description="Query embedding generation latency in ms")
    retrieve_ms: float = Field(default=0.0, description="FAISS vector search latency in ms")
    guardrail_ms: float = Field(default=0.0, description="Guardrail confidence check latency in ms")
    retrieval_leg_total_ms: float = Field(
        default=0.0,
        description="Total Retrieval Leg latency (embed + search + guardrails) - Budget: <200ms"
    )
    generation_ms: float = Field(default=0.0, description="LLM token generation latency in ms (reported separately)")
    total_pipeline_ms: float = Field(default=0.0, description="End-to-end total latency in ms")


class RAGResponse(BaseModel):
    """Complete structured response from the Voice-Enabled RAG pipeline."""
    query: str = Field(..., description="User query (transcribed or provided)")
    answer: str = Field(..., description="Final answer or refusal message")
    is_refusal: bool = Field(default=False, description="Whether the response is an abstention/refusal")
    confidence_score: float = Field(default=0.0, description="Retriever confidence score")
    retrieved_passages: List[RetrievedPassage] = Field(default_factory=list, description="Grounding context")
    latencies: StageLatencyBreakdown = Field(default_factory=StageLatencyBreakdown, description="Stage latency breakdown")
    error: Optional[str] = Field(default=None, description="Error message if pipeline encountered non-fatal failure")
