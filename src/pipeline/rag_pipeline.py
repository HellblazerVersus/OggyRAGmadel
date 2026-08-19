"""End-to-end Voice-Enabled RAG Pipeline orchestrator."""

import time
from typing import Optional, Union
from src.generation.generator import BaseGenerator
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.safety import CompositeGuardrail
from src.pipeline.harness import RobustExecutionHarness
from src.pipeline.schemas import (
    AudioInputRequest,
    GuardrailResult,
    RAGResponse,
    StageLatencyBreakdown,
    TextInputRequest,
)
from src.retrieval.retriever import Retriever
from src.stt.transcriber import BaseSTTTranscriber
from src.utils.logging import logger


class RAGPipeline:
    """Orchestrates Voice -> STT -> Vector Retrieval -> Guardrails -> LLM Generation."""

    def __init__(
        self,
        transcriber: BaseSTTTranscriber,
        retriever: Retriever,
        guardrail: Union[ConfidenceGuardrail, CompositeGuardrail],
        generator: BaseGenerator,
        harness: Optional[RobustExecutionHarness] = None,
        default_language: str = "hi",
    ):
        self.transcriber = transcriber
        self.retriever = retriever
        self.guardrail = guardrail
        self.generator = generator
        self.harness = harness or RobustExecutionHarness()
        self.default_language = default_language

    def process_voice(
        self,
        audio_request: AudioInputRequest,
    ) -> RAGResponse:
        """Processes an incoming voice query through the complete pipeline."""
        t_pipeline_start = time.perf_counter_ns()
        latencies = StageLatencyBreakdown()
        language = audio_request.language or self.default_language

        # 1. Speech-To-Text Stage
        t_stt_0 = time.perf_counter_ns()
        try:
            audio_source = audio_request.audio_path or audio_request.audio_bytes
            if audio_source is None:
                raise ValueError("AudioInputRequest contains neither audio_path nor audio_bytes")

            stt_result = self.harness.execute_with_retry(
                self.transcriber.transcribe,
                audio_source,
                language=language,
                stage_name="STT",
            )
            latencies.stt_ms = (time.perf_counter_ns() - t_stt_0) / 1_000_000.0
            query_text = stt_result.transcribed_text.strip()
            detected_lang = stt_result.detected_language or language

        except Exception as exc:
            latencies.stt_ms = (time.perf_counter_ns() - t_stt_0) / 1_000_000.0
            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            logger.error(f"[Pipeline] STT failed with exception: {exc}")
            return self.harness.create_stt_failure_response(str(exc), language=language, latencies=latencies)

        if not query_text:
            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            return self.harness.create_stt_failure_response(
                "No speech detected in audio input",
                language=detected_lang,
                latencies=latencies,
            )

        # Proceed with text query processing
        return self._process_text_core(
            query=query_text,
            language=detected_lang,
            latencies=latencies,
            t_pipeline_start=t_pipeline_start,
        )

    def process_text(
        self,
        text_request: TextInputRequest,
    ) -> RAGResponse:
        """Processes a direct text query (bypassing STT)."""
        t_pipeline_start = time.perf_counter_ns()
        latencies = StageLatencyBreakdown()
        language = text_request.language or self.default_language

        return self._process_text_core(
            query=text_request.query.strip(),
            language=language,
            latencies=latencies,
            t_pipeline_start=t_pipeline_start,
        )

    def _process_text_core(
        self,
        query: str,
        language: str,
        latencies: StageLatencyBreakdown,
        t_pipeline_start: int,
    ) -> RAGResponse:
        """Core pipeline logic: Pre-guardrails -> Retrieval -> Post-guardrails -> Generation -> Groundedness."""
        
        # 1.5 Pre-Retrieval Safety / Off-Topic Guardrail Check
        if isinstance(self.guardrail, CompositeGuardrail):
            pre_guard = self.guardrail.evaluate_pre_retrieval(query, language=language)
            if not pre_guard.passed:
                latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
                logger.info(f"[Pipeline] Pre-retrieval guardrail refusal: {pre_guard.reason}")
                return RAGResponse(
                    query=query,
                    answer=pre_guard.refusal_message or "Query violates policy.",
                    is_refusal=True,
                    confidence_score=0.0,
                    retrieved_passages=[],
                    latencies=latencies,
                    error=None,
                )

        # 2. Retrieval Leg (Embedding + FAISS Vector Search)
        try:
            retrieval_result, embed_ms, search_ms = self.harness.execute_with_retry(
                self.retriever.retrieve,
                query,
                stage_name="Retrieval",
            )
            latencies.embed_ms = embed_ms
            latencies.retrieve_ms = search_ms

        except Exception as exc:
            logger.error(f"[Pipeline] Retrieval leg failed: {exc}")
            latencies.retrieval_leg_total_ms = latencies.embed_ms + latencies.retrieve_ms
            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            return self.harness.create_retrieval_empty_response(query, language=language, latencies=latencies)

        # 3. Guardrails (Confidence / Relevance Check)
        if isinstance(self.guardrail, CompositeGuardrail):
            guardrail_result, guardrail_ms = self.guardrail.evaluate_post_retrieval(retrieval_result, language=language)
        else:
            guardrail_result, guardrail_ms = self.guardrail.evaluate(retrieval_result, language=language)

        latencies.guardrail_ms = guardrail_ms

        # Compute Retrieval Leg Total (Budget: <200ms)
        latencies.retrieval_leg_total_ms = (
            latencies.embed_ms + latencies.retrieve_ms + latencies.guardrail_ms
        )

        # 4. Guardrail Abstention Decision
        if guardrail_result.is_refusal:
            latencies.generation_ms = 0.0
            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0

            logger.info(f"[Pipeline] Abstaining from answer: {guardrail_result.reason}")
            return RAGResponse(
                query=query,
                answer=guardrail_result.refusal_message or "Unable to answer.",
                is_refusal=True,
                confidence_score=guardrail_result.confidence_score,
                retrieved_passages=retrieval_result.passages,
                latencies=latencies,
                error=None,
            )

        # 5. Answer Generation (Measured separately from 200ms retrieval budget)
        t_gen_0 = time.perf_counter_ns()
        try:
            gen_result = self.harness.execute_with_retry(
                self.generator.generate,
                query,
                retrieval_result.passages,
                language=language,
                stage_name="Generation",
            )
            latencies.generation_ms = (time.perf_counter_ns() - t_gen_0) / 1_000_000.0

            # 6. Post-generation Groundedness Check if CompositeGuardrail
            answer_text = gen_result.answer
            is_refusal = False
            confidence = guardrail_result.confidence_score

            if isinstance(self.guardrail, CompositeGuardrail):
                post_gen_guard = self.guardrail.evaluate_post_generation(answer_text, retrieval_result, language=language)
                if not post_gen_guard.passed:
                    is_refusal = True
                    answer_text = post_gen_guard.refusal_message or answer_text
                    confidence = post_gen_guard.confidence_score

            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0

            return RAGResponse(
                query=query,
                answer=answer_text,
                is_refusal=is_refusal,
                confidence_score=confidence,
                retrieved_passages=retrieval_result.passages,
                latencies=latencies,
                error=None,
            )

        except Exception as exc:
            latencies.generation_ms = (time.perf_counter_ns() - t_gen_0) / 1_000_000.0
            latencies.total_pipeline_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            logger.error(f"[Pipeline] Generator failed: {exc}. Activating fallback recovery.")
            return self.harness.create_generation_fallback_response(
                query=query,
                passages=retrieval_result.passages,
                confidence_score=guardrail_result.confidence_score,
                error_msg=str(exc),
                language=language,
                latencies=latencies,
            )
