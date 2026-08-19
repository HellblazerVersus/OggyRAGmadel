"""Confidence and relevance evaluation guardrails for grounding validation."""

import time
from typing import Optional, Tuple
from src.pipeline.schemas import GuardrailResult, RetrievalResult
from src.utils.logging import logger


class ConfidenceGuardrail:
    """Evaluates retrieval confidence and enforces abstain policies on low relevance."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.75,
        refusal_message_hi: str = "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
        refusal_message_en: str = "I do not have sufficient information in the retrieved context to answer this question.",
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.refusal_message_hi = refusal_message_hi
        self.refusal_message_en = refusal_message_en

    def evaluate(
        self,
        retrieval_result: RetrievalResult,
        language: str = "hi",
    ) -> Tuple[GuardrailResult, float]:
        """Evaluates whether retrieved passages are sufficiently confident to generate an answer.
        
        Returns:
            Tuple of (GuardrailResult, guardrail_eval_time_ms)
        """
        t0 = time.perf_counter_ns()

        if retrieval_result.is_empty or not retrieval_result.passages:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            refusal_msg = self.refusal_message_hi if language == "hi" else self.refusal_message_en
            return (
                GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=0.0,
                    reason="No passages retrieved from index.",
                    refusal_message=refusal_msg,
                ),
                elapsed_ms,
            )

        top_score = retrieval_result.top_score

        if top_score < self.min_confidence_threshold:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            refusal_msg = self.refusal_message_hi if language == "hi" else self.refusal_message_en
            logger.info(
                f"[Guardrail] Confidence {top_score:.4f} below threshold {self.min_confidence_threshold:.4f}. Triggering abstention."
            )
            return (
                GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=top_score,
                    reason=f"Top retrieval similarity score ({top_score:.4f}) is below minimum confidence threshold ({self.min_confidence_threshold:.4f}).",
                    refusal_message=refusal_msg,
                ),
                elapsed_ms,
            )

        # High confidence - passed
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return (
            GuardrailResult(
                passed=True,
                is_refusal=False,
                confidence_score=top_score,
                reason=f"Retrieval score {top_score:.4f} meets threshold {self.min_confidence_threshold:.4f}.",
                refusal_message=None,
            ),
            elapsed_ms,
        )
