"""Resilience harness: retries, exponential backoff, circuit breaking, and error recovery."""

import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from src.pipeline.schemas import (
    GuardrailResult,
    RAGResponse,
    RetrievedPassage,
    StageLatencyBreakdown,
)
from src.utils.logging import logger


class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, fast-reject calls
    HALF_OPEN = "HALF_OPEN"# Trial run


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is rejected by an open circuit breaker."""
    pass


class RobustExecutionHarness:
    """Provides resilient execution wrapper with retries, backoff, and circuit breaker protection."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        enable_circuit_breaker: bool = True,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_recovery_timeout = circuit_recovery_timeout

        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.time()

    @property
    def circuit_state(self) -> CircuitState:
        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._last_state_change > self.circuit_recovery_timeout:
                logger.info("[CircuitBreaker] Recovery timeout expired. Transitioning to HALF_OPEN.")
                self._circuit_state = CircuitState.HALF_OPEN
        return self._circuit_state

    def _record_success(self) -> None:
        if self._circuit_state == CircuitState.HALF_OPEN:
            logger.info("[CircuitBreaker] Operation succeeded in HALF_OPEN. Resetting to CLOSED.")
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0

    def _record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.circuit_failure_threshold:
            logger.error(
                f"[CircuitBreaker] Failure threshold ({self.circuit_failure_threshold}) reached! Tripping circuit to OPEN."
            )
            self._circuit_state = CircuitState.OPEN
            self._last_state_change = time.time()

    def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        stage_name: str = "stage",
        max_retries: Optional[int] = None,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        fallback_value: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Executes a callable with exponential backoff and retry policy."""
        retries = max_retries if max_retries is not None else self.max_retries

        if self.enable_circuit_breaker and self.circuit_state == CircuitState.OPEN:
            raise CircuitBreakerOpenException(
                f"Circuit breaker is OPEN for stage '{stage_name}'. Fast-rejecting request."
            )

        attempt = 0
        while attempt <= retries:
            try:
                result = func(*args, **kwargs)
                if self.enable_circuit_breaker:
                    self._record_success()
                return result
            except retryable_exceptions as e:
                attempt += 1
                if attempt > retries:
                    if self.enable_circuit_breaker:
                        self._record_failure()
                    logger.error(f"[{stage_name}] Failed after {retries} retries: {e}")
                    if fallback_value is not None:
                        logger.warning(f"[{stage_name}] Returning provided fallback value.")
                        return fallback_value
                    raise

                sleep_duration = (self.backoff_factor ** attempt) + random.uniform(0.01, 0.1)
                logger.warning(
                    f"[{stage_name}] Attempt {attempt}/{retries} failed with ({type(e).__name__}: {e}). Retrying in {sleep_duration:.2f}s..."
                )
                time.sleep(sleep_duration)

    # --- Explicit Error Recovery Paths ---

    @staticmethod
    def create_stt_failure_response(
        error_msg: str,
        language: str = "hi",
        latencies: Optional[StageLatencyBreakdown] = None,
    ) -> RAGResponse:
        """Structured error recovery for speech recognition failures."""
        msg = (
            "ध्वनि पहचानी नहीं जा सकी। कृपया स्पष्ट बोलें या पुनः प्रयास करें।"
            if language == "hi"
            else "Audio could not be transcribed. Please speak clearly or try again."
        )
        return RAGResponse(
            query="<STT_FAILED>",
            answer=msg,
            is_refusal=True,
            confidence_score=0.0,
            retrieved_passages=[],
            latencies=latencies or StageLatencyBreakdown(),
            error=f"STT Failure: {error_msg}",
        )

    @staticmethod
    def create_retrieval_empty_response(
        query: str,
        language: str = "hi",
        latencies: Optional[StageLatencyBreakdown] = None,
    ) -> RAGResponse:
        """Structured error recovery when vector search yields zero passages."""
        msg = (
            "इस विषय पर ज्ञानकोष में कोई प्रासंगिक संदर्भ नहीं मिला।"
            if language == "hi"
            else "No relevant passages found in the knowledge index for this query."
        )
        return RAGResponse(
            query=query,
            answer=msg,
            is_refusal=True,
            confidence_score=0.0,
            retrieved_passages=[],
            latencies=latencies or StageLatencyBreakdown(),
            error=None,
        )

    @staticmethod
    def create_generation_fallback_response(
        query: str,
        passages: List[RetrievedPassage],
        confidence_score: float,
        error_msg: str,
        language: str = "hi",
        latencies: Optional[StageLatencyBreakdown] = None,
    ) -> RAGResponse:
        """Graceful degradation when LLM generation times out or fails: extracts top passage directly."""
        fallback_answer = (
            f"[बैकअप संदर्भ]: {passages[0].text.strip()}"
            if passages
            else ("उत्तर उत्पन्न करने में समस्या आई।" if language == "hi" else "Error generating answer.")
        )
        return RAGResponse(
            query=query,
            answer=fallback_answer,
            is_refusal=False,
            confidence_score=confidence_score,
            retrieved_passages=passages,
            latencies=latencies or StageLatencyBreakdown(),
            error=f"Generator Failure (Recovered via fallback): {error_msg}",
        )
