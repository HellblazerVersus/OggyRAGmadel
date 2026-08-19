"""Advanced guardrails: input safety, off-topic detection, groundedness checking, and composite guardrail chain."""

import re
from typing import List, Optional, Tuple
from src.pipeline.schemas import GuardrailResult, RetrievalResult
from src.guardrails.confidence import ConfidenceGuardrail
from src.utils.logging import logger


class InputSafetyGuardrail:
    """Checks query text for unsafe/inappropriate content using keyword blocklists.
    
    Supports both Hindi (Devanagari) and English blocklists for violence, 
    explicit content, harmful instructions, and illegal activity.
    """

    def __init__(self):
        self.english_blocklist = [
            "bomb", "explosive", "weapon", "kill", "murder", "attack", "terrorist",
            "hack", "exploit", "malware", "virus", "steal", "fraud",
            "suicide", "self-harm", "abuse", "torture",
            "porn", "explicit", "nude", "xxx",
            "how to make a bomb", "how to hack", "how to kill",
        ]
        self.hindi_blocklist = [
            "बम बनाने", "बम बनाना", "हथियार", "मारना", "कत्ल", "हिंसा",
            "हैक", "चोरी", "धोखाधड़ी", "अवैध", "नशा", "ड्रग्स",
            "आत्महत्या", "अश्लील", "पोर्न",
            "बम", "विस्फोटक", "आतंकवाद",
        ]
        self.refusal_message_hi = "यह प्रश्न सुरक्षा दिशानिर्देशों का उल्लंघन करता है। मैं इसका उत्तर नहीं दे सकता।"
        self.refusal_message_en = "This query violates safety guidelines. I cannot provide an answer."

    def evaluate(self, query: str, language: str = "hi") -> GuardrailResult:
        """Checks query against safety blocklists."""
        query_lower = query.lower().strip()

        # Check English blocklist
        for keyword in self.english_blocklist:
            if keyword in query_lower:
                logger.info(f"[InputSafety] Blocked query: matched '{keyword}'")
                refusal = self.refusal_message_hi if language == "hi" else self.refusal_message_en
                return GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=0.0,
                    reason=f"Unsafe content detected (matched: '{keyword}')",
                    refusal_message=refusal,
                )

        # Check Hindi blocklist
        for keyword in self.hindi_blocklist:
            if keyword in query:
                logger.info(f"[InputSafety] Blocked query: matched Hindi keyword '{keyword}'")
                refusal = self.refusal_message_hi if language == "hi" else self.refusal_message_en
                return GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=0.0,
                    reason=f"Unsafe content detected (matched: '{keyword}')",
                    refusal_message=refusal,
                )

        return GuardrailResult(
            passed=True,
            is_refusal=False,
            confidence_score=1.0,
            reason="Query passed input safety check.",
        )


class OffTopicDetector:
    """Detects queries that are likely off-topic for a knowledge retrieval system.
    
    Identifies code generation requests, math proofs, fictional scenarios,
    real-time data requests, and other patterns outside the RAG corpus domain.
    """

    def __init__(self):
        self.off_topic_patterns_en = [
            r"\bwrite\s+(a\s+)?(python|java|javascript|code|script|program|function)\b",
            r"\b(prove|proof)\s+(that|the)\b",
            r"\bsolve\s+(the\s+)?(equation|integral|derivative)\b",
            r"\b(once upon a time|story|fiction|imagine|hypothetical)\b",
            r"\b(weather forecast|stock price|live score|current news)\b",
            r"\b(translate|convert)\s+(this|the|from)\b",
            r"\b(how to hack|crack|bypass|pirate)\b",
        ]
        self.off_topic_patterns_hi = [
            r"कोड\s+लिख",
            r"प्रोग्राम\s+बना",
            r"कहानी\s+लिख",
            r"काल्पनिक",
            r"मौसम\s+पूर्वानुमान",
            r"शेयर\s+बाजार",
            r"लाइव\s+स्कोर",
        ]
        self.refusal_message_hi = "यह प्रश्न हमारे ज्ञानकोष के दायरे से बाहर है।"
        self.refusal_message_en = "This query is outside the scope of our knowledge base."

    def evaluate(self, query: str, language: str = "hi") -> GuardrailResult:
        """Checks if query appears off-topic."""
        query_lower = query.lower().strip()

        for pattern in self.off_topic_patterns_en:
            if re.search(pattern, query_lower):
                logger.info(f"[OffTopic] Query matched off-topic pattern: '{pattern}'")
                refusal = self.refusal_message_hi if language == "hi" else self.refusal_message_en
                return GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=0.0,
                    reason=f"Query appears off-topic (matched pattern: '{pattern}')",
                    refusal_message=refusal,
                )

        for pattern in self.off_topic_patterns_hi:
            if re.search(pattern, query):
                logger.info(f"[OffTopic] Query matched Hindi off-topic pattern: '{pattern}'")
                refusal = self.refusal_message_hi if language == "hi" else self.refusal_message_en
                return GuardrailResult(
                    passed=False,
                    is_refusal=True,
                    confidence_score=0.0,
                    reason=f"Query appears off-topic (matched Hindi pattern)",
                    refusal_message=refusal,
                )

        return GuardrailResult(
            passed=True,
            is_refusal=False,
            confidence_score=1.0,
            reason="Query is on-topic.",
        )


class GroundednessChecker:
    """Checks if the generated answer is grounded in the retrieved passages.
    
    Computes token overlap ratio between the answer and retrieved context.
    If the answer contains significant information NOT present in any passage,
    it flags potential hallucination.
    """

    def __init__(self, min_overlap_ratio: float = 0.2):
        self.min_overlap_ratio = min_overlap_ratio

    def evaluate(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
        language: str = "hi",
    ) -> GuardrailResult:
        """Checks answer groundedness via token overlap with retrieved passages."""
        if not answer or not answer.strip():
            return GuardrailResult(
                passed=True,
                is_refusal=False,
                confidence_score=0.0,
                reason="Empty answer — no groundedness check needed.",
            )

        if not retrieval_result.passages:
            return GuardrailResult(
                passed=False,
                is_refusal=False,
                confidence_score=0.0,
                reason="No passages retrieved to verify groundedness.",
            )

        # Build token set from all retrieved passages
        context_tokens = set()
        for passage in retrieval_result.passages:
            tokens = passage.text.lower().split()
            context_tokens.update(tokens)

        # Get answer tokens (filter out very short tokens / stopwords)
        answer_tokens = set(t for t in answer.lower().split() if len(t) > 2)

        if not answer_tokens:
            return GuardrailResult(
                passed=True,
                is_refusal=False,
                confidence_score=1.0,
                reason="No substantive tokens in answer.",
            )

        overlap = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap) / len(answer_tokens)

        if overlap_ratio < self.min_overlap_ratio:
            logger.info(
                f"[Groundedness] Answer overlap ratio {overlap_ratio:.3f} below threshold {self.min_overlap_ratio}. Possible hallucination."
            )
            refusal = (
                "उत्तर संदर्भ सामग्री में पर्याप्त रूप से आधारित नहीं है।"
                if language == "hi"
                else "The answer does not appear to be sufficiently grounded in the retrieved context."
            )
            return GuardrailResult(
                passed=False,
                is_refusal=True,
                confidence_score=overlap_ratio,
                reason=f"Answer groundedness check failed (overlap ratio: {overlap_ratio:.3f} < {self.min_overlap_ratio})",
                refusal_message=refusal,
            )

        return GuardrailResult(
            passed=True,
            is_refusal=False,
            confidence_score=overlap_ratio,
            reason=f"Answer is grounded in context (overlap ratio: {overlap_ratio:.3f})",
        )


class CompositeGuardrail:
    """Chains multiple guardrails together in sequence.
    
    Execution order: InputSafety → OffTopic → Confidence → (post-generation) Groundedness.
    Returns the first failure, or passes if all checks pass.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        groundedness_threshold: float = 0.2,
        enable_input_safety: bool = True,
        enable_off_topic: bool = True,
        enable_groundedness: bool = True,
    ):
        self.input_safety = InputSafetyGuardrail() if enable_input_safety else None
        self.off_topic = OffTopicDetector() if enable_off_topic else None
        self.confidence = ConfidenceGuardrail(min_confidence_threshold=confidence_threshold)
        self.groundedness = GroundednessChecker(min_overlap_ratio=groundedness_threshold) if enable_groundedness else None

    def evaluate_pre_retrieval(self, query: str, language: str = "hi") -> GuardrailResult:
        """Run pre-retrieval guardrails (input safety + off-topic)."""
        if self.input_safety:
            result = self.input_safety.evaluate(query, language=language)
            if not result.passed:
                return result

        if self.off_topic:
            result = self.off_topic.evaluate(query, language=language)
            if not result.passed:
                return result

        return GuardrailResult(
            passed=True,
            is_refusal=False,
            confidence_score=1.0,
            reason="Pre-retrieval guardrails passed.",
        )

    def evaluate_post_retrieval(
        self,
        retrieval_result: RetrievalResult,
        language: str = "hi",
    ) -> Tuple[GuardrailResult, float]:
        """Run post-retrieval confidence check."""
        return self.confidence.evaluate(retrieval_result, language=language)

    def evaluate_post_generation(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
        language: str = "hi",
    ) -> GuardrailResult:
        """Run post-generation groundedness check."""
        if self.groundedness:
            return self.groundedness.evaluate(answer, retrieval_result, language=language)
        return GuardrailResult(
            passed=True,
            is_refusal=False,
            confidence_score=1.0,
            reason="Groundedness check disabled.",
        )
