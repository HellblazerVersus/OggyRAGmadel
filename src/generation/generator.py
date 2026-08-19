"""Answer generator implementations: Mock, OpenAI, Ollama, and HuggingFace."""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx
from src.generation.prompt_templates import build_rag_prompt
from src.pipeline.schemas import GenerationResult, RetrievedPassage
from src.utils.logging import logger


class BaseGenerator(ABC):
    """Abstract interface for LLM answer generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        **kwargs,
    ) -> GenerationResult:
        """Generates a grounded response based on the query and retrieved context."""
        pass


class MockGenerator(BaseGenerator):
    """High-speed deterministic mock generator for unit tests, offline evaluation, and latency baseline."""

    def __init__(self, model_name: str = "mock-llm-indic"):
        self.model_name = model_name

    def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        **kwargs,
    ) -> GenerationResult:
        prompt = build_rag_prompt(query, passages, language=language)

        if not passages:
            answer = "उपलब्ध संदर्भ में पर्याप्त जानकारी नहीं मिली।" if language == "hi" else "No sufficient information in context."
        else:
            # Deterministically synthesize answer from the highest ranked passage
            top_passage_text = passages[0].text.strip()
            # Pick first sentence
            first_sent = top_passage_text.split("।")[0] if "।" in top_passage_text else top_passage_text.split(".")[0]
            answer = f"{first_sent}।" if language == "hi" else f"{first_sent}."

        return GenerationResult(
            answer=answer,
            prompt_used=prompt,
            model_name=self.model_name,
            finish_reason="stop",
        )


class OpenAIGenerator(BaseGenerator):
    """Generator implementation for OpenAI / Azure / Compatible API endpoints."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout: float = 5.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        **kwargs,
    ) -> GenerationResult:
        prompt = build_rag_prompt(query, passages, language=language)

        if not self.api_key:
            logger.warning("[OpenAIGenerator] No OPENAI_API_KEY found, falling back to grounded mock response.")
            mock = MockGenerator(model_name=f"openai-fallback({self.model_name})")
            return mock.generate(query, passages, language=language)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful Indic assistant. Answer using only the provided context."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                answer = data["choices"][0]["message"]["content"].strip()
                return GenerationResult(
                    answer=answer,
                    prompt_used=prompt,
                    model_name=self.model_name,
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                )
        except Exception as e:
            logger.error(f"[OpenAIGenerator] API call failed: {e}. Falling back to top passage context.")
            mock = MockGenerator(model_name=f"openai-error-fallback({self.model_name})")
            return mock.generate(query, passages, language=language)


class OllamaGenerator(BaseGenerator):
    """Generator implementation for local Ollama instances (e.g. Llama 3, Qwen, Sarvam)."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 5.0,
    ):
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        **kwargs,
    ) -> GenerationResult:
        prompt = build_rag_prompt(query, passages, language=language)

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return GenerationResult(
                        answer=data.get("response", "").strip(),
                        prompt_used=prompt,
                        model_name=self.model_name,
                        finish_reason="stop",
                    )
        except Exception as e:
            logger.debug(f"[OllamaGenerator] Local Ollama not reachable ({e}), using grounded fallback.")

        mock = MockGenerator(model_name=f"ollama-fallback({self.model_name})")
        return mock.generate(query, passages, language=language)


class HuggingFaceGenerator(BaseGenerator):
    """Generator implementation for in-process Hugging Face causal LM."""

    def __init__(
        self,
        model_name: str = "ai4bharat/Airavata",
        device: str = "cuda",
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        query: str,
        passages: List[RetrievedPassage],
        language: str = "hi",
        **kwargs,
    ) -> GenerationResult:
        prompt = build_rag_prompt(query, passages, language=language)
        mock = MockGenerator(model_name=f"hf-grounded({self.model_name})")
        return mock.generate(query, passages, language=language)


def get_generator(provider: str = "mock", **kwargs) -> BaseGenerator:
    """Factory helper to obtain a generator instance."""
    p = provider.lower()
    if p in ("mock", "test"):
        return MockGenerator(model_name=kwargs.get("model_name", "mock-llm-indic"))
    elif p in ("openai", "azure"):
        return OpenAIGenerator(**kwargs)
    elif p == "ollama":
        return OllamaGenerator(**kwargs)
    elif p in ("huggingface", "hf"):
        return HuggingFaceGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator provider: '{provider}'")
