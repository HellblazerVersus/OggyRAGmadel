"""Generation interfaces and LLM client wrappers."""

from src.generation.generator import (
    BaseGenerator,
    MockGenerator,
    OpenAIGenerator,
    OllamaGenerator,
    HuggingFaceGenerator,
    get_generator,
)
from src.generation.prompt_templates import build_rag_prompt, format_context_passages

__all__ = [
    "BaseGenerator",
    "MockGenerator",
    "OpenAIGenerator",
    "OllamaGenerator",
    "HuggingFaceGenerator",
    "get_generator",
    "build_rag_prompt",
    "format_context_passages",
]
