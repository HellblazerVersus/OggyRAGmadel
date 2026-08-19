"""Prompt templates for grounded Indic and multilingual RAG generation."""

from typing import List
from src.pipeline.schemas import RetrievedPassage


HINDI_RAG_PROMPT_TEMPLATE = """आप एक सहायक और सटीक AI सहायक हैं। नीचे दिए गए प्रासंगिक संदर्भ (Context) के आधार पर उपयोगकर्ता के प्रश्न का संक्षिप्त और सत्यनिष्ठ उत्तर दें।

नियम:
1. केवल दिए गए संदर्भ में उपलब्ध जानकारी का ही उपयोग करें।
2. अपनी तरफ से कोई नई या असत्यापित जानकारी न जोड़ें।
3. उत्तर स्पष्ट, संक्षिप्त और सटीक होना चाहिए।

प्रासंगिक संदर्भ (Context):
{context}

उपयोगकर्ता का प्रश्न:
{query}

उत्तर:"""


ENGLISH_RAG_PROMPT_TEMPLATE = """You are a helpful and accurate AI assistant. Based ONLY on the provided context passages below, answer the user's question concisely and accurately.

Rules:
1. Rely strictly on the information provided in the context.
2. Do not hallucinate or add facts not present in the context.
3. Keep the answer clear and direct.

Context:
{context}

Question:
{query}

Answer:"""


def format_context_passages(passages: List[RetrievedPassage]) -> str:
    """Formats a list of retrieved passages into a structured context block."""
    if not passages:
        return "कोई संदर्भ उपलब्ध नहीं है।"

    formatted_blocks = []
    for idx, p in enumerate(passages, start=1):
        formatted_blocks.append(f"[संदर्भ {idx}] (Passage ID: {p.passage_id})\n{p.text.strip()}")

    return "\n\n".join(formatted_blocks)


def build_rag_prompt(
    query: str,
    passages: List[RetrievedPassage],
    language: str = "hi",
) -> str:
    """Constructs the grounded RAG prompt."""
    context_str = format_context_passages(passages)
    template = HINDI_RAG_PROMPT_TEMPLATE if language == "hi" else ENGLISH_RAG_PROMPT_TEMPLATE
    return template.format(context=context_str, query=query.strip())
