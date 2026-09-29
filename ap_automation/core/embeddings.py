"""
Embedding generation via OpenAI text-embedding-3-small.
"""

from __future__ import annotations

from openai import OpenAI

from ap_automation.core.config import settings
from ap_automation.core.preprocessing import preprocess_description


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed(text: str) -> list[float]:
    """Embed a single text string. Preprocesses description before embedding."""
    cleaned = preprocess_description(text)
    response = get_client().embeddings.create(
        model=settings.embedding_model,
        input=cleaned,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. More efficient than calling embed() in a loop."""
    cleaned = [preprocess_description(t) for t in texts]
    response = get_client().embeddings.create(
        model=settings.embedding_model,
        input=cleaned,
    )
    return [item.embedding for item in response.data]
