"""
Embedding generation via OpenAI text-embedding-3-small.
Logs token count, cost, and latency to Langfuse as span metadata.
"""

from __future__ import annotations

import time

from openai import OpenAI

from ap_automation.core.config import settings
from ap_automation.core.preprocessing import preprocess_description


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed(text: str, langfuse_span=None) -> list[float]:
    """Embed a single text string. Preprocesses description before embedding."""
    cleaned = preprocess_description(text)
    t0 = time.time()
    response = get_client().embeddings.create(
        model=settings.embedding_model,
        input=cleaned,
    )
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.total_tokens if response.usage else 0
    cost_usd = round(token_count * 0.00000002, 8)  # $0.02 per 1M tokens

    if langfuse_span is not None:
        try:
            langfuse_span.update(metadata={
                "embedding_model": settings.embedding_model,
                "embedding_tokens": token_count,
                "embedding_cost_usd": cost_usd,
                "embedding_latency_ms": latency_ms,
                "input_text": cleaned[:200],
            })
        except Exception:
            pass

    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. More efficient than calling embed() in a loop."""
    cleaned = [preprocess_description(t) for t in texts]
    response = get_client().embeddings.create(
        model=settings.embedding_model,
        input=cleaned,
    )
    return [item.embedding for item in response.data]
