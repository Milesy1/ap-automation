"""
Retrieval module — hybrid dense + sparse (BM25) with RRF score fusion and re-ranking.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi

from ap_automation.core.config import settings
from ap_automation.core.embeddings import embed
from ap_automation.core.models import (
    AmountBand,
    ConfirmationSource,
    RetrievedEvidence,
)


def get_qdrant() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection() -> None:
    """Create Qdrant collection if it does not exist."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )


def _build_filter(vendor_id: str) -> Filter:
    """
    Pre-filter by vendor only.
    POC: vendor ID is the primary signal; entity/amount/currency
    filtering applied post-retrieval if needed.
    """
    return Filter(
        must=[
            FieldCondition(key="vendor_id", match=MatchValue(value=vendor_id)),
        ]
    )


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _recency_weight(timestamp: datetime) -> float:
    """More recent = higher weight. Exponential decay over 2 years."""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    days_old = (now - timestamp).days
    return math.exp(-days_old / 730)


def retrieve(
    description: str,
    vendor_id: str,
    entity_id: str,
    amount_band: AmountBand,
    currency: str,
    top_k: int | None = None,
) -> list[RetrievedEvidence]:
    """
    Hybrid retrieval: dense (Qdrant cosine) + sparse (BM25) fused with RRF.
    Re-ranked by: cosine primary, recency secondary, frequency tertiary.
    Returns up to top_k results.
    """
    k = top_k or settings.top_k
    client = get_qdrant()
    query_vector = embed(description)

    # Filter by vendor only — broadest possible match for POC
    vendor_filter = _build_filter(vendor_id)

    # --- Dense retrieval ---
    dense_response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=vendor_filter,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    dense_results = dense_response.points

    # If vendor filter returns nothing, fall back to unfiltered search
    if not dense_results:
        dense_response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        dense_results = dense_response.points

    if not dense_results:
        return []

    # --- Sparse retrieval (BM25) over the same candidate pool ---
    scroll_filter = vendor_filter if dense_results else None
    all_candidates = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=scroll_filter,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )[0]

    # Fall back to full corpus for BM25 if vendor pool is tiny
    if len(all_candidates) < 3:
        all_candidates = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )[0]

    corpus_descriptions = [p.payload.get("description", "") for p in all_candidates]
    tokenised_corpus = [d.lower().split() for d in corpus_descriptions]
    bm25 = BM25Okapi(tokenised_corpus)
    query_tokens = description.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    candidate_ids = [p.id for p in all_candidates]
    bm25_rank_map: dict[str, int] = {}
    for rank, (idx, _) in enumerate(bm25_ranked):
        if idx < len(candidate_ids):
            bm25_rank_map[str(candidate_ids[idx])] = rank + 1

    dense_rank_map: dict[str, int] = {
        str(r.id): i + 1 for i, r in enumerate(dense_results)
    }

    # --- RRF fusion ---
    all_ids = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
    rrf_scores: dict[str, float] = {}
    for pid in all_ids:
        dense_rank = dense_rank_map.get(pid, k * 2)
        sparse_rank = bm25_rank_map.get(pid, k * 2)
        rrf_scores[pid] = _rrf_score(dense_rank) + _rrf_score(sparse_rank)

    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

    payload_map = {str(p.id): p.payload for p in all_candidates}
    for r in dense_results:
        if str(r.id) not in payload_map:
            payload_map[str(r.id)] = r.payload

    # --- Build RetrievedEvidence list ---
    evidence: list[RetrievedEvidence] = []
    for pid in top_ids:
        payload = payload_map.get(pid)
        if not payload:
            continue
        ts_raw = payload.get("confirmation_timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)

        cosine_sim = next(
            (r.score for r in dense_results if str(r.id) == pid), 0.0
        )

        evidence.append(RetrievedEvidence(
            invoice_id=str(payload.get("invoice_id", "")),
            description=str(payload.get("description", "")),
            gl_account=str(payload.get("gl_account", "")),
            cost_centre=str(payload.get("cost_centre", "")),
            vendor_id=str(payload.get("vendor_id", "")),
            similarity_score=cosine_sim,
            dense_rank=dense_rank_map.get(pid, k * 2),
            sparse_rank=bm25_rank_map.get(pid, k * 2),
            rrf_score=rrf_scores[pid],
            confirmation_timestamp=ts,
            source=ConfirmationSource(
                payload.get("source", ConfirmationSource.HUMAN_CONFIRM)
            ),
        ))

    # --- Re-rank: cosine primary, recency secondary, frequency tertiary ---
    from collections import Counter
    gl_counts = Counter(e.gl_account for e in evidence)
    max_count = max(gl_counts.values(), default=1)

    def rerank_score(e: RetrievedEvidence) -> float:
        cosine = e.similarity_score
        recency = _recency_weight(e.confirmation_timestamp)
        frequency = gl_counts[e.gl_account] / max_count
        return (cosine * 0.6) + (recency * 0.25) + (frequency * 0.15)

    evidence.sort(key=rerank_score, reverse=True)
    return evidence[:k]


def index_outcome(outcome_payload: dict) -> None:
    """Write a confirmed outcome into the Qdrant corpus."""
    client = get_qdrant()
    vector = embed(outcome_payload["description"])
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(
            id=outcome_payload["point_id"],
            vector=vector,
            payload=outcome_payload,
        )]
    )
