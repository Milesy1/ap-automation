"""
Confidence scoring and routing gate.
"""

from __future__ import annotations

from collections import Counter

from ap_automation.core.config import settings
from ap_automation.core.models import RetrievedEvidence, RoutingDecision


# Per-vendor confidence thresholds.
# High-volume vendors with stable patterns get tighter gates over time.
# POC: all vendors start at the default threshold.
VENDOR_THRESHOLDS: dict[str, float] = {}


def get_threshold(vendor_id: str) -> float:
    return VENDOR_THRESHOLDS.get(vendor_id, settings.default_confidence_threshold)


def score(evidence: list[RetrievedEvidence]) -> tuple[str, str, float, float]:
    """
    Compute confidence from retrieved evidence.

    Returns:
        predicted_gl: the GL account with highest weighted agreement
        predicted_cost_centre: the cost centre associated with predicted_gl
        agreement_ratio: proportion of top-k sharing predicted_gl
        weighted_confidence: agreement_ratio × average similarity of agreeing lines
    """
    if not evidence:
        return "", "", 0.0, 0.0

    gl_counts = Counter(e.gl_account for e in evidence)
    predicted_gl, count = gl_counts.most_common(1)[0]
    agreement_ratio = count / len(evidence)

    agreeing = [e for e in evidence if e.gl_account == predicted_gl]
    avg_similarity = sum(e.similarity_score for e in agreeing) / len(agreeing)
    weighted_confidence = agreement_ratio * avg_similarity

    # Cost centre: most common among agreeing lines
    cc_counts = Counter(e.cost_centre for e in agreeing)
    predicted_cc = cc_counts.most_common(1)[0][0] if cc_counts else ""

    return predicted_gl, predicted_cc, agreement_ratio, weighted_confidence


def route(
    evidence: list[RetrievedEvidence],
    vendor_id: str,
    weighted_confidence: float,
) -> tuple[RoutingDecision, str]:
    """
    Determine routing decision.

    Rules (in priority order):
    1. Fewer than min_evidence_lines retrieved → always review (hard rule)
    2. Confidence below vendor threshold → review
    3. Otherwise → auto-post
    """
    if len(evidence) < settings.min_evidence_lines:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Sparse retrieval: only {len(evidence)} lines retrieved "
            f"(minimum {settings.min_evidence_lines} required)",
        )

    threshold = get_threshold(vendor_id)
    if weighted_confidence < threshold:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Confidence {weighted_confidence:.3f} below threshold {threshold:.3f} "
            f"for vendor {vendor_id}",
        )

    return (
        RoutingDecision.AUTO_POST,
        f"Confidence {weighted_confidence:.3f} meets threshold {threshold:.3f}",
    )
