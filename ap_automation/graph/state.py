"""
InvoiceState — the shared state TypedDict that flows through the LangGraph invoice pipeline.
Every node reads from and writes to this dict. LangGraph checkpoints it at each step.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from typing_extensions import TypedDict

from ap_automation.core.models import (
    AmountBand,
    RoutingDecision,
)


class InvoiceState(TypedDict, total=False):
    """
    Full state of an invoice as it flows through the LangGraph pipeline.
    Fields are populated progressively by each node.
    total=False means all fields are optional — nodes only write what they produce.
    """

    # ── Input ─────────────────────────────────────────
    invoice_id: str
    raw_vendor_name: str
    description: str
    amount: float
    currency: str
    entity_id: str
    ground_truth_gl: Optional[str]
    prediction_id: str

    # ── MDM node ──────────────────────────────────────
    canonical_vendor_id: str
    mdm_confidence: float
    amount_band: str                    # AmountBand.value
    mdm_latency_ms: int

    # ── Retrieval node ────────────────────────────────
    evidence: list[dict]                # serialised EvidenceRecord dicts
    evidence_count: int
    retrieval_latency_ms: int

    # ── Confidence node ───────────────────────────────
    predicted_gl: Optional[str]
    predicted_cost_centre: Optional[str]
    agreement_ratio: float
    weighted_confidence: float
    confidence_latency_ms: int

    # ── Routing node ──────────────────────────────────
    routing_decision: str               # RoutingDecision.value
    routing_reason: str

    # ── Agents node ───────────────────────────────────
    agent_results: list[dict]           # AgentResult.to_dict() per agent
    agents_called: int
    agent_latency_ms: int
    vat_treatment: Optional[str]
    cost_centre_mismatch: bool
    entity_valid: Optional[bool]

    # ── Gate node ─────────────────────────────────────
    final_routing: str                  # RoutingDecision.value after agent flags applied
    gate_reason: str
    agent_flags: list[str]              # e.g. ["cost_centre_mismatch", "vat_reverse_charge"]

    # ── Human review node ─────────────────────────────
    review_queued_at: Optional[str]     # ISO timestamp
    reviewer_id: Optional[str]
    review_notes: Optional[str]

    # ── Confirmation node ─────────────────────────────
    confirmed_gl: Optional[str]
    confirmed_cc: Optional[str]
    confirmation_source: Optional[str]  # ConfirmationSource.value
    confirmed_at: Optional[str]         # ISO timestamp
    is_correct: Optional[bool]
    is_false_positive: Optional[bool]

    # ── ERP post node ─────────────────────────────────
    erp_posted: bool
    erp_posted_at: Optional[str]
    tax_code: Optional[str]

    # ── Write-back node ───────────────────────────────
    write_back_complete: bool
    corpus_before: int
    corpus_after: int

    # ── Error tracking ────────────────────────────────
    error: Optional[str]
    failed_node: Optional[str]

    # ── Timing ────────────────────────────────────────
    started_at: str
    completed_at: Optional[str]
    total_latency_ms: Optional[int]
