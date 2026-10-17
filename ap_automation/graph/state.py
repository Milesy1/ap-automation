"""
InvoiceState — the shared state TypedDict that flows through the LangGraph invoice pipeline.
Every node reads from and writes to this dict. LangGraph checkpoints it at each step.
"""
from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class InvoiceState(TypedDict, total=False):
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
    amount_band: str
    mdm_latency_ms: int

    # ── Retrieval node ────────────────────────────────
    evidence: list[dict]
    evidence_count: int
    retrieval_latency_ms: int

    # ── Confidence node ───────────────────────────────
    predicted_gl: Optional[str]
    predicted_cost_centre: Optional[str]
    agreement_ratio: float
    weighted_confidence: float
    confidence_latency_ms: int

    # ── Routing node ──────────────────────────────────
    routing_decision: str
    routing_reason: str

    # ── Agents node ───────────────────────────────────
    agent_results: list[dict]
    agents_called: int
    agent_latency_ms: int
    vat_treatment: Optional[str]
    cost_centre_mismatch: bool
    entity_valid: Optional[bool]

    # ── Rule gate node ────────────────────────────────
    final_routing: str
    gate_reason: str
    agent_flags: list[str]

    # ── LLM gate node ─────────────────────────────────
    llm_reasoning: Optional[str]
    llm_reasoning_confidence: Optional[float]
    llm_escalated: Optional[bool]
    llm_gate_reason: Optional[str]
    llm_latency_ms: Optional[int]

    # ── Human review node ─────────────────────────────
    review_queued_at: Optional[str]
    reviewer_id: Optional[str]
    review_notes: Optional[str]

    # ── Confirmation node ─────────────────────────────
    confirmed_gl: Optional[str]
    confirmed_cc: Optional[str]
    confirmation_source: Optional[str]
    confirmed_at: Optional[str]
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
