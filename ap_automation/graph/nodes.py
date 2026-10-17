"""
LangGraph nodes — one function per pipeline step.
Each node receives the full InvoiceState, does one thing, and returns a partial dict
that LangGraph merges back into the state.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from ap_automation.graph.state import InvoiceState
from ap_automation.core import confidence as conf_module
from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    ConfirmedOutcome,
    ERPPostingPayload,
    RoutingDecision,
    amount_to_band,
)
from ap_automation.core.retrieval import retrieve, index_outcome
from ap_automation.agents.orchestrator import AgentOrchestrator

_orchestrator = AgentOrchestrator()


# ── Node: MDM resolution ──────────────────────────────────────────────────────

def mdm_node(state: InvoiceState) -> dict:
    t0 = time.time()
    canonical_vendor_id, mdm_confidence = resolve_vendor(state["raw_vendor_name"])
    amount_band = amount_to_band(state["amount"])
    return {
        "canonical_vendor_id": canonical_vendor_id,
        "mdm_confidence": mdm_confidence,
        "amount_band": amount_band.value,
        "mdm_latency_ms": int((time.time() - t0) * 1000),
    }


# ── Node: Retrieval ───────────────────────────────────────────────────────────

def retrieval_node(state: InvoiceState) -> dict:
    from ap_automation.core.models import AmountBand
    t0 = time.time()
    evidence_records = retrieve(
        description=state["description"],
        vendor_id=state["canonical_vendor_id"],
        entity_id=state["entity_id"],
        amount_band=AmountBand(state["amount_band"]),
        currency=state["currency"],
    )
    evidence_dicts = [
        {
            "description": e.description,
            "gl_account": e.gl_account,
            "similarity_score": round(e.similarity_score, 4),
            "rrf_score": round(e.rrf_score, 6),
            "dense_rank": e.dense_rank,
            "sparse_rank": e.sparse_rank,
            "source": e.source.value,
            "confirmation_timestamp": e.confirmation_timestamp.isoformat(),
        }
        for e in evidence_records
    ]
    return {
        "evidence": evidence_dicts,
        "evidence_count": len(evidence_dicts),
        "retrieval_latency_ms": int((time.time() - t0) * 1000),
        "_evidence_records": evidence_records,  # passed internally, not in state schema
    }


# ── Node: Confidence scoring ──────────────────────────────────────────────────

def confidence_node(state: InvoiceState) -> dict:
    # Retrieve evidence records — re-retrieve if not in state (Streamlit path)
    evidence_records = state.get("_evidence_records")
    if evidence_records is None:
        from ap_automation.core.models import AmountBand
        evidence_records = retrieve(
            description=state["description"],
            vendor_id=state["canonical_vendor_id"],
            entity_id=state["entity_id"],
            amount_band=AmountBand(state["amount_band"]),
            currency=state["currency"],
        )

    t0 = time.time()
    predicted_gl, predicted_cc, agreement_ratio, weighted_confidence = conf_module.score(evidence_records)
    return {
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "agreement_ratio": round(agreement_ratio, 4),
        "weighted_confidence": round(weighted_confidence, 4),
        "confidence_latency_ms": int((time.time() - t0) * 1000),
        "_evidence_records": evidence_records,
    }


# ── Node: Routing ─────────────────────────────────────────────────────────────

def routing_node(state: InvoiceState) -> dict:
    evidence_records = state.get("_evidence_records", [])
    routing_decision, routing_reason = conf_module.route(
        evidence=evidence_records,
        vendor_id=state["canonical_vendor_id"],
        weighted_confidence=state["weighted_confidence"],
    )
    return {
        "routing_decision": routing_decision.value,
        "routing_reason": routing_reason,
    }


# ── Node: A2A agents ──────────────────────────────────────────────────────────

def agents_node(state: InvoiceState) -> dict:
    context = {
        "predicted_gl": state.get("predicted_gl"),
        "predicted_cost_centre": state.get("predicted_cost_centre"),
        "description": state["description"],
        "amount": state["amount"],
        "currency": state["currency"],
        "vendor_id": state["canonical_vendor_id"],
        "entity_id": state["entity_id"],
        "routing_decision": state.get("routing_decision"),
        "weighted_confidence": state.get("weighted_confidence"),
    }

    t0 = time.time()
    result = _orchestrator.run(context)
    agent_ms = int((time.time() - t0) * 1000)

    vat_treatment = None
    if result.vat and result.vat.success:
        vat_treatment = result.vat.data.get("vat_treatment")

    cost_centre_mismatch = False
    if result.cost_centre and result.cost_centre.success:
        cost_centre_mismatch = result.cost_centre.data.get("cost_centre_mismatch", False)

    entity_valid = None
    if result.entity and result.entity.success:
        entity_valid = result.entity.data.get("entity_valid")

    return {
        "agent_results": [r.to_dict() for r in result.results],
        "agents_called": len(result.results),
        "agent_latency_ms": agent_ms,
        "vat_treatment": vat_treatment,
        "cost_centre_mismatch": cost_centre_mismatch,
        "entity_valid": entity_valid,
        "_agent_orchestration_result": result,
    }


# ── Node: Gate ────────────────────────────────────────────────────────────────

def gate_node(state: InvoiceState) -> dict:
    """
    Final routing decision: applies agent flags on top of the confidence-based routing.
    Any agent flag escalates to human_review regardless of confidence.
    """
    flags: list[str] = []

    if state.get("cost_centre_mismatch"):
        flags.append("cost_centre_mismatch")

    vat = state.get("vat_treatment", "")
    if vat in ("reverse_charge", "blocked_entertainment", "outside_scope"):
        flags.append(f"vat_{vat}")

    if state.get("entity_valid") is False:
        flags.append("entity_validation_failed")

    base_routing = state.get("routing_decision", RoutingDecision.HUMAN_REVIEW.value)

    if flags:
        final_routing = RoutingDecision.HUMAN_REVIEW.value
        gate_reason = f"Agent flags raised: {', '.join(flags)}"
    else:
        final_routing = base_routing
        gate_reason = "No agent flags — base routing applied"

    return {
        "final_routing": final_routing,
        "gate_reason": gate_reason,
        "agent_flags": flags,
    }


# ── Node: Auto post ───────────────────────────────────────────────────────────

def auto_post_node(state: InvoiceState) -> dict:
    """
    Auto-posts the invoice to the ERP (mock in POC).
    Sets confirmed_gl to predicted_gl and triggers write-back.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "confirmed_gl": state.get("predicted_gl"),
        "confirmed_cc": state.get("predicted_cost_centre"),
        "confirmation_source": ConfirmationSource.AUTO_POST.value,
        "confirmed_at": now,
        "erp_posted": True,
        "erp_posted_at": now,
        "is_correct": None,         # unknown until write-back feedback
        "is_false_positive": False,
    }


# ── Node: Human review ────────────────────────────────────────────────────────

def human_review_node(state: InvoiceState) -> dict:
    """
    Parks the invoice for human review.
    In production this writes to a review queue and the graph interrupts here.
    In the POC it sets the status and the UI handles the confirmation step.
    """
    return {
        "review_queued_at": datetime.now(timezone.utc).isoformat(),
        "erp_posted": False,
    }


# ── Node: Confirmation ────────────────────────────────────────────────────────

def confirmation_node(state: InvoiceState) -> dict:
    """
    Receives a human confirmation decision and prepares for write-back.
    Detects false positives (auto-posted predictions corrected by human).
    """
    predicted = state.get("predicted_gl")
    confirmed = state.get("confirmed_gl")
    is_correct = predicted == confirmed if predicted and confirmed else None
    is_false_positive = (
        state.get("routing_decision") == RoutingDecision.AUTO_POST.value
        and state.get("confirmation_source") == ConfirmationSource.HUMAN_CORRECT.value
        and not is_correct
    )
    return {
        "is_correct": is_correct,
        "is_false_positive": is_false_positive,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Node: Write-back ──────────────────────────────────────────────────────────

def write_back_node(state: InvoiceState) -> dict:
    """Indexes the confirmed outcome into Qdrant for future retrieval."""
    from ap_automation.core.retrieval import _collection_size
    from ap_automation.core.models import AmountBand

    corpus_before = _collection_size()

    payload = {
        "point_id": str(uuid4()),
        "invoice_id": state["invoice_id"],
        "vendor_id": state["canonical_vendor_id"],
        "entity_id": state["entity_id"],
        "amount_band": state["amount_band"],
        "currency": state["currency"],
        "gl_account": state.get("confirmed_gl") or state.get("predicted_gl"),
        "cost_centre": state.get("confirmed_cc") or state.get("predicted_cost_centre"),
        "tax_code": state.get("tax_code", "T1"),
        "description": state["description"],
        "confirming_user_id": state.get("reviewer_id", "system"),
        "confirmation_timestamp": state.get("confirmed_at", datetime.now(timezone.utc).isoformat()),
        "prediction_confidence": state.get("weighted_confidence", 0.0),
        "source": state.get("confirmation_source", ConfirmationSource.AUTO_POST.value),
    }
    index_outcome(payload)

    corpus_after = _collection_size()
    completed_at = datetime.now(timezone.utc).isoformat()

    started = state.get("started_at")
    total_ms = None
    if started:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(started)
            total_ms = int(delta.total_seconds() * 1000)
        except Exception:
            pass

    return {
        "write_back_complete": True,
        "corpus_before": corpus_before,
        "corpus_after": corpus_after,
        "completed_at": completed_at,
        "total_latency_ms": total_ms,
    }


# ── Edge router ───────────────────────────────────────────────────────────────

def route_after_gate(state: InvoiceState) -> str:
    """Conditional edge: directs to auto_post or human_review based on gate_node output."""
    final = state.get("final_routing", RoutingDecision.HUMAN_REVIEW.value)
    if final == RoutingDecision.AUTO_POST.value:
        return "auto_post"
    return "human_review"
