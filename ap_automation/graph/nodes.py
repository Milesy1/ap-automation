"""
LangGraph nodes — one function per pipeline step.
Each node receives the full InvoiceState, does one thing, and returns a partial dict
that LangGraph merges back into the state.
"""
from __future__ import annotations

import json
import os
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
        "_evidence_records": evidence_records,
    }


# ── Node: Confidence scoring ──────────────────────────────────────────────────

def confidence_node(state: InvoiceState) -> dict:
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


# ── Node: Rule gate ───────────────────────────────────────────────────────────

def gate_node(state: InvoiceState) -> dict:
    """
    Rule-based gate: applies agent flags on top of confidence routing.
    Any agent flag escalates to human_review regardless of confidence.
    This runs before the LLM gate and cannot be overridden by it.
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


# ── Node: LLM reasoning gate ──────────────────────────────────────────────────

def llm_gate_node(state: InvoiceState) -> dict:
    """
    LLM reasoning layer on top of the rule gate.
    Reads the full invoice context and agent results.
    Can escalate auto_post to human_review but CANNOT reverse human_review to auto_post.
    Returns plain-English reasoning and a reasoning_confidence score.
    """
    t0 = time.time()

    # Build the prompt
    agent_summary = []
    for r in state.get("agent_results", []):
        agent_summary.append({
            "agent": r.get("agent", r.get("agent_name", "unknown")),
            "success": r.get("success"),
            "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning", ""),
            "data": {k: v for k, v in r.items() if k not in ("agent", "agent_name", "success", "confidence", "reasoning", "latency_ms", "error")},
        })

    prompt = f"""You are a senior AP controller reviewing an invoice routing decision made by an automated system.

INVOICE:
- Description: {state.get("description")}
- Vendor: {state.get("raw_vendor_name", state.get("canonical_vendor_id"))}
- Amount: {state.get("currency")} {state.get("amount")}
- Entity: {state.get("entity_id")}

SYSTEM DECISION:
- Predicted GL: {state.get("predicted_gl")}
- Predicted Cost Centre: {state.get("predicted_cost_centre")}
- Weighted Confidence: {state.get("weighted_confidence")}
- Rule Gate Decision: {state.get("final_routing")}
- Rule Gate Reason: {state.get("gate_reason")}
- Agent Flags: {state.get("agent_flags", [])}

AGENT RESULTS:
{json.dumps(agent_summary, indent=2)}

Your task:
1. Review the rule gate decision and agent results
2. Decide whether the routing decision is correct
3. If the rule gate said AUTO_POST but you see a risk the rules missed, escalate to HUMAN_REVIEW
4. If the rule gate said HUMAN_REVIEW, you MUST keep it as HUMAN_REVIEW — you cannot override governance rules
5. Return ONLY valid JSON in this exact format:

{{
  "routing": "auto_post" or "human_review",
  "reasoning": "Plain English explanation of the decision in 2-3 sentences. Be specific about amounts, GL codes, flags.",
  "reasoning_confidence": 0.0 to 1.0,
  "escalated": true or false
}}"""

    try:
        import httpx
        # Try st.secrets first, then env var
        api_key = ""
        try:
            import streamlit as st
            all_secrets = dict(st.secrets)
            print(f"[LLM Gate] Available secret keys: {list(all_secrets.keys())}")
            api_key = str(all_secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
            print(f"[LLM Gate] API key found: {bool(api_key)}, length: {len(api_key)}")
        except Exception as se:
            print(f"[LLM Gate] st.secrets error: {se}")
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            print(f"[LLM Gate] env var API key found: {bool(api_key)}")

        if not api_key:
            raise ValueError("No ANTHROPIC_API_KEY found")

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15.0,
        )

        raw = response.json()["content"][0]["text"].strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        routing = result.get("routing", state.get("final_routing"))
        reasoning = result.get("reasoning", "")
        reasoning_confidence = float(result.get("reasoning_confidence", 0.0))
        escalated = result.get("escalated", False)

        # Safety: LLM can never reverse a human_review to auto_post
        if state.get("final_routing") == RoutingDecision.HUMAN_REVIEW.value:
            routing = RoutingDecision.HUMAN_REVIEW.value
            if escalated:
                escalated = False

        llm_latency_ms = int((time.time() - t0) * 1000)

        return {
            "final_routing": routing,
            "llm_reasoning": reasoning,
            "llm_reasoning_confidence": reasoning_confidence,
            "llm_escalated": escalated,
            "llm_gate_reason": f"LLM: {reasoning}",
            "llm_latency_ms": llm_latency_ms,
        }

    except Exception as e:
        print(f"[LLM Gate] Error: {e} — falling back to rule gate decision")
        return {
            "llm_reasoning": f"LLM gate unavailable: {str(e)[:100]}. Rule gate decision stands.",
            "llm_reasoning_confidence": 0.0,
            "llm_escalated": False,
            "llm_gate_reason": "Rule gate only (LLM unavailable)",
            "llm_latency_ms": int((time.time() - t0) * 1000),
        }


# ── Node: Auto post ───────────────────────────────────────────────────────────

def auto_post_node(state: InvoiceState) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "confirmed_gl": state.get("predicted_gl"),
        "confirmed_cc": state.get("predicted_cost_centre"),
        "confirmation_source": ConfirmationSource.AUTO_POST.value,
        "confirmed_at": now,
        "erp_posted": True,
        "erp_posted_at": now,
        "is_correct": None,
        "is_false_positive": False,
    }


# ── Node: Human review ────────────────────────────────────────────────────────

def human_review_node(state: InvoiceState) -> dict:
    return {
        "review_queued_at": datetime.now(timezone.utc).isoformat(),
        "erp_posted": False,
    }


# ── Node: Confirmation ────────────────────────────────────────────────────────

def confirmation_node(state: InvoiceState) -> dict:
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
    from ap_automation.core.retrieval import _collection_size

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
    """Conditional edge after LLM gate: uses final_routing which LLM may have escalated."""
    final = state.get("final_routing", RoutingDecision.HUMAN_REVIEW.value)
    if final == RoutingDecision.AUTO_POST.value:
        return "auto_post"
    return "human_review"
