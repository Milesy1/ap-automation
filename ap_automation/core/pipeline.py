"""
Main prediction pipeline — orchestrates MDM → retrieval → confidence → routing → A2A agents.
Full Langfuse v4 tracing on every invoice prediction.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from ap_automation.core import confidence as conf_module
from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    ConfirmedOutcome,
    ERPPostingPayload,
    InvoiceLine,
    Prediction,
    ResolvedInvoiceLine,
    RoutingDecision,
    amount_to_band,
)
from ap_automation.core.retrieval import retrieve, index_outcome
from ap_automation.agents.orchestrator import AgentOrchestrator
from ap_automation.graph.nodes import llm_gate_node as _llm_gate_fn

# Module-level singletons
_lf_client = None
_orchestrator = AgentOrchestrator()


def _get_langfuse():
    """Return a Langfuse client or None. Retries until successful."""
    global _lf_client

    if _lf_client is not None:
        return _lf_client

    print("[Langfuse] Initialising client...")

    try:
        try:
            import streamlit as st
            pub = str(st.secrets.get("LANGFUSE_PUBLIC_KEY", "") or "").strip()
            sec = str(st.secrets.get("LANGFUSE_SECRET_KEY", "") or "").strip()
            host = str(st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com") or "https://cloud.langfuse.com").strip()
            if pub and sec:
                print(f"[Langfuse] Keys from st.secrets. Host: {host}")
                os.environ["LANGFUSE_PUBLIC_KEY"] = pub
                os.environ["LANGFUSE_SECRET_KEY"] = sec
                os.environ["LANGFUSE_HOST"] = host
            else:
                print("[Langfuse] st.secrets empty.")
        except Exception as e:
            print(f"[Langfuse] st.secrets unavailable: {e}")

        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            print("[Langfuse] No keys — tracing disabled.")
            return None

        from langfuse import get_client
        client = get_client()
        print(f"[Langfuse] Auth check: {client.auth_check()}")
        _lf_client = client
        return _lf_client

    except Exception as e:
        print(f"[Langfuse] Init error: {e}")
        return None


def predict(invoice: InvoiceLine) -> Prediction:
    """
    Full prediction pipeline for a single invoice line.
    Uses start_as_current_observation to open the parent span first,
    then passes its id to the orchestrator for correct child nesting.
    """
    lf = _get_langfuse()
    prediction_id = str(uuid4())

    def _run_pipeline():
        # ── Step 1: MDM resolution ────────────────────────
        t0 = time.time()
        canonical_vendor_id, mdm_confidence = resolve_vendor(invoice.raw_vendor_name)
        amount_band = amount_to_band(invoice.amount)
        mdm_ms = int((time.time() - t0) * 1000)

        resolved = ResolvedInvoiceLine(
            invoice_line=invoice,
            canonical_vendor_id=canonical_vendor_id,
            mdm_confidence=mdm_confidence,
            amount_band=amount_band,
        )

        # ── Step 2: Retrieval ─────────────────────────────
        t0 = time.time()
        evidence = retrieve(
            description=invoice.description,
            vendor_id=canonical_vendor_id,
            entity_id=invoice.entity_id,
            amount_band=amount_band,
            currency=invoice.currency,
        )
        retrieval_ms = int((time.time() - t0) * 1000)

        # ── Step 3: Confidence scoring ────────────────────
        t0 = time.time()
        predicted_gl, predicted_cc, agreement_ratio, weighted_confidence = conf_module.score(evidence)
        conf_ms = int((time.time() - t0) * 1000)

        # ── Step 4: Routing ───────────────────────────────
        routing_decision, routing_reason = conf_module.route(
            evidence=evidence,
            vendor_id=canonical_vendor_id,
            weighted_confidence=weighted_confidence,
        )

        return (
            resolved, evidence, predicted_gl, predicted_cc,
            agreement_ratio, weighted_confidence, routing_decision,
            routing_reason, canonical_vendor_id, mdm_confidence,
            amount_band, mdm_ms, retrieval_ms, conf_ms,
        )

    if lf:
        try:
            with lf.start_as_current_observation(
                as_type="span",
                name="invoice_prediction",
                input={
                    "invoice_id": str(invoice.invoice_id),
                    "vendor_raw": invoice.raw_vendor_name,
                    "description": invoice.description,
                    "amount": invoice.amount,
                    "currency": invoice.currency,
                    "entity_id": invoice.entity_id,
                    "ground_truth_gl": invoice.ground_truth_gl or "",
                },
            ) as parent_span:
                print(f"[Langfuse] Parent span opened: {parent_span.id}")

                (
                    resolved, evidence, predicted_gl, predicted_cc,
                    agreement_ratio, weighted_confidence, routing_decision,
                    routing_reason, canonical_vendor_id, mdm_confidence,
                    amount_band, mdm_ms, retrieval_ms, conf_ms,
                ) = _run_pipeline()

                # ── Step 4b: A2A agents (nested under parent) ─
                agent_context = {
                    "predicted_gl": predicted_gl,
                    "predicted_cost_centre": predicted_cc,
                    "description": invoice.description,
                    "amount": invoice.amount,
                    "currency": invoice.currency,
                    "vendor_id": canonical_vendor_id,
                    "entity_id": invoice.entity_id,
                    "routing_decision": routing_decision.value,
                    "weighted_confidence": weighted_confidence,
                }
                t0 = time.time()
                agent_result = _orchestrator.run(
                    agent_context,
                    langfuse_client=lf,
                    trace_id=prediction_id,
                    parent_observation_id=parent_span.id,
                )
                agent_ms = int((time.time() - t0) * 1000)

                lf.update_current_span(
                    output={
                        "predicted_gl": predicted_gl,
                        "routing_decision": routing_decision.value,
                        "weighted_confidence": round(weighted_confidence, 4),
                        "evidence_count": len(evidence),
                        "vat_treatment": agent_result.vat.data.get("vat_treatment") if agent_result.vat and agent_result.vat.success else "unknown",
                    },
                    metadata={
                        "canonical_vendor_id": canonical_vendor_id,
                        "mdm_confidence": mdm_confidence,
                        "mdm_latency_ms": mdm_ms,
                        "evidence_count": len(evidence),
                        "top_similarity": round(evidence[0].similarity_score, 4) if evidence else 0,
                        "top_gl_codes": [e.gl_account for e in evidence[:3]],
                        "retrieval_latency_ms": retrieval_ms,
                        "agreement_ratio": round(agreement_ratio, 4),
                        "weighted_confidence": round(weighted_confidence, 4),
                        "confidence_latency_ms": conf_ms,
                        "routing_decision": routing_decision.value,
                        "routing_reason": routing_reason,
                        "amount_band": amount_band.value,
                        "a2a_agents_called": agent_result.to_dict().get("agents_called", 0),
                        "a2a_latency_ms": agent_ms,
                        "vat_treatment": agent_result.vat.data.get("vat_treatment") if agent_result.vat and agent_result.vat.success else None,
                        "cost_centre_mismatch": agent_result.cost_centre.data.get("cost_centre_mismatch") if agent_result.cost_centre and agent_result.cost_centre.success else None,
                        "entity_valid": agent_result.entity.data.get("entity_valid") if agent_result.entity and agent_result.entity.success else None,
                    },
                )

            lf.flush()
            print(f"[Langfuse] Trace flushed for {prediction_id[:8]}.")

        except Exception as e:
            print(f"[Langfuse] Trace error: {e}")
            # Fall through — still run pipeline without tracing
            (
                resolved, evidence, predicted_gl, predicted_cc,
                agreement_ratio, weighted_confidence, routing_decision,
                routing_reason, canonical_vendor_id, mdm_confidence,
                amount_band, mdm_ms, retrieval_ms, conf_ms,
            ) = _run_pipeline()
            agent_result = _orchestrator.run({
                "predicted_gl": predicted_gl,
                "predicted_cost_centre": predicted_cc,
                "description": invoice.description,
                "amount": invoice.amount,
                "currency": invoice.currency,
                "vendor_id": canonical_vendor_id,
                "entity_id": invoice.entity_id,
                "routing_decision": routing_decision.value,
                "weighted_confidence": weighted_confidence,
            })
    else:
        # No Langfuse — run pipeline untraced
        (
            resolved, evidence, predicted_gl, predicted_cc,
            agreement_ratio, weighted_confidence, routing_decision,
            routing_reason, canonical_vendor_id, mdm_confidence,
            amount_band, mdm_ms, retrieval_ms, conf_ms,
        ) = _run_pipeline()
        agent_result = _orchestrator.run({
            "predicted_gl": predicted_gl,
            "predicted_cost_centre": predicted_cc,
            "description": invoice.description,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "vendor_id": canonical_vendor_id,
            "entity_id": invoice.entity_id,
            "routing_decision": routing_decision.value,
            "weighted_confidence": weighted_confidence,
        })

    # ── LLM Gate ──────────────────────────────────────
    _cost_centre_mismatch = (
        agent_result.cost_centre.data.get("cost_centre_mismatch", False)
        if agent_result.cost_centre and agent_result.cost_centre.success else False
    )
    _entity_valid = (
        agent_result.entity.data.get("entity_valid")
        if agent_result.entity and agent_result.entity.success else None
    )
    _vat_treatment = (
        agent_result.vat.data.get("vat_treatment")
        if agent_result.vat and agent_result.vat.success else None
    )
    _agent_flags = []
    if _cost_centre_mismatch:
        _agent_flags.append("cost_centre_mismatch")
    if _vat_treatment in ("reverse_charge", "blocked_entertainment", "outside_scope"):
        _agent_flags.append(f"vat_{_vat_treatment}")
    if _entity_valid is False:
        _agent_flags.append("entity_validation_failed")
    _gate_routing = routing_decision.value if not _agent_flags else "human_review"
    _gate_reason = f"Agent flags raised: {chr(39)}{chr(44).join(_agent_flags)}{chr(39)}" if _agent_flags else routing_reason
    _llm_state = {
        "description": invoice.description,
        "raw_vendor_name": invoice.raw_vendor_name,
        "amount": invoice.amount,
        "currency": invoice.currency,
        "entity_id": invoice.entity_id,
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "weighted_confidence": weighted_confidence,
        "routing_decision": routing_decision.value,
        "final_routing": _gate_routing,
        "gate_reason": _gate_reason,
        "agent_flags": _agent_flags,
        "agent_results": [r.to_dict() for r in agent_result.results],
        "cost_centre_mismatch": _cost_centre_mismatch,
        "entity_valid": _entity_valid,
        "vat_treatment": _vat_treatment,
    }
    # Resolve API key at pipeline level where st.secrets is accessible
    _api_key = ""
    try:
        import streamlit as _st
        _api_key = str(_st.secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
    except Exception:
        pass
    if not _api_key:
        import os as _os
        _api_key = _os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"[Pipeline] API key resolved: {bool(_api_key)}, length: {len(_api_key)}")
    _llm_state["_api_key"] = _api_key
    _llm_result = _llm_gate_fn(_llm_state)
    llm_reasoning = _llm_result.get("llm_reasoning", "")
    llm_reasoning_confidence = _llm_result.get("llm_reasoning_confidence", 0.0)
    llm_escalated = _llm_result.get("llm_escalated", False)
    llm_latency_ms = _llm_result.get("llm_latency_ms", 0)
    final_routing = _llm_result.get("final_routing", _gate_routing)

    # ── Provenance JSON ───────────────────────────────
    provenance = {
        "prediction_id": prediction_id,
        "invoice_id": str(invoice.invoice_id),
        "vendor_raw": invoice.raw_vendor_name,
        "vendor_canonical": canonical_vendor_id,
        "mdm_confidence": mdm_confidence,
        "amount_band": amount_band.value,
        "evidence_count": len(evidence),
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "agreement_ratio": round(agreement_ratio, 4),
        "weighted_confidence": round(weighted_confidence, 4),
        "routing_decision": routing_decision.value,
        "routing_reason": routing_reason,
        "final_routing": final_routing,
        "llm_reasoning": llm_reasoning,
        "llm_reasoning_confidence": llm_reasoning_confidence,
        "llm_escalated": llm_escalated,
        "llm_latency_ms": llm_latency_ms,
        "a2a_agents": agent_result.to_dict(),
        "retrieved_evidence": [
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
            for e in evidence
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return Prediction(
        prediction_id=uuid4(),
        resolved_line=resolved,
        evidence=evidence,
        predicted_gl=predicted_gl,
        predicted_cost_centre=predicted_cc,
        agreement_ratio=agreement_ratio,
        weighted_confidence=weighted_confidence,
        routing_decision=routing_decision,
        routing_reason=routing_reason,
        provenance=provenance,
    )


def confirm(
    prediction: Prediction,
    confirmed_gl: str,
    confirmed_cc: str,
    confirming_user_id: str,
    source: ConfirmationSource,
) -> tuple[ConfirmedOutcome, ERPPostingPayload]:
    """Confirm a prediction (auto or human) and prepare ERP posting payload."""
    invoice = prediction.resolved_line.invoice_line
    now = datetime.now(timezone.utc)

    lf = _get_langfuse()
    if lf:
        try:
            is_correct = prediction.predicted_gl == confirmed_gl if prediction.predicted_gl else None
            is_false_positive = (
                prediction.routing_decision == RoutingDecision.AUTO_POST
                and source == ConfirmationSource.HUMAN_CORRECT
                and prediction.predicted_gl != confirmed_gl
            )
            with lf.start_as_current_observation(
                as_type="span",
                name="invoice_confirmation",
                input={
                    "invoice_id": str(invoice.invoice_id),
                    "predicted_gl": prediction.predicted_gl,
                    "confirmed_gl": confirmed_gl,
                    "source": source.value,
                    "confirming_user": confirming_user_id,
                    "ground_truth_gl": invoice.ground_truth_gl or "",
                },
                metadata={
                    "confidence": round(prediction.weighted_confidence, 4),
                    "routing_decision": prediction.routing_decision.value,
                    "correct": is_correct,
                    "false_positive": is_false_positive,
                },
            ):
                lf.score_current_trace(
                    name="gate.correct",
                    value=1.0 if is_correct else 0.0,
                    comment=f"Predicted {prediction.predicted_gl}, confirmed {confirmed_gl}",
                )
                if is_false_positive:
                    lf.score_current_trace(
                        name="gate.false_positive",
                        value=1.0,
                        comment=f"Auto-post corrected: {prediction.predicted_gl} -> {confirmed_gl}",
                    )
                if invoice.ground_truth_gl:
                    recall = 1.0 if prediction.predicted_gl == invoice.ground_truth_gl else 0.0
                    lf.score_current_trace(
                        name="retrieval.recall_at_10",
                        value=recall,
                        comment=f"Predicted: {prediction.predicted_gl}, Ground truth: {invoice.ground_truth_gl}",
                    )
                    conf_band = f"{int(prediction.weighted_confidence * 10) * 10}-{int(prediction.weighted_confidence * 10) * 10 + 10}pct"
                    lf.score_current_trace(
                        name="gate.calibration",
                        value=1.0 if is_correct else 0.0,
                        comment=f"Confidence band: {conf_band}, correct: {is_correct}",
                    )
            lf.flush()
        except Exception as e:
            print(f"[Langfuse] Confirmation trace error: {e}")

    outcome = ConfirmedOutcome(
        prediction_id=prediction.prediction_id,
        invoice_id=invoice.invoice_id,
        canonical_vendor_id=prediction.resolved_line.canonical_vendor_id,
        entity_id=invoice.entity_id,
        amount_band=prediction.resolved_line.amount_band,
        currency=invoice.currency,
        gl_account=confirmed_gl,
        cost_centre=confirmed_cc,
        confirming_user_id=confirming_user_id,
        confirmation_timestamp=now,
        prediction_confidence=prediction.weighted_confidence,
        source=source,
        description=invoice.description,
    )

    erp_payload = ERPPostingPayload(
        invoice_id=str(invoice.invoice_id),
        vendor_id=prediction.resolved_line.canonical_vendor_id,
        entity_id=invoice.entity_id,
        gl_account=confirmed_gl,
        cost_centre=confirmed_cc,
        tax_code=outcome.tax_code,
        amount=invoice.amount,
        currency=invoice.currency,
        description=invoice.description,
        posted_by=confirming_user_id,
        posted_at=now,
        trace_id=str(prediction.prediction_id),
    )

    return outcome, erp_payload


def write_back(outcome: ConfirmedOutcome) -> None:
    """Index a confirmed outcome into Qdrant for future retrieval."""
    from ap_automation.core.retrieval import _collection_size
    corpus_before = _collection_size()

    payload = {
        "point_id": str(uuid4()),
        "invoice_id": str(outcome.invoice_id),
        "vendor_id": outcome.canonical_vendor_id,
        "entity_id": outcome.entity_id,
        "amount_band": outcome.amount_band.value,
        "currency": outcome.currency,
        "gl_account": outcome.gl_account,
        "cost_centre": outcome.cost_centre,
        "tax_code": outcome.tax_code,
        "description": outcome.description,
        "confirming_user_id": outcome.confirming_user_id,
        "confirmation_timestamp": outcome.confirmation_timestamp.isoformat(),
        "prediction_confidence": outcome.prediction_confidence,
        "source": outcome.source.value,
    }
    index_outcome(payload)

    lf = _get_langfuse()
    if lf:
        try:
            corpus_after = _collection_size()
            with lf.start_as_current_observation(
                as_type="span",
                name="write_back",
                input={
                    "vendor_id": outcome.canonical_vendor_id,
                    "gl_account": outcome.gl_account,
                    "source": outcome.source.value,
                },
                metadata={
                    "corpus_before": corpus_before,
                    "corpus_after": corpus_after,
                    "delta": corpus_after - corpus_before,
                },
            ):
                pass
            lf.flush()
        except Exception as e:
            print(f"[Langfuse] Write-back trace error: {e}")
