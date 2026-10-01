"""
Main prediction pipeline — orchestrates MDM → retrieval → confidence → routing.
Full Langfuse tracing on every invoice prediction.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from ap_automation.core import confidence as conf_module
from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import (
    AmountBand,
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


def _get_langfuse():
    """Return a Langfuse client or None if not configured."""
    try:
        from ap_automation.core.config import settings
        if not getattr(settings, 'langfuse_public_key', '') or not getattr(settings, 'langfuse_secret_key', ''):
            return None
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=getattr(settings, 'langfuse_host', 'https://cloud.langfuse.com'),
        )
    except Exception:
        return None


def predict(invoice: InvoiceLine) -> Prediction:
    """
    Full prediction pipeline for a single invoice line.

    Steps:
    1. MDM resolution
    2. Hybrid retrieval (dense + BM25 + RRF + re-rank)
    3. Confidence scoring
    4. Routing decision
    5. Build provenance JSON
    """
    lf = _get_langfuse()
    trace = None
    if lf:
        try:
            trace = lf.trace(
                name="invoice_prediction",
                input={
                    "invoice_id": str(invoice.invoice_id),
                    "vendor_raw": invoice.raw_vendor_name,
                    "description": invoice.description,
                    "amount": invoice.amount,
                    "currency": invoice.currency,
                    "entity_id": invoice.entity_id,
                },
            )
        except Exception:
            trace = None

    # ── Step 1: MDM resolution ────────────────────────
    t0 = time.time()
    span_mdm = trace.span(name="mdm_resolution", input={"vendor_raw": invoice.raw_vendor_name}) if trace else None
    canonical_vendor_id, mdm_confidence = resolve_vendor(invoice.raw_vendor_name)
    amount_band = amount_to_band(invoice.amount)
    if span_mdm:
        try:
            span_mdm.end(output={
                "canonical_id": canonical_vendor_id,
                "mdm_confidence": mdm_confidence,
                "amount_band": amount_band.value,
                "latency_ms": int((time.time() - t0) * 1000),
            })
        except Exception:
            pass

    resolved = ResolvedInvoiceLine(
        invoice_line=invoice,
        canonical_vendor_id=canonical_vendor_id,
        mdm_confidence=mdm_confidence,
        amount_band=amount_band,
    )

    # ── Step 2: Retrieval ─────────────────────────────
    t0 = time.time()
    span_retrieval = trace.span(name="qdrant_retrieval", input={
        "description": invoice.description,
        "vendor_id": canonical_vendor_id,
        "entity_id": invoice.entity_id,
    }) if trace else None

    evidence = retrieve(
        description=invoice.description,
        vendor_id=canonical_vendor_id,
        entity_id=invoice.entity_id,
        amount_band=amount_band,
        currency=invoice.currency,
    )

    if span_retrieval:
        try:
            span_retrieval.end(output={
                "evidence_count": len(evidence),
                "top_similarity_scores": [round(e.similarity_score, 4) for e in evidence[:3]],
                "top_rrf_scores": [round(e.rrf_score, 6) for e in evidence[:3]],
                "top_gl_codes": [e.gl_account for e in evidence[:3]],
                "latency_ms": int((time.time() - t0) * 1000),
            })
        except Exception:
            pass

    # ── Step 3: Confidence scoring ────────────────────
    t0 = time.time()
    span_conf = trace.span(name="confidence_scoring", input={"evidence_count": len(evidence)}) if trace else None
    predicted_gl, predicted_cc, agreement_ratio, weighted_confidence = conf_module.score(evidence)
    if span_conf:
        try:
            span_conf.end(output={
                "predicted_gl": predicted_gl,
                "agreement_ratio": round(agreement_ratio, 4),
                "weighted_confidence": round(weighted_confidence, 4),
                "latency_ms": int((time.time() - t0) * 1000),
            })
        except Exception:
            pass

    # ── Step 4: Routing ───────────────────────────────
    t0 = time.time()
    span_route = trace.span(name="routing_decision") if trace else None
    routing_decision, routing_reason = conf_module.route(
        evidence=evidence,
        vendor_id=canonical_vendor_id,
        weighted_confidence=weighted_confidence,
    )
    if span_route:
        try:
            span_route.end(output={
                "routing_decision": routing_decision.value,
                "routing_reason": routing_reason,
                "latency_ms": int((time.time() - t0) * 1000),
            })
        except Exception:
            pass

    # ── Step 5: Provenance JSON ───────────────────────
    provenance = {
        "prediction_id": str(uuid4()),
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

    # Update trace output
    if trace:
        try:
            trace.update(
                output={
                    "predicted_gl": predicted_gl,
                    "routing_decision": routing_decision.value,
                    "weighted_confidence": round(weighted_confidence, 4),
                    "evidence_count": len(evidence),
                },
                metadata={
                    "mdm_confidence": mdm_confidence,
                    "agreement_ratio": round(agreement_ratio, 4),
                    "amount_band": amount_band.value,
                }
            )
            lf.flush()
        except Exception:
            pass

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

    # Log confirmation to Langfuse
    lf = _get_langfuse()
    if lf:
        try:
            trace = lf.trace(
                name="invoice_confirmation",
                input={
                    "invoice_id": str(invoice.invoice_id),
                    "predicted_gl": prediction.predicted_gl,
                    "confirmed_gl": confirmed_gl,
                    "source": source.value,
                    "confirming_user": confirming_user_id,
                    "confidence": round(prediction.weighted_confidence, 4),
                },
            )
            # Score: was prediction correct?
            if prediction.predicted_gl and confirmed_gl:
                lf.score(
                    trace_id=trace.id,
                    name="gate.correct",
                    value=1.0 if prediction.predicted_gl == confirmed_gl else 0.0,
                    comment=f"Predicted {prediction.predicted_gl}, confirmed {confirmed_gl}",
                )
                if prediction.predicted_gl != confirmed_gl and source == ConfirmationSource.HUMAN_CORRECT:
                    lf.score(
                        trace_id=trace.id,
                        name="gate.false_positive",
                        value=1.0,
                        comment=f"Auto-post corrected: {prediction.predicted_gl} → {confirmed_gl}",
                    )
            lf.flush()
        except Exception:
            pass

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
    from uuid import uuid4
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

    # Log write-back to Langfuse
    lf = _get_langfuse()
    if lf:
        try:
            corpus_after = _collection_size()
            lf.trace(
                name="write_back",
                input={"vendor_id": outcome.canonical_vendor_id, "gl_account": outcome.gl_account},
                output={
                    "corpus_before": corpus_before,
                    "corpus_after": corpus_after,
                    "delta": corpus_after - corpus_before,
                    "source": outcome.source.value,
                },
            )
            lf.flush()
        except Exception:
            pass
