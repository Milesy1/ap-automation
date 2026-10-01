"""
Main prediction pipeline — orchestrates MDM → retrieval → confidence → routing.
Full Langfuse v4 tracing on every invoice prediction.
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
        import os
        os.environ['LANGFUSE_PUBLIC_KEY'] = settings.langfuse_public_key
        os.environ['LANGFUSE_SECRET_KEY'] = settings.langfuse_secret_key
        os.environ['LANGFUSE_HOST'] = getattr(settings, 'langfuse_host', 'https://cloud.langfuse.com')
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


def predict(invoice: InvoiceLine) -> Prediction:
    """
    Full prediction pipeline for a single invoice line.
    Steps: MDM → retrieval → confidence → routing → provenance
    """
    lf = _get_langfuse()

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

    # ── Step 5: Provenance JSON ───────────────────────
    prediction_id = str(uuid4())
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

    # ── Langfuse trace ────────────────────────────────
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
                },
            ):
                lf.update_current_span(
                    output={
                        "predicted_gl": predicted_gl,
                        "routing_decision": routing_decision.value,
                        "weighted_confidence": round(weighted_confidence, 4),
                        "evidence_count": len(evidence),
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
            with lf.start_as_current_observation(
                as_type="span",
                name="write_back",
                input={"vendor_id": outcome.canonical_vendor_id, "gl_account": outcome.gl_account, "source": outcome.source.value},
                metadata={"corpus_before": corpus_before, "corpus_after": corpus_after, "delta": corpus_after - corpus_before},
            ):
                pass
            lf.flush()
        except Exception:
            pass

