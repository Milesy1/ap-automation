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