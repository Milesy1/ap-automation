"""
Streamlit UI — AP Automation POC demo.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from ap_automation.core.config import settings
from ap_automation.core.database import get_postings, init_db
from ap_automation.core.mdm import load_vendor_map, resolve_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    InvoiceCategory,
    InvoiceLine,
    RoutingDecision,
)
from ap_automation.core.pipeline import confirm, predict, write_back
from ap_automation.core.retrieval import ensure_collection, index_outcome
from ap_automation.core.sheets import ensure_sheet_headers
import httpx

ERP_URL = "http://localhost:8001"

st.set_page_config(
    page_title="AP Automation POC",
    page_icon="🧾",
    layout="wide",
)

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total": 0,
        "auto_posted": 0,
        "reviewed": 0,
        "correct": 0,
        "batch_results": [],
    }
if "initialised" not in st.session_state:
    st.session_state.initialised = False


def initialise() -> None:
    if not st.session_state.initialised:
        init_db()
        ensure_collection()
        load_vendor_map("data/vendor_map.csv")
        try:
            ensure_sheet_headers()
        except Exception:
            pass
        st.session_state.initialised = True


def post_to_erp(payload: dict) -> bool:
    try:
        r = httpx.post(f"{ERP_URL}/post-invoice", json=payload, timeout=5)
        return r.status_code == 201
    except Exception:
        return False


def process_invoice(inv: InvoiceLine, auto_confirm: bool = False) -> dict:
    prediction = predict(inv)
    result = {
        "invoice_id": str(inv.invoice_id),
        "vendor": inv.raw_vendor_name,
        "description": inv.description,
        "amount": inv.amount,
        "currency": inv.currency,
        "predicted_gl": prediction.predicted_gl,
        "predicted_cc": prediction.predicted_cost_centre,
        "confidence": prediction.weighted_confidence,
        "routing": prediction.routing_decision.value,
        "routing_reason": prediction.routing_reason,
        "ground_truth_gl": inv.ground_truth_gl,
        "category": inv.category.value,
        "provenance": prediction.provenance,
        "prediction": prediction,
        "correct": prediction.predicted_gl == inv.ground_truth_gl if inv.ground_truth_gl else None,
    }
    if auto_confirm and prediction.routing_decision == RoutingDecision.AUTO_POST:
        outcome, erp_payload = confirm(
            prediction=prediction,
            confirmed_gl=prediction.predicted_gl,
            confirmed_cc=prediction.predicted_cost_centre,
            confirming_user_id=settings.auto_post_user_id,
            source=ConfirmationSource.AUTO_POST,
        )
        post_to_erp(erp_payload.model_dump(mode="json"))
        write_back(outcome)
        result["posted"] = True
    else:
        result["posted"] = False
    return result


initialise()

st.title("🧾 AP Automation POC")
st.caption("Self-learning non-PO invoice coding — RAG + confidence gate + write-back loop")

tabs = st.tabs([
    "📄 Single Invoice",
    "📦 Batch Processing",
    "📊 Automation Rate",
    "🏦 Posted Invoices",
    "🔍 Audit Trail",
    "⚙️ Setup",
])

# ── TAB 1: Single Invoice ──────────────────────────────
with tabs[0]:
    st.subheader("Submit a single invoice")
    col1, col2 = st.columns(2)
    with col1:
        vendor = st.text_input("Vendor name", value="Meridian Facilities Ltd")
        description = st.text_input("Description", value="Office cleaning services monthly")
        amount = st.number_input("Amount", value=450.0, min_value=0.01)
    with col2:
        currency = st.selectbox("Currency", ["GBP", "EUR", "USD", "CHF"])
        entity = st.selectbox("Entity", ["UK001", "UK002", "IE001"])
        cost_centre = st.selectbox("Cost centre", ["CC100", "CC200", "CC300", "CC400", "CC500"])

    if st.button("🔍 Predict", type="primary"):
        inv = InvoiceLine(
            invoice_id=uuid4(),
            raw_vendor_name=vendor,
            description=description,
            amount=amount,
            currency=currency,
            entity_id=entity,
            cost_centre=cost_centre,
            category=InvoiceCategory.ROUTINE,
        )
        with st.spinner("Running pipeline..."):
            prediction = predict(inv)

        if prediction.routing_decision == RoutingDecision.AUTO_POST:
            st.success(f"✅ Auto-post — confidence {prediction.weighted_confidence:.1%}")
        else:
            st.warning(f"👤 Human review required — {prediction.routing_reason}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted GL", prediction.predicted_gl)
        col_b.metric("Confidence", f"{prediction.weighted_confidence:.1%}")
        col_c.metric("Evidence lines", len(prediction.evidence))

        if prediction.evidence:
            st.subheader("Top retrieved evidence")
            st.dataframe(pd.DataFrame([{
                "Description": e.description,
                "GL": e.gl_account,
                "Similarity": f"{e.similarity_score:.3f}",
                "RRF score": f"{e.rrf_score:.5f}",
                "Source": e.source.value,
            } for e in prediction.evidence[:5]]), use_container_width=True)

        st.subheader("Action")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("✅ Confirm prediction"):
                outcome, erp_payload = confirm(
                    prediction=prediction,
                    confirmed_gl=prediction.predicted_gl,
                    confirmed_cc=prediction.predicted_cost_centre,
                    confirming_user_id="demo_user",
                    source=ConfirmationSource.HUMAN_CONFIRM,
                )
                posted = post_to_erp(erp_payload.model_dump(mode="json"))
                write_back(outcome)
                st.success("Posted to ERP ✓ — write-back complete" if posted else "Write-back complete (ERP mock not running)")

        with action_col2:
            corrected_gl = st.text_input("Correct GL (if wrong)", value=prediction.predicted_gl)
            if st.button("✏️ Submit correction"):
                outcome, erp_payload = confirm(
                    prediction=prediction,
                    confirmed_gl=corrected_gl,
                    confirmed_cc=prediction.predicted_cost_centre,
                    confirming_user_id="demo_user",
                    source=ConfirmationSource.HUMAN_CORRECT,
                )
                post_to_erp(erp_payload.model_dump(mode="json"))
                write_back(outcome)
                st.success(f"Correction posted (GL: {corrected_gl}) — write-back complete")

        with st.expander("🔍 Full provenance JSON"):
            st.json(prediction.provenance)

# ── TAB 2: Batch Processing ────────────────────────────
with tabs[1]:
    st.subheader("Batch invoice processing")
    uploaded = st.file_uploader("Upload invoices CSV", type=["csv"])
    batch_size = st.slider("Reporting batch size", 50, 200, 100)

    if uploaded:
        df = pd.read_csv(StringIO(uploaded.getvalue().decode()))
        st.info(f"{len(df)} invoices loaded")

        if st.button("▶️ Run batch", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            results = []
            batch_metrics = []

            for i, row in df.iterrows():
                inv = InvoiceLine(
                    invoice_id=uuid4(),
                    raw_vendor_name=str(row["raw_vendor_name"]),
                    description=str(row["description"]),
                    amount=float(row["amount"]),
                    currency=str(row.get("currency", "GBP")),
                    entity_id=str(row.get("entity_id", "UK001")),
                    cost_centre=str(row.get("cost_centre", "CC100")),
                    category=InvoiceCategory(row.get("category", "routine")),
                    ground_truth_gl=str(row.get("ground_truth_gl", "")),
                )
                result = process_invoice(inv, auto_confirm=True)
                results.append(result)
                n = i + 1
                progress.progress(n / len(df))
                status.text(f"Processing {n}/{len(df)} — {result['vendor'][:40]}")

                if n % batch_size == 0 or n == len(df):
                    batch = results[-batch_size:]
                    auto = sum(1 for r in batch if r["routing"] == "auto_post")
                    correct = sum(1 for r in batch if r.get("correct") is True)
                    batch_metrics.append({
                        "batch": n,
                        "auto_post_rate": auto / len(batch),
                        "accuracy": correct / len(batch) if batch else 0,
                    })

            st.session_state.metrics["batch_results"] = batch_metrics
            st.success(f"Batch complete — {len(results)} invoices processed")
            st.dataframe(pd.DataFrame([{
                "Vendor": r["vendor"][:40],
                "GL": r["predicted_gl"],
                "Confidence": f"{r['confidence']:.1%}",
                "Routing": r["routing"],
                "Correct": "✓" if r.get("correct") else ("✗" if r.get("correct") is False else "—"),
            } for r in results]), use_container_width=True)

# ── TAB 3: Automation Rate ─────────────────────────────
with tabs[2]:
    st.subheader("Automation rate — write-back effect")
    batch_results = st.session_state.metrics.get("batch_results", [])
    if batch_results:
        chart_df = pd.DataFrame(batch_results).set_index("batch")
        st.line_chart(chart_df[["auto_post_rate", "accuracy"]])
        st.caption("auto_post_rate = % straight-through | accuracy = % correct vs ground truth")
        col1, col2, col3 = st.columns(3)
        first = batch_results[0]["auto_post_rate"]
        last = batch_results[-1]["auto_post_rate"]
        col1.metric("Start automation rate", f"{first:.1%}")
        col2.metric("End automation rate", f"{last:.1%}")
        col3.metric("Improvement", f"+{(last - first):.1%}", delta=f"{(last - first):.1%}")
    else:
        st.info("Run a batch in the Batch Processing tab to see the automation rate chart.")

# ── TAB 4: Posted Invoices ─────────────────────────────
with tabs[3]:
    st.subheader("ERP ledger — posted invoices")
    postings = get_postings(limit=200)
    if postings:
        st.dataframe(pd.DataFrame(postings), use_container_width=True)
        st.caption(f"{len(postings)} postings in ledger")
    else:
        st.info("No invoices posted yet.")

# ── TAB 5: Audit Trail ─────────────────────────────────
with tabs[4]:
    st.subheader("Audit trail")
    from ap_automation.core.database import get_audit_records
    records = get_audit_records(limit=100)
    if records:
        for rec in records[:10]:
            with st.expander(f"Invoice {rec['invoice_id'][:8]}… — GL {rec['confirmed_gl']} — {rec['source']}"):
                st.json(json.loads(rec["provenance_json"]))
    else:
        st.info("No audit records yet.")

# ── TAB 6: Setup ───────────────────────────────────────
with tabs[5]:
    st.subheader("Setup & initialisation")

    st.markdown("**Step 1: Generate synthetic dataset**")
    if st.button("Generate 1,000-invoice dataset"):
        with st.spinner("Generating..."):
            import random
            random.seed(42)
            from ap_automation.core.dataset import generate_dataset
            invoices = generate_dataset()
        st.success(f"Generated {len(invoices)} invoices → data/invoices.csv")

    st.markdown("**Step 2: Seed the Qdrant corpus**")
    seed_count = st.slider("Seed with N routine invoices (historical corpus)", 50, 300, 150)
    if st.button("Seed corpus"):
        path = Path("data/invoices.csv")
        if not path.exists():
            st.error("Generate the dataset first.")
        else:
            load_vendor_map("data/vendor_map.csv")
            with open(path) as f:
                rows = list(csv.DictReader(f))
            routine = [r for r in rows if r["category"] == "routine"][:seed_count]
            progress = st.progress(0)
            for i, row in enumerate(routine):
                canonical_id, _ = resolve_vendor(row["raw_vendor_name"])
                payload = {
                    "point_id": str(uuid4()),
                    "invoice_id": row["invoice_id"],
                    "vendor_id": canonical_id,
                    "entity_id": row["entity_id"],
                    "amount_band": "small",
                    "currency": row["currency"],
                    "gl_account": row["ground_truth_gl"],
                    "cost_centre": row["cost_centre"],
                    "tax_code": "STANDARD",
                    "description": row["description"],
                    "confirming_user_id": "seed",
                    "confirmation_timestamp": "2024-01-01T00:00:00",
                    "prediction_confidence": 1.0,
                    "source": "human_confirm",
                }
                index_outcome(payload)
                progress.progress((i + 1) / len(routine))
            st.success(f"Seeded {len(routine)} historical lines into Qdrant")

    st.markdown("**Qdrant status**")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        info = client.get_collection(settings.qdrant_collection)
        st.success(f"Collection `{settings.qdrant_collection}` — {info.points_count} points")
    except Exception as e:
        st.error(f"Qdrant not reachable: {e}")
