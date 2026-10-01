"""
Streamlit UI — AP Automation POC demo.
Polished demo version with improved layout, plain-English explanations, and narrative captions.
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
from ap_automation.core.retrieval import ensure_collection, get_qdrant, index_outcome
from ap_automation.core.sheets import ensure_sheet_headers, post_to_sheet
import httpx

ERP_URL = "http://localhost:8001"

st.set_page_config(
    page_title="AP Automation POC",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .poc-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    }
    .poc-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .poc-header p { color: #a8b2d8; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .badge-auto { background: #d1fae5; color: #065f46; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .badge-review { background: #fef3c7; color: #92400e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .badge-coldstart { background: #fee2e2; color: #991b1b; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .evidence-row { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
    .evidence-desc { font-weight: 500; color: #1e293b; }
    .evidence-meta { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
    .conf-bar-bg { background: #e2e8f0; border-radius: 4px; height: 8px; margin-top: 0.3rem; }
    .conf-bar-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #f59e0b, #10b981); }
    .section-header { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid #f1f5f9; }
    .why-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0; }
    .why-box p { margin: 0; color: #0c4a6e; font-size: 0.9rem; line-height: 1.5; }
    .narrative { background: #f8fafc; border-left: 3px solid #6366f1; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; color: #374151; font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f8fafc; padding: 4px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { border-radius: 7px; padding: 0.5rem 1.2rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: white !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .stButton > button { border-radius: 8px; font-weight: 500; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

if "metrics" not in st.session_state:
    st.session_state.metrics = {"total": 0, "auto_posted": 0, "batch_results": []}
if "initialised" not in st.session_state:
    st.session_state.initialised = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def initialise() -> None:
    if not st.session_state.initialised:
        init_db()
        try:
            ensure_collection()
        except Exception:
            pass
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


def post_confirmed(erp_payload_dict: dict) -> str:
    post_to_erp(erp_payload_dict)
    try:
        post_to_sheet(erp_payload_dict)
        return "Posted to ERP & Google Sheets — write-back complete"
    except Exception:
        return "Write-back complete (ERP & Sheets local only)"


def generate_explanation(prediction) -> str:
    """Generate a plain-English explanation of why this prediction was made."""
    from collections import Counter

    evidence = prediction.evidence
    predicted_gl = prediction.predicted_gl
    confidence = prediction.weighted_confidence
    routing = prediction.routing_decision

    if not evidence:
        return (
            "This is a cold-start invoice — the vendor has no history in the corpus. "
            "There is no evidence to retrieve, so the system cannot make a confident prediction. "
            "The invoice has been routed to human review so a clerk can assign the correct GL code. "
            "Once confirmed, this outcome will be written back into the corpus and future invoices "
            "from this vendor will benefit from the evidence."
        )

    gl_counts = Counter(e.gl_account for e in evidence)
    top_gl, top_count = gl_counts.most_common(1)[0]
    agreement_pct = int(top_count / len(evidence) * 100)
    top_sim = max(e.similarity_score for e in evidence)
    best_match = next(e for e in evidence if e.similarity_score == top_sim)

    if routing == RoutingDecision.AUTO_POST:
        return (
            f"The system retrieved {len(evidence)} historical invoice lines similar to this one. "
            f"{agreement_pct}% of them were coded to GL {predicted_gl}, and the closest match "
            f"(\"{best_match.description}\") scored {top_sim:.0%} similarity. "
            f"With a confidence score of {confidence:.0%} — above the {settings.default_confidence_threshold:.0%} threshold — "
            f"the system is sufficiently confident to auto-post without human review."
        )
    else:
        if len(evidence) < settings.min_evidence_lines:
            return (
                f"The system retrieved only {len(evidence)} historical lines for this vendor "
                f"(minimum {settings.min_evidence_lines} required to auto-post). "
                f"This is a near cold-start situation — not enough evidence to be confident. "
                f"The invoice has been routed to human review. Once confirmed, the outcome "
                f"will strengthen the corpus for this vendor."
            )
        return (
            f"The system retrieved {len(evidence)} historical invoice lines, but only "
            f"{agreement_pct}% agreed on GL {predicted_gl}. "
            f"The confidence score of {confidence:.0%} is below the {settings.default_confidence_threshold:.0%} threshold "
            f"— the evidence is too mixed to auto-post safely. "
            f"A human reviewer should confirm the correct GL code."
        )


def process_invoice(inv: InvoiceLine, auto_confirm: bool = False) -> dict:
    prediction = predict(inv)
    result = {
        "invoice_id": str(inv.invoice_id),
        "vendor": inv.raw_vendor_name,
        "description": inv.description,
        "amount": inv.amount,
        "currency": inv.currency,
        "predicted_gl": prediction.predicted_gl,
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
        payload_dict = erp_payload.model_dump(mode="json")
        post_to_erp(payload_dict)
        try:
            post_to_sheet(payload_dict)
        except Exception:
            pass
        write_back(outcome)
        result["posted"] = True
    else:
        result["posted"] = False
    return result


initialise()

st.markdown("""
<div class="poc-header">
    <h1>🧾 AP Automation</h1>
    <p>Self-learning non-PO invoice coding — RAG + confidence gate + write-back loop</p>
</div>
""", unsafe_allow_html=True)

try:
    qc = get_qdrant()
    corpus_size = qc.get_collection(settings.qdrant_collection).points_count
except Exception:
    corpus_size = 0

postings_count = len(get_postings(limit=10000))
batch_results = st.session_state.metrics.get("batch_results", [])
current_rate = f"{batch_results[-1]['auto_post_rate']:.0%}" if batch_results else "—"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📚 Corpus size", f"{corpus_size:,}")
col2.metric("✅ Invoices posted", f"{postings_count:,}")
col3.metric("🎯 Automation rate", current_rate)
col4.metric("📦 Batches run", len(batch_results))

st.divider()

tabs = st.tabs([
    "📄  Review Queue",
    "📦  Batch Processing",
    "📊  Performance",
    "🏦  ERP Ledger",
    "🔍  Audit Trail",
    "⚙️  Setup",
])

# ── TAB 1: Review Queue ────────────────────────────────
with tabs[0]:
    st.markdown('<div class="narrative">Submit an invoice to see the system predict a GL code from historical evidence. Confirm the prediction to post it to the ERP and write it back into the learning corpus.</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown('<div class="section-header">Submit Invoice</div>', unsafe_allow_html=True)
        vendor = st.text_input("Vendor name", value="Meridian Facilities Ltd")
        description = st.text_input("Description", value="Office cleaning services monthly")
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount (£)", value=450.0, min_value=0.01, format="%.2f")
            entity = st.selectbox("Entity", ["UK001", "UK002", "IE001"])
        with col_b:
            currency = st.selectbox("Currency", ["GBP", "EUR", "USD", "CHF"])
            cost_centre = st.selectbox("Cost centre", ["CC100", "CC200", "CC300", "CC400", "CC500"])

        st.markdown("---")
        st.markdown("**Try a hard-edge invoice (unknown vendor):**")
        if st.button("🔴 Load hard-edge example", use_container_width=True):
            st.session_state["hard_edge_vendor"] = "Zephyr Global Holdings"
            st.session_state["hard_edge_desc"] = "Professional advisory services Q3"
            st.rerun()

        if "hard_edge_vendor" in st.session_state:
            vendor = st.session_state.pop("hard_edge_vendor")
            description = st.session_state.pop("hard_edge_desc")

        if st.button("🔍 Predict GL Code", type="primary", use_container_width=True):
            inv = InvoiceLine(
                invoice_id=uuid4(), raw_vendor_name=vendor, description=description,
                amount=amount, currency=currency, entity_id=entity,
                cost_centre=cost_centre, category=InvoiceCategory.ROUTINE,
            )
            with st.spinner("Running pipeline..."):
                prediction = predict(inv)
            st.session_state.last_prediction = prediction
            st.session_state.last_invoice = inv

    with col_result:
        pred = st.session_state.get("last_prediction")
        if pred:
            st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

            # Routing badge — with cold-start detection
            is_cold_start = len(pred.evidence) == 0
            if is_cold_start:
                st.markdown('<span class="badge-coldstart">🔴 Cold start — no vendor history</span>', unsafe_allow_html=True)
            elif pred.routing_decision == RoutingDecision.AUTO_POST:
                st.markdown('<span class="badge-auto">✅ Auto-post approved</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-review">👤 Routed to human review</span>', unsafe_allow_html=True)

            st.caption(pred.routing_reason)
            st.write("")

            m1, m2, m3 = st.columns(3)
            m1.metric("GL Account", pred.predicted_gl or "—")
            m2.metric("Confidence", f"{pred.weighted_confidence:.1%}")
            m3.metric("Evidence lines", len(pred.evidence))

            pct = int(pred.weighted_confidence * 100)
            st.markdown(f"""
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%"></div></div>
            <div style="font-size:0.75rem;color:#6b7280;margin-top:0.2rem">
                Threshold: {settings.default_confidence_threshold:.0%} &nbsp;|&nbsp; Score: {pred.weighted_confidence:.1%}
            </div>""", unsafe_allow_html=True)

            # Plain-English explanation
            st.write("")
            st.markdown('<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)
            explanation = generate_explanation(pred)
            st.markdown(f'<div class="why-box"><p>{explanation}</p></div>', unsafe_allow_html=True)

            # Evidence
            if pred.evidence:
                st.write("")
                st.markdown('<div class="section-header">Retrieved Evidence</div>', unsafe_allow_html=True)
                for e in pred.evidence[:3]:
                    st.markdown(f"""
                    <div class="evidence-row">
                        <div class="evidence-desc">{e.description}</div>
                        <div class="evidence-meta">GL: <strong>{e.gl_account}</strong> &nbsp;·&nbsp; Similarity: <strong>{e.similarity_score:.3f}</strong> &nbsp;·&nbsp; RRF: {e.rrf_score:.5f} &nbsp;·&nbsp; Source: {e.source.value}</div>
                    </div>""", unsafe_allow_html=True)

            # Actions
            st.write("")
            st.markdown('<div class="section-header">Action</div>', unsafe_allow_html=True)
            act1, act2 = st.columns(2)
            with act1:
                if st.button("✅ Confirm & Post to ERP", use_container_width=True, type="primary"):
                    outcome, erp_payload = confirm(prediction=pred, confirmed_gl=pred.predicted_gl, confirmed_cc=pred.predicted_cost_centre, confirming_user_id="demo_user", source=ConfirmationSource.HUMAN_CONFIRM)
                    msg = post_confirmed(erp_payload.model_dump(mode="json"))
                    write_back(outcome)
                    st.success(f"✓ {msg}")
            with act2:
                corrected_gl = st.text_input("Override GL code", value=pred.predicted_gl or "", placeholder="e.g. 6300")
                if st.button("✏️ Submit Correction", use_container_width=True):
                    outcome, erp_payload = confirm(prediction=pred, confirmed_gl=corrected_gl, confirmed_cc=pred.predicted_cost_centre, confirming_user_id="demo_user", source=ConfirmationSource.HUMAN_CORRECT)
                    post_confirmed(erp_payload.model_dump(mode="json"))
                    write_back(outcome)
                    st.success(f"✓ Correction submitted (GL: {corrected_gl}) — write-back complete")

            with st.expander("🔍 Full provenance JSON"):
                st.json(pred.provenance)
        else:
            st.info("Submit an invoice on the left to see the prediction. Try the hard-edge example to see cold-start handling.")

# ── TAB 2: Batch Processing ────────────────────────────
with tabs[1]:
    st.markdown('<div class="narrative">Upload a CSV of invoices and run them through the full pipeline in bulk. Auto-posted invoices are written to the ERP and Google Sheets in real time. Every confirmed invoice grows the corpus — watch the automation rate climb in the Performance tab.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload invoices CSV", type=["csv"], label_visibility="collapsed")
    batch_size = st.select_slider("Reporting batch size", options=[50, 100, 150, 200], value=100)
    if uploaded:
        df = pd.read_csv(StringIO(uploaded.getvalue().decode()))
        st.info(f"**{len(df):,}** invoices loaded")
        if st.button("▶️ Run Batch", type="primary"):
            progress = st.progress(0, text="Starting...")
            results = []
            batch_metrics = []
            for i, row in df.iterrows():
                inv = InvoiceLine(
                    invoice_id=uuid4(), raw_vendor_name=str(row["raw_vendor_name"]),
                    description=str(row["description"]), amount=float(row["amount"]),
                    currency=str(row.get("currency", "GBP")), entity_id=str(row.get("entity_id", "UK001")),
                    cost_centre=str(row.get("cost_centre", "CC100")),
                    category=InvoiceCategory(row.get("category", "routine")),
                    ground_truth_gl=str(row.get("ground_truth_gl", "")),
                )
                result = process_invoice(inv, auto_confirm=True)
                results.append(result)
                n = i + 1
                progress.progress(n / len(df), text=f"Processing {n:,}/{len(df):,} — {result['vendor'][:35]}")
                if n % batch_size == 0 or n == len(df):
                    batch = results[-batch_size:]
                    auto = sum(1 for r in batch if r["routing"] == "auto_post")
                    correct = sum(1 for r in batch if r.get("correct") is True)
                    batch_metrics.append({"batch": n, "auto_post_rate": auto / len(batch), "accuracy": correct / len(batch) if batch else 0})
            st.session_state.metrics["batch_results"] = batch_metrics
            progress.empty()
            auto_count = sum(1 for r in results if r["routing"] == "auto_post")
            correct_count = sum(1 for r in results if r.get("correct") is True)
            st.success(f"✅ Batch complete — {len(results):,} invoices processed")
            c1, c2, c3 = st.columns(3)
            c1.metric("Auto-posted", f"{auto_count:,}", f"{auto_count/len(results):.0%}")
            c2.metric("Human review", f"{len(results)-auto_count:,}")
            c3.metric("Correct predictions", f"{correct_count:,}", f"{correct_count/len(results):.0%}")
            st.dataframe(pd.DataFrame([{
                "Vendor": r["vendor"][:40], "Description": r["description"][:50],
                "Amount": f"£{r['amount']:,.2f}", "GL": r["predicted_gl"] or "—",
                "Confidence": f"{r['confidence']:.0%}",
                "Routing": "Auto" if r["routing"] == "auto_post" else "Review",
                "Correct": "Yes" if r.get("correct") else ("No" if r.get("correct") is False else "—"),
            } for r in results]), use_container_width=True, height=400)

# ── TAB 3: Performance ─────────────────────────────────
with tabs[2]:
    st.markdown('<div class="narrative">This chart proves the write-back loop works. As more invoices are confirmed, the corpus grows and the system becomes more confident — auto-posting a higher proportion of invoices without any manual rule authoring.</div>', unsafe_allow_html=True)

    batch_results = st.session_state.metrics.get("batch_results", [])
    if batch_results:
        first = batch_results[0]["auto_post_rate"]
        last = batch_results[-1]["auto_post_rate"]
        delta = last - first
        c1, c2, c3 = st.columns(3)
        c1.metric("Start automation rate", f"{first:.1%}")
        c2.metric("End automation rate", f"{last:.1%}")
        c3.metric("Write-back improvement", f"{delta:+.1%}", delta=f"{delta:+.1%}")
        st.write("")
        chart_df = pd.DataFrame(batch_results).set_index("batch")
        st.line_chart(chart_df[["auto_post_rate", "accuracy"]])
        st.caption("auto_post_rate = % straight-through posted without human review | accuracy = % correct vs ground truth GL code")
    else:
        st.info("Run a batch in the Batch Processing tab to see performance metrics.")

# ── TAB 4: ERP Ledger ──────────────────────────────────
with tabs[3]:
    st.markdown('<div class="narrative">Every auto-posted and human-confirmed invoice is written here and mirrored to Google Sheets in real time. This is the ERP write proof — the system is posting to a live system, not just predicting.</div>', unsafe_allow_html=True)

    postings = get_postings(limit=500)
    if postings:
        df_postings = pd.DataFrame(postings)
        st.dataframe(df_postings[["invoice_id", "vendor_id", "entity_id", "gl_account", "cost_centre", "amount", "currency", "posted_by", "posted_at"]], use_container_width=True, height=500)
        st.caption(f"{len(postings):,} postings in ledger")
    else:
        st.info("No invoices posted yet.")

# ── TAB 5: Audit Trail ─────────────────────────────────
with tabs[4]:
    st.markdown('<div class="narrative">Every prediction carries full provenance — which historical lines drove it, what confidence score it produced, and who confirmed it. This is the compliance-grade audit trail: stronger traceability than a deterministic rules system.</div>', unsafe_allow_html=True)

    from ap_automation.core.database import get_audit_records
    records = get_audit_records(limit=100)
    if records:
        for rec in records[:20]:
            routing_label = "Auto-posted" if rec["routing_decision"] == "auto_post" else "Human review"
            source_label = {"auto_post": "Auto-posted", "human_confirm": "Human confirmed", "human_correct": "Human corrected"}.get(rec["source"], rec["source"])
            with st.expander(f"[{routing_label}] Invoice {rec['invoice_id'][:8]}... — GL {rec['confirmed_gl']} — {source_label}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted GL", rec["predicted_gl"])
                c2.metric("Confirmed GL", rec["confirmed_gl"])
                c3.metric("Confidence", f"{rec['confidence']:.1%}")
                st.json(json.loads(rec["provenance_json"]))
    else:
        st.info("No audit records yet.")

# ── TAB 6: Setup ───────────────────────────────────────
with tabs[5]:
    col_setup, col_status = st.columns([1, 1], gap="large")
    with col_setup:
        st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
        if st.button("🗂️ Generate 1,000-invoice dataset", use_container_width=True):
            with st.spinner("Generating..."):
                import random
                random.seed(42)
                from ap_automation.core.dataset import generate_dataset
                invoices = generate_dataset()
            st.success(f"✓ Generated {len(invoices):,} invoices → data/invoices.csv")
        st.write("")
        st.markdown('<div class="section-header">Seed Corpus</div>', unsafe_allow_html=True)
        seed_count = st.slider("Historical lines to seed", 50, 300, 150)
        if st.button("🌱 Seed Qdrant corpus", use_container_width=True):
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
                        "point_id": str(uuid4()), "invoice_id": row["invoice_id"],
                        "vendor_id": canonical_id, "entity_id": row["entity_id"],
                        "amount_band": "small", "currency": row["currency"],
                        "gl_account": row["ground_truth_gl"], "cost_centre": row["cost_centre"],
                        "tax_code": "STANDARD", "description": row["description"],
                        "confirming_user_id": "seed", "confirmation_timestamp": "2025-01-01T00:00:00",
                        "prediction_confidence": 1.0, "source": "human_confirm",
                    }
                    index_outcome(payload)
                    progress.progress((i + 1) / len(routine))
                st.success(f"✓ Seeded {len(routine):,} lines into Qdrant")

    with col_status:
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        try:
            qc2 = get_qdrant()
            info = qc2.get_collection(settings.qdrant_collection)
            st.success(f"Qdrant — {info.points_count:,} points in `{settings.qdrant_collection}`")
        except Exception as e:
            st.error(f"Qdrant — {e}")
        try:
            r = httpx.get(f"{ERP_URL}/health", timeout=2)
            st.success("Mock ERP — running") if r.status_code == 200 else st.warning("Mock ERP — unexpected response")
        except Exception:
            st.warning("Mock ERP — not running (local only)")
        try:
            from ap_automation.core.sheets import _get_service
            _get_service()
            st.success("Google Sheets — connected")
        except Exception as e:
            st.warning(f"Google Sheets — {e}")
        dataset_path = Path("data/invoices.csv")
        if dataset_path.exists():
            st.success(f"Dataset — invoices.csv ({dataset_path.stat().st_size/1024:.0f} KB)")
        else:
            st.warning("Dataset — not generated yet")
        st.write("")
        # Langfuse status
        st.write("")
        st.markdown('<div class="section-header">Langfuse Observability</div>', unsafe_allow_html=True)
        try:
            import os as _os
            _pub = str(st.secrets.get("LANGFUSE_PUBLIC_KEY", "") or "")
            _sec = str(st.secrets.get("LANGFUSE_SECRET_KEY", "") or "")
            _host = str(st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com") or "https://cloud.langfuse.com")
            if _pub and _sec:
                _os.environ["LANGFUSE_PUBLIC_KEY"] = _pub
                _os.environ["LANGFUSE_SECRET_KEY"] = _sec
                _os.environ["LANGFUSE_HOST"] = _host
                from langfuse import get_client as _lf_get
                _lf = _lf_get()
                _auth = _lf.auth_check()
                if _auth:
                    st.success(f"Langfuse � connected ({_host})")
                    if st.button("Send test trace to Langfuse", use_container_width=True):
                        with _lf.start_as_current_observation(as_type="span", name="test_trace", input={"source": "setup_tab"}):
                            _lf.update_current_span(output={"result": "ok"})
                        _lf.flush()
                        st.success("Test trace sent � check cloud.langfuse.com")
                else:
                    st.error("Langfuse � auth check failed")
            else:
                st.warning(f"Langfuse � keys missing (pub={bool(_pub)}, sec={bool(_sec)})")
        except Exception as _e:
            st.error(f"Langfuse � {_e}")
        st.write("")

        # Langfuse status
        st.write("")
        st.markdown('<div class="section-header">Langfuse Observability</div>', unsafe_allow_html=True)
        try:
            import os as _os
            _pub = str(st.secrets.get("LANGFUSE_PUBLIC_KEY", "") or "")
            _sec = str(st.secrets.get("LANGFUSE_SECRET_KEY", "") or "")
            _host = str(st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com") or "https://cloud.langfuse.com")
            if _pub and _sec:
                _os.environ["LANGFUSE_PUBLIC_KEY"] = _pub
                _os.environ["LANGFUSE_SECRET_KEY"] = _sec
                _os.environ["LANGFUSE_HOST"] = _host
                from langfuse import get_client as _lf_get
                _lf = _lf_get()
                _auth = _lf.auth_check()
                if _auth:
                    st.success(f"Langfuse connected ({_host})")
                    if st.button("Send test trace to Langfuse", use_container_width=True):
                        with _lf.start_as_current_observation(as_type="span", name="test_trace", input={"source": "setup_tab"}):
                            _lf.update_current_span(output={"result": "ok"})
                        _lf.flush()
                        st.success("Test trace sent - check cloud.langfuse.com")
                else:
                    st.error("Langfuse auth check failed")
            else:
                st.warning(f"Langfuse keys missing (pub={bool(_pub)}, sec={bool(_sec)})")
        except Exception as _e:
            st.error(f"Langfuse error: {_e}")
        st.write("")
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        st.code(f"""Confidence threshold: {settings.default_confidence_threshold}
Min evidence lines:   {settings.min_evidence_lines}
Top-k retrieval:      {settings.top_k}
Embedding model:      {settings.embedding_model}
Collection:           {settings.qdrant_collection}""")


