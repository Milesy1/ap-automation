"""
LangGraph visualiser tab for the Streamlit UI.
Shows the invoice graph as a diagram, with the active node highlighted,
state at each step, agent results inline, and final routing decision.
"""
from __future__ import annotations

import streamlit as st


# Node display config: order, label, colour, description
_NODES = [
    ("mdm",          "MDM Resolution",     "#6366f1", "Resolves vendor canonical ID and amount band"),
    ("retrieval",    "Retrieval",          "#8b5cf6", "Fetches evidence from Qdrant via hybrid search"),
    ("confidence",   "Confidence Scoring", "#a855f7", "Scores evidence, predicts GL code"),
    ("routing",      "Base Routing",       "#ec4899", "Confidence-based routing decision"),
    ("agents",       "A2A Agents",         "#f59e0b", "VAT, Cost Centre, Entity agents run in parallel"),
    ("gate",         "Rule Gate",          "#ef4444", "Applies agent flags — governance beats confidence"),
    ("llm_gate",     "LLM Gate",           "#dc2626", "Claude reviews context and reasons over the decision"),
    ("auto_post",    "Auto Post",          "#10b981", "Posts to ERP automatically"),
    ("human_review", "Human Review",       "#f97316", "Parked for human review"),
    ("confirmation", "Confirmation",       "#3b82f6", "Human confirms or corrects the prediction"),
    ("write_back",   "Write-back",         "#14b8a6", "Indexes confirmed outcome into Qdrant"),
]


def render_graph_tab(graph_state: dict | None = None) -> None:
    st.markdown("## Invoice Processing Graph")
    st.caption(
        "Each node does one thing. State flows left to right. "
        "The active path is highlighted based on the last prediction."
    )

    if graph_state is None:
        st.info("Submit an invoice to see the graph execute.")
        _render_static_graph()
        return

    _render_live_graph(graph_state)
    _render_state_breakdown(graph_state)
    _render_llm_reasoning(graph_state)
    _render_agent_results(graph_state)
    _render_routing_summary(graph_state)


def _active_nodes(state: dict) -> list[str]:
    active = []
    if state.get("canonical_vendor_id"):
        active.append("mdm")
    if state.get("evidence_count") is not None:
        active.append("retrieval")
    if state.get("predicted_gl") is not None:
        active.append("confidence")
    if state.get("routing_decision"):
        active.append("routing")
    if state.get("agents_called") is not None:
        active.append("agents")
    if state.get("gate_reason"):
        active.append("gate")
    if state.get("llm_reasoning") is not None:
        active.append("llm_gate")
    if state.get("erp_posted"):
        active.append("auto_post")
    elif state.get("review_queued_at"):
        active.append("human_review")
    if state.get("confirmed_gl") and not state.get("erp_posted"):
        active.append("confirmation")
    if state.get("write_back_complete"):
        active.append("write_back")
    return active


def _render_static_graph() -> None:
    cols = st.columns(len(_NODES))
    for i, (node_id, label, colour, desc) in enumerate(_NODES):
        with cols[i]:
            st.markdown(
                f"""<div style="background:{colour}22;border:2px solid {colour}66;border-radius:8px;
                padding:8px 4px;text-align:center;min-height:80px;display:flex;
                flex-direction:column;justify-content:center;">
                <div style="font-size:10px;font-weight:600;color:{colour};">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_live_graph(state: dict) -> None:
    active = _active_nodes(state)
    final_routing = state.get("final_routing", "")

    cols = st.columns(len(_NODES))
    for i, (node_id, label, colour, desc) in enumerate(_NODES):
        is_active = node_id in active
        is_skipped = (
            (node_id == "auto_post" and final_routing != "auto_post") or
            (node_id in ("human_review", "confirmation") and final_routing == "auto_post")
        )
        with cols[i]:
            if is_active:
                border = f"3px solid {colour}"
                bg = f"{colour}33"
                opacity = "1"
                tick = "✓ "
            elif is_skipped:
                border = "2px dashed #64748b44"
                bg = "#64748b11"
                opacity = "0.35"
                tick = ""
            else:
                border = f"2px solid {colour}44"
                bg = f"{colour}11"
                opacity = "0.55"
                tick = ""

            escalated = node_id == "llm_gate" and state.get("llm_escalated")
            badge = " ⚡" if escalated else ""

            st.markdown(
                f"""<div style="background:{bg};border:{border};border-radius:8px;
                padding:8px 4px;text-align:center;min-height:80px;display:flex;
                flex-direction:column;justify-content:center;opacity:{opacity};"
                title="{desc}">
                <div style="font-size:10px;font-weight:700;color:{colour};">{tick}{label}{badge}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)


def _render_state_breakdown(state: dict) -> None:
    st.markdown("### State at each node")

    node_states = {
        "mdm": {
            "canonical_vendor_id": state.get("canonical_vendor_id"),
            "mdm_confidence": state.get("mdm_confidence"),
            "amount_band": state.get("amount_band"),
            "latency_ms": state.get("mdm_latency_ms"),
        },
        "retrieval": {
            "evidence_count": state.get("evidence_count"),
            "latency_ms": state.get("retrieval_latency_ms"),
        },
        "confidence": {
            "predicted_gl": state.get("predicted_gl"),
            "predicted_cost_centre": state.get("predicted_cost_centre"),
            "agreement_ratio": state.get("agreement_ratio"),
            "weighted_confidence": state.get("weighted_confidence"),
            "latency_ms": state.get("confidence_latency_ms"),
        },
        "routing": {
            "routing_decision": state.get("routing_decision"),
            "routing_reason": state.get("routing_reason"),
        },
        "agents": {
            "agents_called": state.get("agents_called"),
            "vat_treatment": state.get("vat_treatment"),
            "cost_centre_mismatch": state.get("cost_centre_mismatch"),
            "entity_valid": state.get("entity_valid"),
            "latency_ms": state.get("agent_latency_ms"),
        },
        "gate": {
            "final_routing": state.get("final_routing"),
            "gate_reason": state.get("gate_reason"),
            "agent_flags": state.get("agent_flags", []),
        },
    }

    active = _active_nodes(state)
    for node_id, label, colour, desc in _NODES:
        if node_id not in node_states or node_id not in active:
            continue
        with st.expander(f"{label}", expanded=(node_id in ("gate", "llm_gate"))):
            s = node_states[node_id]
            cols = st.columns(2)
            items = [(k, v) for k, v in s.items() if v is not None]
            for j, (k, v) in enumerate(items):
                with cols[j % 2]:
                    if isinstance(v, float):
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                    elif isinstance(v, list):
                        st.markdown(f"**{k.replace('_', ' ').title()}**")
                        if v:
                            for item in v:
                                st.markdown(f"- `{item}`")
                        else:
                            st.markdown("_none_")
                    else:
                        st.metric(k.replace("_", " ").title(), str(v))


def _render_llm_reasoning(state: dict) -> None:
    """Render the LLM gate reasoning prominently."""
    reasoning = state.get("llm_reasoning")
    if not reasoning:
        return

    st.markdown("### LLM Gate Reasoning")

    escalated = state.get("llm_escalated", False)
    llm_confidence = state.get("llm_reasoning_confidence", 0.0)
    llm_latency = state.get("llm_latency_ms", 0)

    if escalated:
        st.error("Claude escalated this invoice to human review — a risk was identified that the rule gate missed.")
    else:
        final = state.get("final_routing", "")
        if final == "auto_post":
            st.success("Claude confirmed the auto-post decision.")
        else:
            st.warning("Claude agreed with the rule gate: human review required.")

    st.markdown(
        f"""<div style="background:#1e293b;border-left:4px solid #dc2626;border-radius:6px;
        padding:16px 20px;margin:8px 0;">
        <div style="color:#f1f5f9;font-size:14px;line-height:1.6;">{reasoning}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        st.metric("LLM Confidence", f"{llm_confidence:.2f}")
    with cols[1]:
        st.metric("Escalated", "Yes" if escalated else "No")
    with cols[2]:
        st.metric("LLM Latency", f"{llm_latency} ms")


def _render_agent_results(state: dict) -> None:
    results = state.get("agent_results", [])
    if not results:
        return

    st.markdown("### A2A Agent Results")
    for r in results:
        agent = r.get("agent", r.get("agent_name", "unknown"))
        success = r.get("success", False)
        label = "OK" if success else "FAILED"
        with st.expander(f"{agent}  [{label}]", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Status", label)
            with cols[1]:
                st.metric("Confidence", f"{r.get('confidence', 0):.2f}")
            with cols[2]:
                st.metric("Latency", f"{r.get('latency_ms', 0)} ms")
            if r.get("reasoning"):
                st.caption(r["reasoning"])
            extra = {k: v for k, v in r.items() if k not in ("agent", "agent_name", "success", "confidence", "reasoning", "latency_ms", "error")}
            if extra:
                st.json(extra)


def _render_routing_summary(state: dict) -> None:
    final = state.get("final_routing")
    if not final:
        return

    st.markdown("### Final Routing Decision")
    flags = state.get("agent_flags", [])
    escalated = state.get("llm_escalated", False)

    if final == "auto_post":
        st.success(f"AUTO POST — {state.get('gate_reason', '')}")
    else:
        if escalated:
            st.error(f"HUMAN REVIEW — Claude escalated: {state.get('llm_reasoning', '')[:120]}...")
        elif flags:
            st.warning(f"HUMAN REVIEW — Agent flags: {', '.join(flags)}")
        else:
            st.warning(f"HUMAN REVIEW — {state.get('gate_reason', '')}")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Predicted GL", state.get("predicted_gl", "N/A"))
    with cols[1]:
        st.metric("Confidence", f"{state.get('weighted_confidence', 0):.2f}")
    with cols[2]:
        st.metric("Evidence", state.get("evidence_count", 0))
    with cols[3]:
        ms = state.get("total_latency_ms")
        st.metric("Total Latency", f"{ms} ms" if ms else "N/A")
