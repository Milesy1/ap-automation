"""
LangGraph invoice pipeline graph.
Builds and compiles the StateGraph that orchestrates the full invoice lifecycle.

Usage:
    from ap_automation.graph.graph import build_graph
    graph = build_graph()
    result = graph.invoke(initial_state)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from ap_automation.graph.state import InvoiceState
from ap_automation.graph.nodes import (
    mdm_node,
    retrieval_node,
    confidence_node,
    routing_node,
    agents_node,
    gate_node,
    auto_post_node,
    human_review_node,
    confirmation_node,
    write_back_node,
    route_after_gate,
)


def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the invoice processing StateGraph.

    Nodes:
        mdm          → resolves vendor and amount band
        retrieval    → fetches evidence from Qdrant
        confidence   → scores evidence, predicts GL
        routing      → base routing decision from confidence
        agents       → runs A2A specialist agents in parallel
        gate         → applies agent flags, finalises routing
        auto_post    → posts to ERP automatically
        human_review → parks invoice for human review (interrupt point)
        confirmation → receives human decision
        write_back   → indexes confirmed outcome into Qdrant

    Edges:
        Linear: mdm → retrieval → confidence → routing → agents → gate
        Conditional after gate: auto_post or human_review
        auto_post → write_back → END
        human_review → (interrupt) → confirmation → write_back → END
    """
    graph = StateGraph(InvoiceState)

    # ── Register nodes ────────────────────────────────
    graph.add_node("mdm", mdm_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agents", agents_node)
    graph.add_node("gate", gate_node)
    graph.add_node("auto_post", auto_post_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("write_back", write_back_node)

    # ── Linear edges ──────────────────────────────────
    graph.set_entry_point("mdm")
    graph.add_edge("mdm", "retrieval")
    graph.add_edge("retrieval", "confidence")
    graph.add_edge("confidence", "routing")
    graph.add_edge("routing", "agents")
    graph.add_edge("agents", "gate")

    # ── Conditional edge after gate ───────────────────
    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "auto_post": "auto_post",
            "human_review": "human_review",
        },
    )

    # ── Post-routing edges ────────────────────────────
    graph.add_edge("auto_post", "write_back")
    graph.add_edge("human_review", END)          # graph pauses here; resumes on confirmation
    graph.add_edge("confirmation", "write_back")
    graph.add_edge("write_back", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["confirmation"],       # pause before confirmation for human input
    )


def build_graph_no_interrupt(checkpointer=None) -> StateGraph:
    """
    Variant without interrupt — used in the POC/Streamlit where the UI handles
    human review synchronously. Runs the full graph end-to-end.
    """
    graph = StateGraph(InvoiceState)

    graph.add_node("mdm", mdm_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agents", agents_node)
    graph.add_node("gate", gate_node)
    graph.add_node("auto_post", auto_post_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("confirmation", confirmation_node)
    graph.add_node("write_back", write_back_node)

    graph.set_entry_point("mdm")
    graph.add_edge("mdm", "retrieval")
    graph.add_edge("retrieval", "confidence")
    graph.add_edge("confidence", "routing")
    graph.add_edge("routing", "agents")
    graph.add_edge("agents", "gate")

    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "auto_post": "auto_post",
            "human_review": "human_review",
        },
    )

    graph.add_edge("auto_post", "write_back")
    graph.add_edge("human_review", "confirmation")   # no interrupt — flows straight through
    graph.add_edge("confirmation", "write_back")
    graph.add_edge("write_back", END)

    return graph.compile(checkpointer=checkpointer)
