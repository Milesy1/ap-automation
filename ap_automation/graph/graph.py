"""
LangGraph invoice pipeline graph.
Builds and compiles the StateGraph that orchestrates the full invoice lifecycle.
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
    llm_gate_node,
    auto_post_node,
    human_review_node,
    confirmation_node,
    write_back_node,
    route_after_gate,
)


def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the invoice processing StateGraph.

    Pipeline: mdm → retrieval → confidence → routing → agents → gate → llm_gate
    Conditional after llm_gate: auto_post or human_review
    auto_post → write_back → END
    human_review → (interrupt) → confirmation → write_back → END
    """
    graph = StateGraph(InvoiceState)

    graph.add_node("mdm", mdm_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agents", agents_node)
    graph.add_node("gate", gate_node)
    graph.add_node("llm_gate", llm_gate_node)
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
    graph.add_edge("gate", "llm_gate")

    graph.add_conditional_edges(
        "llm_gate",
        route_after_gate,
        {
            "auto_post": "auto_post",
            "human_review": "human_review",
        },
    )

    graph.add_edge("auto_post", "write_back")
    graph.add_edge("human_review", END)
    graph.add_edge("confirmation", "write_back")
    graph.add_edge("write_back", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["confirmation"],
    )


def build_graph_no_interrupt(checkpointer=None) -> StateGraph:
    """POC variant without interrupt — full graph end-to-end."""
    graph = StateGraph(InvoiceState)

    graph.add_node("mdm", mdm_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agents", agents_node)
    graph.add_node("gate", gate_node)
    graph.add_node("llm_gate", llm_gate_node)
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
    graph.add_edge("gate", "llm_gate")

    graph.add_conditional_edges(
        "llm_gate",
        route_after_gate,
        {
            "auto_post": "auto_post",
            "human_review": "human_review",
        },
    )

    graph.add_edge("auto_post", "write_back")
    graph.add_edge("human_review", "confirmation")
    graph.add_edge("confirmation", "write_back")
    graph.add_edge("write_back", END)

    return graph.compile(checkpointer=checkpointer)
