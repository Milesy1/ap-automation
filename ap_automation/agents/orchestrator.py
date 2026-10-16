"""
A2A Agent Orchestrator.
Holds a registry of specialist agents, calls all relevant ones in parallel,
and merges results into a single AgentOrchestrationResult.
Each agent call is traced as a child span in Langfuse.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ap_automation.agents.base import A2AAgent, AgentResult
from ap_automation.agents.vat_agent import VATAgent
from ap_automation.agents.cost_centre_agent import CostCentreAgent
from ap_automation.agents.entity_agent import EntityAgent


class AgentOrchestrationResult:
    """Merged result from all agent calls."""

    def __init__(self, results: list[AgentResult]):
        self.results = results
        self.by_agent = {r.agent_name: r for r in results}

    @property
    def vat(self) -> AgentResult | None:
        return self.by_agent.get("vat_agent")

    @property
    def cost_centre(self) -> AgentResult | None:
        return self.by_agent.get("cost_centre_agent")

    @property
    def entity(self) -> AgentResult | None:
        return self.by_agent.get("entity_agent")

    @property
    def all_successful(self) -> bool:
        return all(r.success for r in self.results)

    def to_dict(self) -> dict:
        return {
            "agents_called": len(self.results),
            "all_successful": self.all_successful,
            "results": [r.to_dict() for r in self.results],
        }


class AgentOrchestrator:
    """
    Discovers and calls all registered A2A agents in parallel.
    Safe to use alongside the existing pipeline — never raises,
    never modifies GL prediction or routing decision.
    """

    def __init__(self):
        self._registry: list[A2AAgent] = [
            VATAgent(),
            CostCentreAgent(),
            EntityAgent(),
        ]

    def register(self, agent: A2AAgent) -> None:
        """Register a new specialist agent at runtime."""
        self._registry.append(agent)

    def agent_cards(self) -> list[dict]:
        """Return AgentCards for all registered agents."""
        return [a.agent_card() for a in self._registry]

    def run(
        self,
        context: dict[str, Any],
        langfuse_client=None,
    ) -> AgentOrchestrationResult:
        """
        Call all relevant agents in parallel.
        Each agent that returns can_handle(context) = True is called.
        Results are collected, Langfuse child spans created per agent.
        Never raises — all exceptions caught per agent.
        """
        eligible = [a for a in self._registry if a.can_handle(context)]

        if not eligible:
            return AgentOrchestrationResult([])

        results: list[AgentResult] = []

        with ThreadPoolExecutor(max_workers=len(eligible)) as executor:
            futures = {
                executor.submit(self._call_agent, agent, context): agent
                for agent in eligible
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Langfuse child span per agent
                if langfuse_client:
                    try:
                        with langfuse_client.start_as_current_observation(
                            as_type="span",
                            name=f"a2a.{result.agent_name}",
                            input=context,
                            metadata={
                                "agent": result.agent_name,
                                "success": result.success,
                                "latency_ms": result.latency_ms,
                            },
                        ):
                            langfuse_client.update_current_span(
                                output=result.to_dict()
                            )
                    except Exception:
                        pass  # never let observability break the pipeline

        return AgentOrchestrationResult(results)

    @staticmethod
    def _call_agent(agent: A2AAgent, context: dict[str, Any]) -> AgentResult:
        """Call a single agent — catches all exceptions."""
        t0 = time.time()
        try:
            result = agent.run(context)
            result.latency_ms = int((time.time() - t0) * 1000)
            return result
        except Exception as e:
            return AgentResult(
                agent_name=agent.name,
                success=False,
                error=str(e),
                latency_ms=int((time.time() - t0) * 1000),
            )
