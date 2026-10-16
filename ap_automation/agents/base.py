"""
A2A Agent base class and AgentResult model.
Every specialist agent inherits from A2AAgent and implements can_handle() and run().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Structured result from a specialist agent call."""
    agent_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""
    error: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "success": self.success,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "error": self.error,
            **self.data,
        }


class A2AAgent(ABC):
    """
    Abstract base class for all A2A specialist agents.

    Each agent:
    - Publishes a name and description (AgentCard equivalent)
    - Implements can_handle() to declare when it is relevant
    - Implements run() to produce an AgentResult
    - Is stateless — all context passed via run()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Plain-English description of what this agent does."""
        ...

    @property
    def version(self) -> str:
        return "1.0.0"

    def agent_card(self) -> dict:
        """A2A AgentCard — describes this agent's capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_fields": self.input_fields(),
            "output_fields": self.output_fields(),
        }

    def input_fields(self) -> list[str]:
        """Fields this agent reads from the context dict."""
        return []

    def output_fields(self) -> list[str]:
        """Fields this agent adds to the result."""
        return []

    @abstractmethod
    def can_handle(self, context: dict[str, Any]) -> bool:
        """
        Return True if this agent should run for this context.
        Agents that cannot contribute return False — orchestrator skips them.
        """
        ...

    @abstractmethod
    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Execute the agent and return a structured result.
        Must never raise — catch all exceptions and return AgentResult(success=False).
        """
        ...
