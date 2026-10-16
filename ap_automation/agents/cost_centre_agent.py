"""
Cost Centre Specialist Agent.
Infers the correct cost centre from vendor, GL code, and entity
where cost centre is missing or ambiguous.
Deterministic rules — no LLM required.
"""
from __future__ import annotations

from ap_automation.agents.base import A2AAgent, AgentResult

# GL code -> default cost centre mapping
# In production this would be loaded from the chart of accounts / tenant config
GL_COST_CENTRE_MAP: dict[str, str] = {
    "6300": "FACILITIES",
    "6301": "FACILITIES",
    "6302": "FACILITIES",
    "6303": "FACILITIES",
    "6400": "TRAVEL",
    "6401": "TRAVEL",
    "6402": "TRAVEL",
    "6500": "PROFESSIONAL",
    "6501": "PROFESSIONAL",
    "6502": "PROFESSIONAL",
    "6600": "IT",
    "6601": "IT",
    "6602": "IT",
    "6700": "MARKETING",
    "6701": "MARKETING",
    "6800": "GENERAL",
    "6900": "HR",
    "6901": "HR",
    "7000": "HR",
    "7001": "HR",
    "7100": "FACILITIES",
    "7101": "FACILITIES",
}


class CostCentreAgent(A2AAgent):
    """
    Cost centre inference specialist.
    Runs when cost centre is missing or when GL code implies a specific cost centre
    that differs from the one provided.
    """

    @property
    def name(self) -> str:
        return "cost_centre_agent"

    @property
    def description(self) -> str:
        return (
            "Infers or validates the cost centre for an invoice based on GL code "
            "and vendor. Flags mismatches between predicted GL and cost centre."
        )

    def input_fields(self) -> list[str]:
        return ["predicted_gl", "predicted_cost_centre", "vendor_id", "entity_id"]

    def output_fields(self) -> list[str]:
        return ["inferred_cost_centre", "cost_centre_mismatch", "cost_centre_note"]

    def can_handle(self, context: dict) -> bool:
        # Run when cost centre is missing or when we have a GL mapping to check against
        gl = context.get("predicted_gl", "")
        return gl in GL_COST_CENTRE_MAP

    def run(self, context: dict) -> AgentResult:
        try:
            gl = context.get("predicted_gl", "")
            provided_cc = context.get("predicted_cost_centre", "") or ""

            inferred_cc = GL_COST_CENTRE_MAP.get(gl, "GENERAL")
            mismatch = (
                bool(provided_cc)
                and provided_cc.upper() != inferred_cc.upper()
            )

            if not provided_cc:
                reasoning = f"Cost centre missing -- inferred {inferred_cc} from GL {gl}."
            elif mismatch:
                reasoning = (
                    f"Cost centre mismatch: provided {provided_cc}, "
                    f"GL {gl} maps to {inferred_cc}. Review recommended."
                )
            else:
                reasoning = f"Cost centre {provided_cc} consistent with GL {gl}."

            return AgentResult(
                agent_name=self.name,
                success=True,
                confidence=0.85,
                reasoning=reasoning,
                data={
                    "inferred_cost_centre": inferred_cc,
                    "provided_cost_centre": provided_cc,
                    "cost_centre_mismatch": mismatch,
                    "cost_centre_note": reasoning,
                },
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )
