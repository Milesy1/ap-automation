"""
Entity Specialist Agent.
Validates and infers the correct legal entity for an invoice
in multi-entity deployments. Flags entity mismatches.
Deterministic rules — no LLM required.
"""
from __future__ import annotations

from ap_automation.agents.base import A2AAgent, AgentResult

# Vendor -> expected entity mapping
# In production this would be loaded from tenant config / MDM
VENDOR_ENTITY_MAP: dict[str, str] = {
    "MERIDIAN_FACILITIES": "UK001",
    "SWIFT_TRAVEL": "UK001",
    "APEX_EQUIPMENT": "UK001",
    "VERTEX_CONSULTING": "UK001",
    "NOVA_SUPPLIES": "UK001",
    "PINNACLE_SERVICES": "UK001",
    "CREST_SOLUTIONS": "UK001",
    "ATLAS_RESOURCES": "UK001",
    "SUMMIT_PARTNERS": "UK001",
    "HORIZON_TECH": "UK001",
}

# GL codes that are entity-specific
ENTITY_SPECIFIC_GL: set[str] = {
    "7000", "7001",  # payroll -- always entity-specific
    "7100", "7101",  # construction -- entity matters for CIS
}


class EntityAgent(A2AAgent):
    """
    Legal entity validation specialist.
    Validates that the invoice is being posted to the correct legal entity.
    Flags entity mismatches and entity-sensitive GL codes.
    """

    @property
    def name(self) -> str:
        return "entity_agent"

    @property
    def description(self) -> str:
        return (
            "Validates the legal entity for an invoice in multi-entity deployments. "
            "Flags entity mismatches and entity-sensitive GL codes (payroll, CIS)."
        )

    def input_fields(self) -> list[str]:
        return ["vendor_id", "entity_id", "predicted_gl"]

    def output_fields(self) -> list[str]:
        return ["entity_valid", "expected_entity", "entity_sensitive", "entity_note"]

    def can_handle(self, context: dict) -> bool:
        vendor_id = context.get("vendor_id", "")
        gl = context.get("predicted_gl", "")
        # Run if vendor is in our map or GL is entity-sensitive
        return vendor_id in VENDOR_ENTITY_MAP or gl in ENTITY_SPECIFIC_GL

    def run(self, context: dict) -> AgentResult:
        try:
            vendor_id = context.get("vendor_id", "")
            entity_id = context.get("entity_id", "") or ""
            gl = context.get("predicted_gl", "")

            expected_entity = VENDOR_ENTITY_MAP.get(vendor_id)
            entity_sensitive = gl in ENTITY_SPECIFIC_GL

            if expected_entity and entity_id:
                entity_valid = entity_id.upper() == expected_entity.upper()
                if entity_valid:
                    reasoning = f"Entity {entity_id} matches expected entity for {vendor_id}."
                else:
                    reasoning = (
                        f"Entity mismatch: invoice on {entity_id}, "
                        f"vendor {vendor_id} expected on {expected_entity}. Review required."
                    )
            elif expected_entity and not entity_id:
                entity_valid = False
                reasoning = f"Entity missing -- {vendor_id} should be on {expected_entity}."
            else:
                entity_valid = True
                reasoning = "Entity not in vendor map -- no validation possible."

            if entity_sensitive:
                reasoning += f" GL {gl} is entity-sensitive -- confirm entity before posting."

            return AgentResult(
                agent_name=self.name,
                success=True,
                confidence=0.90 if expected_entity else 0.60,
                reasoning=reasoning,
                data={
                    "entity_valid": entity_valid,
                    "provided_entity": entity_id,
                    "expected_entity": expected_entity or "unknown",
                    "entity_sensitive": entity_sensitive,
                    "entity_note": reasoning,
                },
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )
