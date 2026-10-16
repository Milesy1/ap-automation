"""
VAT Specialist Agent.
Determines UK VAT treatment for an invoice based on GL code,
vendor type, amount, and description.
Deterministic rules — no LLM required.

VAT treatments:
- STANDARD (20%): most business expenses
- REDUCED (5%): domestic fuel, energy, some construction
- ZERO (0%): food, books, children's clothing, exports
- EXEMPT: insurance, finance, education, postage
- OUTSIDE_SCOPE: wages, rates, dividends
- REVERSE_CHARGE: construction services (CIS), overseas digital services
- BLOCKED: entertainment, business entertainment, client hospitality
"""
from __future__ import annotations

from ap_automation.agents.base import A2AAgent, AgentResult

# GL code -> VAT treatment mapping
# Based on common UK chart of accounts patterns
GL_VAT_MAP: dict[str, tuple[str, str]] = {
    # Facilities / premises
    "6300": ("STANDARD", "Facilities services -- standard rated"),
    "6301": ("STANDARD", "Cleaning services -- standard rated"),
    "6302": ("STANDARD", "Maintenance services -- standard rated"),
    "6303": ("REDUCED", "Energy/utilities -- reduced rate applies"),
    # Travel
    "6400": ("STANDARD", "Travel and subsistence -- standard rated"),
    "6401": ("ZERO", "Rail travel -- zero rated"),
    "6402": ("ZERO", "Air travel (domestic) -- zero rated"),
    "6403": ("EXEMPT", "Insurance on travel -- exempt"),
    # Professional services
    "6500": ("STANDARD", "Professional services -- standard rated"),
    "6501": ("STANDARD", "Legal services -- standard rated"),
    "6502": ("STANDARD", "Consultancy -- standard rated"),
    "6503": ("EXEMPT", "Financial services -- exempt"),
    # IT and software
    "6600": ("STANDARD", "IT services -- standard rated"),
    "6601": ("STANDARD", "Software licences -- standard rated"),
    "6602": ("STANDARD", "Hardware -- standard rated"),
    # Marketing
    "6700": ("STANDARD", "Marketing and advertising -- standard rated"),
    "6701": ("ZERO", "Printed materials (books/leaflets) -- zero rated"),
    # Entertainment -- BLOCKED input VAT
    "6800": ("BLOCKED", "Business entertainment -- input VAT blocked"),
    "6801": ("BLOCKED", "Client hospitality -- input VAT blocked"),
    "6802": ("BLOCKED", "Staff entertainment -- input VAT blocked"),
    # Training
    "6900": ("STANDARD", "Training -- standard rated"),
    "6901": ("EXEMPT", "Education from eligible body -- exempt"),
    # Payroll / HR (outside scope)
    "7000": ("OUTSIDE_SCOPE", "Payroll costs -- outside scope of VAT"),
    "7001": ("OUTSIDE_SCOPE", "Wages and salaries -- outside scope"),
    # Construction (reverse charge applies)
    "7100": ("REVERSE_CHARGE", "Construction services -- domestic reverse charge (CIS)"),
    "7101": ("REVERSE_CHARGE", "Building works -- domestic reverse charge (CIS)"),
}

# Description keywords that signal specific VAT treatments
DESCRIPTION_SIGNALS: list[tuple[list[str], str, str]] = [
    (["entertainment", "hospitality", "client dinner", "team lunch", "christmas party"],
     "BLOCKED", "Entertainment keyword detected -- input VAT blocked"),
    (["construction", "cis", "building work", "scaffolding", "groundwork"],
     "REVERSE_CHARGE", "Construction keyword detected -- domestic reverse charge"),
    (["insurance"], "EXEMPT", "Insurance keyword -- exempt"),
    (["postage", "royal mail", "stamps"], "EXEMPT", "Postage -- exempt"),
    (["salary", "wages", "payroll", "paye"], "OUTSIDE_SCOPE", "Payroll -- outside scope"),
    (["export", "overseas", "eu ", "international"], "ZERO", "Overseas supply -- zero rated"),
]

# Amount thresholds
HIGH_VALUE_THRESHOLD = 25_000.0  # flag for review regardless


class VATAgent(A2AAgent):
    """
    UK VAT treatment specialist.
    Determines VAT code, blocks entertainment, flags reverse charge.
    """

    @property
    def name(self) -> str:
        return "vat_agent"

    @property
    def description(self) -> str:
        return (
            "Determines UK VAT treatment for an invoice. "
            "Detects standard, zero, exempt, blocked, reverse charge, and outside scope. "
            "Flags high-value invoices for VAT review."
        )

    def input_fields(self) -> list[str]:
        return ["predicted_gl", "description", "amount", "currency", "vendor_id"]

    def output_fields(self) -> list[str]:
        return ["vat_treatment", "vat_rate", "vat_flag", "review_required", "vat_note"]

    def can_handle(self, context: dict) -> bool:
        # Always run for GBP invoices; skip non-GBP (FX agent handles those)
        return context.get("currency", "GBP") == "GBP"

    def run(self, context: dict) -> AgentResult:
        try:
            gl = context.get("predicted_gl", "")
            description = (context.get("description", "") or "").lower()
            amount = float(context.get("amount", 0.0))
            vendor_id = context.get("vendor_id", "")

            vat_treatment, reasoning, review_required = self._classify(
                gl, description, amount
            )

            vat_rate = self._treatment_to_rate(vat_treatment)
            vat_flag = vat_treatment in ("BLOCKED", "REVERSE_CHARGE", "OUTSIDE_SCOPE")

            # High-value flag
            if amount >= HIGH_VALUE_THRESHOLD and vat_treatment == "STANDARD":
                review_required = True
                reasoning += f" High-value invoice (£{amount:,.2f}) -- VAT review recommended."

            return AgentResult(
                agent_name=self.name,
                success=True,
                confidence=0.90 if gl in GL_VAT_MAP else 0.70,
                reasoning=reasoning,
                data={
                    "vat_treatment": vat_treatment,
                    "vat_rate": vat_rate,
                    "vat_flag": vat_flag,
                    "review_required": review_required,
                    "vat_note": reasoning,
                    "gl_code": gl,
                },
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _classify(
        self, gl: str, description: str, amount: float
    ) -> tuple[str, str, bool]:
        """Return (vat_treatment, reasoning, review_required)."""

        # 1. Description signals take priority (catches entertainment regardless of GL)
        for keywords, treatment, reason in DESCRIPTION_SIGNALS:
            if any(kw in description for kw in keywords):
                return treatment, reason, treatment in ("BLOCKED", "REVERSE_CHARGE")

        # 2. GL code mapping
        if gl in GL_VAT_MAP:
            treatment, reason = GL_VAT_MAP[gl]
            return treatment, reason, treatment in ("BLOCKED", "REVERSE_CHARGE")

        # 3. Default: standard rated with low confidence flag
        return (
            "STANDARD",
            f"GL {gl} not in VAT map -- defaulting to standard rate. Manual review recommended.",
            True,
        )

    @staticmethod
    def _treatment_to_rate(treatment: str) -> str:
        return {
            "STANDARD": "20%",
            "REDUCED": "5%",
            "ZERO": "0%",
            "EXEMPT": "N/A",
            "OUTSIDE_SCOPE": "N/A",
            "REVERSE_CHARGE": "20% (RC)",
            "BLOCKED": "20% (blocked)",
        }.get(treatment, "20%")
