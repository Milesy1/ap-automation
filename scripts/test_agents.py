from ap_automation.agents.vat_agent import VATAgent
from ap_automation.agents.cost_centre_agent import CostCentreAgent
from ap_automation.agents.entity_agent import EntityAgent
from ap_automation.agents.orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
print("Agent cards:")
for card in orch.agent_cards():
    print("  -", card["name"], ":", card["description"][:60])

context = {
    "predicted_gl": "6300",
    "predicted_cost_centre": "FACILITIES",
    "description": "Office cleaning monthly",
    "amount": 450.0,
    "currency": "GBP",
    "vendor_id": "MERIDIAN_FACILITIES",
    "entity_id": "UK001",
    "routing_decision": "auto_post",
    "weighted_confidence": 0.85,
}

result = orch.run(context)
print("\nAgents called:", result.to_dict()["agents_called"])
print("VAT treatment:", result.vat.data["vat_treatment"])
print("VAT note:", result.vat.data["vat_note"])
print("Cost centre:", result.cost_centre.data["inferred_cost_centre"])
print("Entity valid:", result.entity.data["entity_valid"])
print("All successful:", result.all_successful)
