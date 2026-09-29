"""
Synthetic dataset generator — 1,000 invoices across three tiers.
700 routine / 200 near-edge / 100 hard-edge.
All labelled with ground-truth GL codes.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from uuid import uuid4

from ap_automation.core.models import InvoiceCategory, InvoiceLine


# --- GL code catalogue ---
GL_CODES = {
    "6100": "Office Supplies",
    "6200": "IT & Software",
    "6300": "Facilities & Maintenance",
    "6400": "Professional Services",
    "6500": "Travel & Accommodation",
    "6600": "Utilities",
    "6700": "Marketing & Advertising",
    "6800": "Training & Development",
    "7100": "Equipment (CapEx)",
    "7200": "Leasehold Improvements (CapEx)",
}

COST_CENTRES = ["CC100", "CC200", "CC300", "CC400", "CC500"]
ENTITIES = ["UK001", "UK002", "IE001"]
CURRENCIES = ["GBP", "GBP", "GBP", "EUR", "USD"]  # weighted towards GBP

# --- Vendor catalogue ---
VENDORS: dict[str, dict] = {
    "MERIDIAN_FACILITIES": {
        "raw_names": ["Meridian Facilities Ltd", "Meridian Facilities", "Meridian Fac. Ltd"],
        "gl": "6300",
        "typical_amounts": (200, 5000),
    },
    "TECHSOURCE_IT": {
        "raw_names": ["TechSource IT Solutions", "TechSource IT", "Tech Source Ltd"],
        "gl": "6200",
        "typical_amounts": (500, 15000),
    },
    "GLOBAL_OFFICE": {
        "raw_names": ["Global Office Supplies", "Global Office Ltd", "Global Office"],
        "gl": "6100",
        "typical_amounts": (50, 800),
    },
    "HORIZON_CONSULT": {
        "raw_names": ["Horizon Consulting Group", "Horizon Consulting", "Horizon Group Ltd"],
        "gl": "6400",
        "typical_amounts": (2000, 25000),
    },
    "SWIFT_TRAVEL": {
        "raw_names": ["Swift Travel Services", "Swift Travel Ltd", "SwiftTravel"],
        "gl": "6500",
        "typical_amounts": (150, 3000),
    },
    "POWERLINE_UTILS": {
        "raw_names": ["Powerline Utilities", "Powerline Util. Ltd", "Powerline"],
        "gl": "6600",
        "typical_amounts": (300, 2000),
    },
    "BRANDWORKS_MKT": {
        "raw_names": ["Brandworks Marketing", "BrandWorks Ltd", "Brandworks Agency"],
        "gl": "6700",
        "typical_amounts": (1000, 20000),
    },
    "LEARN_FORWARD": {
        "raw_names": ["Learn Forward Training", "LearnForward Ltd", "Learn Forward"],
        "gl": "6800",
        "typical_amounts": (500, 5000),
    },
    "APEX_EQUIPMENT": {
        "raw_names": ["Apex Equipment Co", "Apex Equipment Ltd", "Apex Co."],
        "gl": "7100",
        "typical_amounts": (5000, 80000),
    },
}

# Description templates per GL
DESCRIPTIONS: dict[str, list[str]] = {
    "6100": [
        "Office stationery and printer consumables",
        "Paper, pens and office supplies Q{q}",
        "Monthly office supplies order",
        "Printer cartridges and stationery",
        "Office consumables replenishment",
    ],
    "6200": [
        "Annual software licence renewal",
        "Cloud infrastructure services Q{q}",
        "IT support and maintenance contract",
        "Software subscription renewal",
        "Cybersecurity platform licence",
        "SaaS platform annual fee",
    ],
    "6300": [
        "Office cleaning services monthly",
        "Building maintenance and repairs",
        "HVAC servicing and inspection",
        "Facility management services Q{q}",
        "Pest control and grounds maintenance",
    ],
    "6400": [
        "Management consulting services",
        "Legal advisory fees Q{q}",
        "Audit and assurance services",
        "HR consulting project",
        "Financial advisory retainer",
    ],
    "6500": [
        "Business travel accommodation",
        "Flight and hotel Q{q}",
        "Rail travel and expenses",
        "Client meeting travel costs",
        "Conference accommodation and travel",
    ],
    "6600": [
        "Electricity supply monthly",
        "Gas and electricity Q{q}",
        "Water rates quarterly",
        "Utilities combined bill",
        "Energy supply invoice",
    ],
    "6700": [
        "Digital marketing campaign",
        "Social media advertising Q{q}",
        "Brand design and production",
        "PR retainer monthly",
        "Marketing materials and print",
    ],
    "6800": [
        "Staff training programme",
        "Leadership development course",
        "Compliance training Q{q}",
        "Technical skills workshop",
        "Professional development programme",
    ],
    "7100": [
        "Server hardware purchase",
        "Network equipment installation",
        "Production machinery purchase",
        "Capital equipment acquisition",
    ],
}

NEW_VENDORS = [
    ("NOVACORP_SVCS", "NovaCorp Services Ltd", "6400", (1000, 10000)),
    ("DATABRIDGE_IT", "DataBridge IT Solutions", "6200", (2000, 20000)),
    ("CLEANZONE_FAC", "CleanZone Facilities", "6300", (300, 3000)),
    ("SUMMIT_TRAIN", "Summit Training Academy", "6800", (800, 6000)),
    ("GREENPATH_UTL", "GreenPath Utilities", "6600", (400, 2500)),
]


def _random_description(gl: str, near_edge: bool = False) -> str:
    templates = DESCRIPTIONS.get(gl, ["Invoice for services rendered"])
    tmpl = random.choice(templates)
    q = random.randint(1, 4)
    desc = tmpl.format(q=q)
    if near_edge:
        # Add noise to make it ambiguous
        noises = ["- see attached", "ref PO-{n}".format(n=random.randint(10000, 99999)), "various items", "as per agreement"]
        desc = f"{desc} {random.choice(noises)}"
    return desc


def generate_dataset(
    n_routine: int = 700,
    n_near_edge: int = 200,
    n_hard_edge: int = 100,
    output_path: str = "data/invoices.csv",
    vendor_map_path: str = "data/vendor_map.csv",
) -> list[InvoiceLine]:
    """Generate synthetic invoice dataset and write to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    invoices: list[InvoiceLine] = []
    vendor_map_rows: list[dict] = []

    # Build vendor map
    for canonical_id, info in VENDORS.items():
        for raw_name in info["raw_names"]:
            vendor_map_rows.append({"raw_name": raw_name, "canonical_id": canonical_id})

    # --- Routine invoices ---
    vendor_keys = list(VENDORS.keys())
    for _ in range(n_routine):
        vendor_id = random.choice(vendor_keys)
        info = VENDORS[vendor_id]
        raw_name = random.choice(info["raw_names"])
        gl = info["gl"]
        amount = round(random.uniform(*info["typical_amounts"]), 2)
        invoices.append(InvoiceLine(
            invoice_id=uuid4(),
            raw_vendor_name=raw_name,
            description=_random_description(gl),
            amount=amount,
            currency=random.choice(CURRENCIES),
            entity_id=random.choice(ENTITIES),
            cost_centre=random.choice(COST_CENTRES),
            category=InvoiceCategory.ROUTINE,
            ground_truth_gl=gl,
        ))

    # --- Near-edge invoices ---
    for _ in range(n_near_edge):
        # Mix: some known vendors with ambiguous descriptions, some new vendors
        if random.random() < 0.5:
            vendor_id = random.choice(vendor_keys)
            info = VENDORS[vendor_id]
            raw_name = random.choice(info["raw_names"])
            gl = info["gl"]
            amount = round(random.uniform(*info["typical_amounts"]), 2)
            desc = _random_description(gl, near_edge=True)
        else:
            nv = random.choice(NEW_VENDORS)
            vendor_id, raw_name, gl, amount_range = nv
            amount = round(random.uniform(*amount_range), 2)
            desc = _random_description(gl, near_edge=True)
            vendor_map_rows.append({"raw_name": raw_name, "canonical_id": vendor_id})

        invoices.append(InvoiceLine(
            invoice_id=uuid4(),
            raw_vendor_name=raw_name,
            description=desc,
            amount=amount,
            currency=random.choice(CURRENCIES),
            entity_id=random.choice(ENTITIES),
            cost_centre=random.choice(COST_CENTRES),
            category=InvoiceCategory.NEAR_EDGE,
            ground_truth_gl=gl,
        ))

    # --- Hard-edge invoices ---
    hard_vendors = [
        ("UNKNOWN_VENDOR_A", "Zephyr Global Holdings", "6400", (5000, 50000)),
        ("UNKNOWN_VENDOR_B", "Arcturus Supply Co", "6100", (100, 500)),
        ("UNKNOWN_VENDOR_C", "Pulsar Tech GmbH", "6200", (3000, 30000)),
        ("UNKNOWN_VENDOR_D", "Meridian East Europe s.r.o.", "6300", (200, 2000)),
        ("UNKNOWN_VENDOR_E", "Solaris Energy Partners", "6600", (500, 5000)),
    ]
    for i in range(n_hard_edge):
        hv = hard_vendors[i % len(hard_vendors)]
        vendor_id, raw_name, gl, amount_range = hv
        amount = round(random.uniform(*amount_range), 2)
        currency = random.choice(["GBP", "EUR", "USD", "CHF"])
        desc = _random_description(gl, near_edge=True)

        invoices.append(InvoiceLine(
            invoice_id=uuid4(),
            raw_vendor_name=raw_name,
            description=desc,
            amount=amount,
            currency=currency,
            entity_id=random.choice(ENTITIES),
            cost_centre=random.choice(COST_CENTRES),
            category=InvoiceCategory.HARD_EDGE,
            ground_truth_gl=gl,
        ))

    # Write invoices CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "invoice_id", "raw_vendor_name", "description", "amount",
            "currency", "entity_id", "cost_centre", "category", "ground_truth_gl",
        ])
        writer.writeheader()
        for inv in invoices:
            writer.writerow({
                "invoice_id": str(inv.invoice_id),
                "raw_vendor_name": inv.raw_vendor_name,
                "description": inv.description,
                "amount": inv.amount,
                "currency": inv.currency,
                "entity_id": inv.entity_id,
                "cost_centre": inv.cost_centre or "",
                "category": inv.category.value,
                "ground_truth_gl": inv.ground_truth_gl or "",
            })

    # Write vendor map CSV
    with open(vendor_map_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_name", "canonical_id"])
        writer.writeheader()
        seen = set()
        for row in vendor_map_rows:
            key = (row["raw_name"], row["canonical_id"])
            if key not in seen:
                writer.writerow(row)
                seen.add(key)

    print(f"Generated {len(invoices)} invoices → {output_path}")
    print(f"Vendor map → {vendor_map_path}")
    return invoices


if __name__ == "__main__":
    random.seed(42)
    generate_dataset()
