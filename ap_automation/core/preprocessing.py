"""
Description preprocessing pipeline.
Cleans invoice descriptions before embedding.
"""

from __future__ import annotations

import re


ABBREVIATION_MAP: dict[str, str] = {
    "svcs": "services",
    "svc": "service",
    "maint": "maintenance",
    "mgmt": "management",
    "admin": "administration",
    "supp": "supplies",
    "equip": "equipment",
    "consult": "consulting",
    "prof": "professional",
    "misc": "miscellaneous",
    "sub": "subscription",
    "lic": "license",
    "util": "utilities",
    "tele": "telecommunications",
    "trans": "transport",
    "acc": "accommodation",
    "ref": "reference",
    "dept": "department",
    "q1": "quarter one",
    "q2": "quarter two",
    "q3": "quarter three",
    "q4": "quarter four",
}

# PO reference patterns to strip
PO_PATTERNS = [
    r"\bpo[#\-\s]?\d+\b",
    r"\bpurchase order[#\-\s]?\d+\b",
    r"\binv[#\-\s]?\d+\b",
    r"\binvoice[#\-\s]?\d+\b",
    r"\bref[#\-\s]?\w+\b",
    r"\b\d{6,}\b",  # long numeric sequences (invoice numbers)
]


def preprocess_description(description: str) -> str:
    """
    Clean and normalise an invoice description for embedding.

    Steps:
    1. Lowercase
    2. Strip PO references and invoice numbers
    3. Expand common abbreviations
    4. Remove special characters
    5. Collapse whitespace
    """
    text = description.lower().strip()

    # Strip PO references
    for pattern in PO_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Expand abbreviations (word boundary match)
    for abbr, expansion in ABBREVIATION_MAP.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text)

    # Remove special characters except spaces and hyphens
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
