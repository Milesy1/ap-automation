"""
MDM resolution — vendor name → canonical ID.
POC stub: CSV lookup with fuzzy matching.
Production: MDM service API call.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


VENDOR_MAP: dict[str, str] = {}


def load_vendor_map(path: str = "data/vendor_map.csv") -> None:
    """Load vendor name → canonical ID mapping from CSV."""
    global VENDOR_MAP
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            VENDOR_MAP[_normalise(row["raw_name"])] = row["canonical_id"].strip()


def _normalise(name: str) -> str:
    """Normalise vendor name for matching."""
    name = name.lower().strip()
    name = re.sub(r"\b(ltd|limited|plc|inc|llc|gmbh|co|corp|group|agency|services|solutions|partners)\b\.?", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_vendor(raw_name: str) -> tuple[str, float]:
    """
    Resolve raw vendor name to canonical ID.
    Returns (canonical_vendor_id, confidence).
    """
    if not VENDOR_MAP:
        load_vendor_map()

    normalised = _normalise(raw_name)

    # Exact match
    if normalised in VENDOR_MAP:
        return VENDOR_MAP[normalised], 1.0

    # Prefix / substring match
    for key, canonical_id in VENDOR_MAP.items():
        if normalised.startswith(key) or key.startswith(normalised):
            return canonical_id, 0.9

    # Token overlap match
    tokens = set(normalised.split())
    best_score = 0.0
    best_id = f"NEW_{re.sub(r'[^A-Z0-9]', '_', raw_name.upper())[:20]}"

    for key, canonical_id in VENDOR_MAP.items():
        key_tokens = set(key.split())
        if not key_tokens:
            continue
        overlap = len(tokens & key_tokens) / max(len(tokens | key_tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best_id = canonical_id

    if best_score >= 0.4:
        return best_id, best_score

    derived_id = f"NEW_{re.sub(r'[^A-Z0-9]', '_', raw_name.upper())[:20]}"
    return derived_id, 0.3
