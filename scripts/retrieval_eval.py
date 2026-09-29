"""
Retrieval Evaluation Script
Runs a set of test queries against the Qdrant corpus and produces a
detailed report showing what was retrieved, similarity scores, agreement
ratios, and whether the predicted GL matches ground truth.

Usage:
    cd C:\\Users\\Miles\\Desktop\\Projects\\ap-automation
    python scripts/retrieval_eval.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ap_automation.core.config import settings
from ap_automation.core.mdm import load_vendor_map, resolve_vendor
from ap_automation.core.models import AmountBand, amount_to_band
from ap_automation.core.retrieval import retrieve

# ── Test cases ────────────────────────────────────────────────────────────────
# Each test: (vendor_name, description, amount, entity, expected_gl)
TEST_CASES = [
    # Routine — known vendors, clean descriptions
    ("Meridian Facilities Ltd",       "Office cleaning services monthly",          450.0,   "UK001", "6300"),
    ("Meridian Facilities Ltd",       "Building maintenance and repairs",           1200.0,  "UK001", "6300"),
    ("Meridian Facilities Ltd",       "Facility management services Q3",            800.0,   "UK001", "6300"),
    ("TechSource IT Solutions",       "Annual software licence renewal",            5000.0,  "UK001", "6200"),
    ("TechSource IT Solutions",       "Cloud infrastructure services Q2",           8000.0,  "UK002", "6200"),
    ("Global Office Supplies",        "Paper, pens and office supplies Q4",         150.0,   "UK001", "6100"),
    ("Global Office Supplies",        "Printer cartridges and stationery",          220.0,   "UK002", "6100"),
    ("Horizon Consulting Group",      "Management consulting services",             12000.0, "UK001", "6400"),
    ("Swift Travel Services",         "Business travel accommodation",              900.0,   "UK001", "6500"),
    ("Powerline Utilities",           "Electricity supply monthly",                 600.0,   "UK001", "6600"),
    ("Brandworks Marketing",          "Digital marketing campaign",                 5000.0,  "UK001", "6700"),
    ("Learn Forward Training",        "Staff training programme",                   2000.0,  "UK001", "6800"),
    ("Apex Equipment Co",             "Server hardware purchase",                   45000.0, "UK001", "7100"),

    # Near-edge — ambiguous descriptions
    ("Meridian Facilities Ltd",       "Monthly invoice - see attached",             450.0,   "UK001", "6300"),
    ("TechSource IT Solutions",       "IT svcs Q3 - ref PO-99123",                 3000.0,  "UK001", "6200"),
    ("Global Office Supplies",        "Misc office items as per agreement",          180.0,   "UK002", "6100"),

    # Hard-edge — unknown vendors
    ("Zephyr Global Holdings",        "Management consulting services",             15000.0, "UK001", "6400"),
    ("Arcturus Supply Co",            "Office consumables replenishment",           200.0,   "UK001", "6100"),
    ("Pulsar Tech GmbH",              "Software subscription renewal",              4000.0,  "UK001", "6200"),
]


def evaluate_query(
    vendor_name: str,
    description: str,
    amount: float,
    entity: str,
    expected_gl: str,
) -> dict:
    canonical_id, mdm_conf = resolve_vendor(vendor_name)
    amount_band = amount_to_band(amount)

    evidence = retrieve(
        description=description,
        vendor_id=canonical_id,
        entity_id=entity,
        amount_band=amount_band,
        currency="GBP",
    )

    if not evidence:
        return {
            "vendor": vendor_name,
            "canonical_id": canonical_id,
            "mdm_confidence": round(mdm_conf, 3),
            "description": description,
            "amount": amount,
            "expected_gl": expected_gl,
            "evidence_count": 0,
            "predicted_gl": "",
            "agreement_ratio": 0.0,
            "weighted_confidence": 0.0,
            "correct": False,
            "routing": "human_review",
            "routing_reason": "Sparse retrieval: 0 lines retrieved",
            "top_evidence": [],
        }

    gl_counts = Counter(e.gl_account for e in evidence)
    predicted_gl, count = gl_counts.most_common(1)[0]
    agreement_ratio = count / len(evidence)
    agreeing = [e for e in evidence if e.gl_account == predicted_gl]
    avg_sim = sum(e.similarity_score for e in agreeing) / len(agreeing)
    weighted_confidence = agreement_ratio * avg_sim

    routing = "auto_post" if (
        len(evidence) >= settings.min_evidence_lines and
        weighted_confidence >= settings.default_confidence_threshold
    ) else "human_review"

    routing_reason = (
        f"Confidence {weighted_confidence:.3f} meets threshold {settings.default_confidence_threshold}"
        if routing == "auto_post"
        else f"Confidence {weighted_confidence:.3f} below threshold {settings.default_confidence_threshold}"
        if len(evidence) >= settings.min_evidence_lines
        else f"Sparse retrieval: only {len(evidence)} lines retrieved"
    )

    return {
        "vendor": vendor_name,
        "canonical_id": canonical_id,
        "mdm_confidence": round(mdm_conf, 3),
        "description": description,
        "amount": amount,
        "expected_gl": expected_gl,
        "evidence_count": len(evidence),
        "predicted_gl": predicted_gl,
        "agreement_ratio": round(agreement_ratio, 3),
        "weighted_confidence": round(weighted_confidence, 3),
        "correct": predicted_gl == expected_gl,
        "routing": routing,
        "routing_reason": routing_reason,
        "top_evidence": [
            {
                "description": e.description,
                "gl_account": e.gl_account,
                "similarity_score": round(e.similarity_score, 4),
                "rrf_score": round(e.rrf_score, 6),
                "source": e.source.value,
            }
            for e in evidence[:3]
        ],
    }


def print_report(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    auto_posted = sum(1 for r in results if r["routing"] == "auto_post")
    auto_correct = sum(1 for r in results if r["routing"] == "auto_post" and r["correct"])
    no_evidence = sum(1 for r in results if r["evidence_count"] == 0)

    print("\n" + "="*80)
    print("AP AUTOMATION — RETRIEVAL EVALUATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Corpus size: {settings.qdrant_collection} | Confidence threshold: {settings.default_confidence_threshold}")
    print("="*80)

    print(f"\n{'SUMMARY':}")
    print(f"  Total test cases:        {total}")
    print(f"  Correct predictions:     {correct}/{total} ({correct/total:.0%})")
    print(f"  Auto-posted:             {auto_posted}/{total} ({auto_posted/total:.0%})")
    print(f"  Auto-post accuracy:      {auto_correct}/{auto_posted} ({auto_correct/auto_posted:.0%})" if auto_posted else "  Auto-post accuracy:      N/A")
    print(f"  Zero evidence (review):  {no_evidence}/{total}")

    print(f"\n{'DETAIL':}")
    print(f"{'Vendor':<30} {'Description':<40} {'Exp GL':<8} {'Pred GL':<8} {'Conf':<7} {'Ev':<4} {'Routing':<12} {'OK'}")
    print("-"*120)

    for r in results:
        desc = r["description"][:38] + ".." if len(r["description"]) > 40 else r["description"]
        vendor = r["vendor"][:28] + ".." if len(r["vendor"]) > 30 else r["vendor"]
        routing = "AUTO" if r["routing"] == "auto_post" else "REVIEW"
        ok = "✓" if r["correct"] else "✗" if r["predicted_gl"] else "—"
        print(f"{vendor:<30} {desc:<40} {r['expected_gl']:<8} {r['predicted_gl']:<8} {r['weighted_confidence']:<7.3f} {r['evidence_count']:<4} {routing:<12} {ok}")

    print("\n" + "="*80)
    print("TOP EVIDENCE PER QUERY")
    print("="*80)

    for r in results:
        print(f"\n▶ {r['vendor']} | \"{r['description'][:60]}\"")
        print(f"  MDM: {r['canonical_id']} (conf: {r['mdm_confidence']}) | Expected GL: {r['expected_gl']} | Predicted: {r['predicted_gl'] or 'none'}")
        print(f"  Evidence: {r['evidence_count']} lines | Agreement: {r['agreement_ratio']:.0%} | Confidence: {r['weighted_confidence']:.3f} | Routing: {r['routing'].upper()}")
        if r["top_evidence"]:
            for i, e in enumerate(r["top_evidence"]):
                print(f"  [{i+1}] GL:{e['gl_account']} sim:{e['similarity_score']:.4f} rrf:{e['rrf_score']:.6f} | {e['description'][:60]}")
        else:
            print("  [No evidence retrieved]")


def save_csv(results: list[dict], path: str = "data/retrieval_eval.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        fields = [
            "vendor", "canonical_id", "mdm_confidence", "description", "amount",
            "expected_gl", "evidence_count", "predicted_gl", "agreement_ratio",
            "weighted_confidence", "correct", "routing", "routing_reason"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\n✓ CSV saved → {path}")


def save_json(results: list[dict], path: str = "data/retrieval_eval.json") -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON saved → {path}")


if __name__ == "__main__":
    print("Loading vendor map...")
    load_vendor_map("data/vendor_map.csv")

    print(f"Running {len(TEST_CASES)} test queries against Qdrant...")
    results = []
    for i, (vendor, desc, amount, entity, expected_gl) in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {vendor[:40]}...")
        result = evaluate_query(vendor, desc, amount, entity, expected_gl)
        results.append(result)

    print_report(results)
    save_csv(results)
    save_json(results)
