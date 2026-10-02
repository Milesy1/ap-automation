"""
Langfuse Trace Analysis Script — v2 Observations API
Pulls all observations and generates a full analysis report.

Usage:
    cd C:\\Users\\Miles\\Desktop\\Projects\\ap-automation
    python scripts/langfuse_analysis.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

PUBLIC_KEY = "pk-lf-beb692d4-53fe-4246-8922-a6347ba30b55"
SECRET_KEY = "sk-lf-deef0ed8-59cb-45f7-92f8-807ea0c2502e"
HOST = "https://cloud.langfuse.com"
AUTH = HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)

OUTPUT_DIR = Path("data/langfuse_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_observations(name: str, limit: int = 500) -> list[dict]:
    """Fetch observations via Langfuse v2 API."""
    print(f"Fetching '{name}' observations...")
    observations = []
    page = 1
    while True:
        r = requests.get(
            f"{HOST}/api/public/v2/observations",
            auth=AUTH,
            params={"name": name, "type": "SPAN", "limit": 50, "page": page},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  API error {r.status_code}: {r.text[:200]}")
            break
        data = r.json()
        items = data.get("data", [])
        if not items:
            break
        observations.extend(items)
        total = data.get("meta", {}).get("totalItems", 0)
        print(f"  {len(observations)}/{total}...")
        if len(observations) >= total or len(observations) >= limit:
            break
        page += 1
    print(f"  Total fetched: {len(observations)}")
    return observations


def analyse_predictions(obs: list[dict]) -> dict:
    total = len(obs)
    if not total:
        return {}

    routing_counts = Counter()
    confidence_scores = []
    evidence_counts = []
    latencies = []
    gl_distribution = Counter()
    vendor_counts = Counter()

    for o in obs:
        output = o.get("output") or {}
        metadata = o.get("metadata") or {}
        input_data = o.get("input") or {}

        routing = output.get("routing_decision") or metadata.get("routing_decision", "unknown")
        routing_counts[routing] += 1

        for val, lst in [
            (output.get("weighted_confidence") or metadata.get("weighted_confidence"), confidence_scores),
            (output.get("evidence_count") or metadata.get("evidence_count"), evidence_counts),
        ]:
            if val is not None:
                try:
                    lst.append(float(val))
                except Exception:
                    pass

        lat = metadata.get("retrieval_latency_ms")
        if lat is not None:
            try:
                latencies.append(int(lat))
            except Exception:
                pass

        gl = output.get("predicted_gl")
        if gl:
            gl_distribution[gl] += 1

        vendor = input_data.get("vendor_raw")
        if vendor:
            vendor_counts[vendor] += 1

    auto_count = routing_counts.get("auto_post", 0)
    return {
        "total_predictions": total,
        "routing_breakdown": dict(routing_counts),
        "auto_post_count": auto_count,
        "human_review_count": total - auto_count,
        "auto_post_rate": round(auto_count / total, 4),
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else 0,
        "avg_evidence_count": round(sum(evidence_counts) / len(evidence_counts), 2) if evidence_counts else 0,
        "avg_retrieval_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_retrieval_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "gl_distribution": dict(gl_distribution.most_common(10)),
        "top_vendors": dict(vendor_counts.most_common(10)),
    }


def analyse_confirmations(obs: list[dict]) -> dict:
    total = len(obs)
    if not total:
        return {}

    correct = 0
    false_positives = 0
    source_counts = Counter()

    for o in obs:
        metadata = o.get("metadata") or {}
        input_data = o.get("input") or {}
        if metadata.get("correct") is True:
            correct += 1
        if metadata.get("false_positive") is True:
            false_positives += 1
        source_counts[input_data.get("source", "unknown")] += 1

    return {
        "total_confirmations": total,
        "correct_count": correct,
        "accuracy_rate": round(correct / total, 4),
        "false_positive_count": false_positives,
        "false_positive_rate": round(false_positives / total, 4),
        "source_breakdown": dict(source_counts),
    }


def analyse_writebacks(obs: list[dict]) -> dict:
    total = len(obs)
    if not total:
        return {}

    corpus_sizes = []
    for o in obs:
        metadata = o.get("metadata") or {}
        after = metadata.get("corpus_after")
        if after is not None:
            try:
                corpus_sizes.append(int(after))
            except Exception:
                pass

    return {
        "total_writebacks": total,
        "corpus_start": min(corpus_sizes) if corpus_sizes else 0,
        "corpus_end": max(corpus_sizes) if corpus_sizes else 0,
        "corpus_growth": (max(corpus_sizes) - min(corpus_sizes)) if len(corpus_sizes) > 1 else 0,
    }


def save_csv(obs: list[dict], path: Path) -> None:
    if not obs:
        return
    rows = []
    for o in obs:
        output = o.get("output") or {}
        metadata = o.get("metadata") or {}
        input_data = o.get("input") or {}
        rows.append({
            "observation_id": o.get("id", ""),
            "trace_id": o.get("traceId", ""),
            "name": o.get("name", ""),
            "timestamp": str(o.get("startTime", "")),
            "latency_ms": o.get("latency", ""),
            "invoice_id": input_data.get("invoice_id", ""),
            "vendor_raw": input_data.get("vendor_raw", ""),
            "description": str(input_data.get("description", ""))[:80],
            "amount": input_data.get("amount", ""),
            "predicted_gl": output.get("predicted_gl", ""),
            "weighted_confidence": output.get("weighted_confidence", ""),
            "evidence_count": output.get("evidence_count", ""),
            "routing_decision": output.get("routing_decision", metadata.get("routing_decision", "")),
            "retrieval_latency_ms": metadata.get("retrieval_latency_ms", ""),
            "correct": metadata.get("correct", ""),
            "false_positive": metadata.get("false_positive", ""),
            "corpus_after": metadata.get("corpus_after", ""),
            "source": input_data.get("source", ""),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows → {path}")


def print_report(pred: dict, conf: dict, wb: dict) -> None:
    print("\n" + "=" * 65)
    print("AP AUTOMATION — LANGFUSE TRACE ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if pred:
        print(f"\nPREDICTION METRICS ({pred['total_predictions']} invoices)")
        print(f"  Auto-posted:              {pred['auto_post_count']} ({pred['auto_post_rate']:.1%})")
        print(f"  Human review:             {pred['human_review_count']}")
        print(f"  Average confidence:       {pred['avg_confidence']:.1%}")
        print(f"  Average evidence lines:   {pred['avg_evidence_count']:.1f}")
        print(f"  Avg retrieval latency:    {pred['avg_retrieval_latency_ms']:.0f}ms")
        print(f"  p95 retrieval latency:    {pred['p95_retrieval_latency_ms']}ms")
        print(f"\n  GL Distribution (top 5):")
        for gl, n in list(pred["gl_distribution"].items())[:5]:
            print(f"    GL {gl}: {n} invoices")
        print(f"\n  Top Vendors (top 5):")
        for v, n in list(pred["top_vendors"].items())[:5]:
            print(f"    {v[:40]}: {n}")

    if conf:
        print(f"\nACCURACY METRICS ({conf['total_confirmations']} confirmations)")
        print(f"  Correct predictions:      {conf['correct_count']} ({conf['accuracy_rate']:.1%})")
        print(f"  False positives:          {conf['false_positive_count']} ({conf['false_positive_rate']:.1%})")
        print(f"  Source breakdown:         {conf['source_breakdown']}")

    if wb:
        print(f"\nWRITE-BACK METRICS ({wb['total_writebacks']} write-backs)")
        print(f"  Corpus start:             {wb['corpus_start']:,} points")
        print(f"  Corpus end:               {wb['corpus_end']:,} points")
        print(f"  Corpus growth:            +{wb['corpus_growth']:,} points")

    print("\n" + "=" * 65)


def main():
    pred_obs = fetch_observations("invoice_prediction")
    conf_obs = fetch_observations("invoice_confirmation")
    wb_obs = fetch_observations("write_back")

    pred_analysis = analyse_predictions(pred_obs)
    conf_analysis = analyse_confirmations(conf_obs)
    wb_analysis = analyse_writebacks(wb_obs)

    print("\nSaving outputs...")
    save_csv(pred_obs + conf_obs + wb_obs, OUTPUT_DIR / "all_observations.csv")
    save_csv(pred_obs, OUTPUT_DIR / "prediction_observations.csv")
    save_csv(conf_obs, OUTPUT_DIR / "confirmation_observations.csv")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": pred_analysis,
        "confirmations": conf_analysis,
        "write_backs": wb_analysis,
    }
    with open(OUTPUT_DIR / "analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {OUTPUT_DIR / 'analysis.json'}")

    print_report(pred_analysis, conf_analysis, wb_analysis)
    print(f"\nAll outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
