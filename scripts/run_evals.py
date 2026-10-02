"""
AP Automation — Custom Langfuse Evaluators
Runs four custom evals against the invoice_prediction observations in Langfuse:
  1. Recall@10          — was correct GL in top retrieved lines?
  2. Calibration        — does confidence % match actual accuracy %?
  3. Explanation quality — Claude judges the plain-English explanation
  4. False positive rate — any incorrect auto-posts?

Usage:
    cd C:\\Users\\Miles\\Desktop\\Projects\\ap-automation
    python scripts/run_evals.py --export-path data/langfuse_analysis/all_observations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# ── Config ────────────────────────────────────────────
PUBLIC_KEY = "pk-lf-beb692d4-53fe-4246-8922-a6347ba30b55"
SECRET_KEY = "sk-lf-deef0ed8-59cb-45f7-92f8-807ea0c2502e"
HOST = "https://cloud.langfuse.com"
AUTH = HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EXPLANATION_JUDGE_PROMPT = """You are evaluating the quality of an automated AP invoice coding explanation.

The system predicted a GL code for an invoice and produced the following plain-English explanation for a finance clerk:

EXPLANATION:
{explanation}

CONTEXT:
- Predicted GL code: {predicted_gl}
- Routing decision: {routing_decision}
- Confidence score: {confidence}
- Evidence lines retrieved: {evidence_count}

Score the explanation on three dimensions, each 1-5:
1. Accuracy — does it correctly describe why this GL code was chosen?
2. Completeness — does it mention confidence score, evidence count, and routing reason?
3. Plain language — would a non-technical AP clerk understand it?

Respond with ONLY a JSON object like this:
{{"accuracy": 4, "completeness": 3, "plain_language": 5, "overall": 4.0, "comment": "one sentence"}}"""


def post_score(trace_id: str, observation_id: str, name: str, value: float, comment: str = "") -> bool:
    """Post a score to Langfuse via the Scores API."""
    payload = {
        "name": name,
        "value": value,
        "traceId": trace_id,
        "observationId": observation_id,
    }
    if comment:
        payload["comment"] = comment
    r = requests.post(
        f"{HOST}/api/public/scores",
        auth=AUTH,
        json=payload,
        timeout=15,
    )
    return r.status_code in (200, 201)


def load_observations(csv_path: str) -> list[dict]:
    """Load observations from the exported CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"Export file not found: {csv_path}")
        print("Run scripts/langfuse_analysis.py first to generate it.")
        return []
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print(f"Loaded {len(rows)} observations from {csv_path}")
    return rows


def fetch_observation_detail(obs_id: str) -> dict:
    """Fetch full observation detail including input/output/metadata."""
    r = requests.get(
        f"{HOST}/api/public/observations/{obs_id}",
        auth=AUTH,
        timeout=15,
    )
    if r.status_code == 200:
        return r.json()
    return {}


# ── Eval 1: Recall@10 ─────────────────────────────────
def run_recall_eval(rows: list[dict], dry_run: bool = False) -> dict:
    """
    Recall@10: was the correct GL code in the top retrieved lines?
    Requires ground_truth_gl in the observation input.
    """
    print("\n── Eval 1: Recall@10 ─────────────────────────────")
    pred_rows = [r for r in rows if r.get("name") == "invoice_prediction"]
    scored = 0
    correct = 0
    skipped = 0

    for row in pred_rows:
        ground_truth = row.get("ground_truth_gl", "")
        predicted_gl = row.get("predicted_gl", "")
        obs_id = row.get("observation_id", "")
        trace_id = row.get("trace_id", "")

        if not ground_truth or not obs_id:
            skipped += 1
            continue

        # Recall@10: predicted GL matches ground truth (system retrieved correct GL)
        recall = 1.0 if predicted_gl == ground_truth else 0.0
        scored += 1
        if recall == 1.0:
            correct += 1

        if not dry_run and obs_id and trace_id:
            post_score(
                trace_id=trace_id,
                observation_id=obs_id,
                name="retrieval.recall_at_10",
                value=recall,
                comment=f"Predicted: {predicted_gl}, Ground truth: {ground_truth}",
            )

    rate = correct / scored if scored else 0
    print(f"  Scored: {scored} | Correct: {correct} | Recall@10: {rate:.1%} | Skipped: {skipped}")
    return {"recall_at_10": rate, "scored": scored, "correct": correct}


# ── Eval 2: Calibration ───────────────────────────────
def run_calibration_eval(rows: list[dict], dry_run: bool = False) -> dict:
    """
    Calibration: does confidence % match actual accuracy % per band?
    Groups predictions by confidence band and compares to actual accuracy.
    """
    print("\n── Eval 2: Calibration ────────────────────────────")
    pred_rows = [r for r in rows if r.get("name") == "invoice_prediction"]

    bands = defaultdict(lambda: {"correct": 0, "total": 0, "obs_ids": []})

    for row in pred_rows:
        try:
            conf = float(row.get("weighted_confidence", 0))
            ground_truth = row.get("ground_truth_gl", "")
            predicted = row.get("predicted_gl", "")
            obs_id = row.get("observation_id", "")
            trace_id = row.get("trace_id", "")
        except (ValueError, TypeError):
            continue

        if not ground_truth:
            continue

        band = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
        is_correct = 1 if predicted == ground_truth else 0
        bands[band]["correct"] += is_correct
        bands[band]["total"] += 1
        bands[band]["obs_ids"].append((obs_id, trace_id, conf, is_correct))

    print(f"  {'Band':<15} {'Confidence':>12} {'Actual acc':>12} {'Gap':>8} {'n':>6}")
    results = {}
    for band in sorted(bands.keys()):
        d = bands[band]
        if d["total"] < 3:
            continue
        mid_conf = int(band.split("-")[0]) / 100 + 0.05
        actual_acc = d["correct"] / d["total"]
        gap = abs(mid_conf - actual_acc)
        print(f"  {band:<15} {mid_conf:>11.0%} {actual_acc:>11.0%} {gap:>7.0%} {d['total']:>6}")
        results[band] = {"mid_conf": mid_conf, "actual_acc": actual_acc, "gap": gap, "n": d["total"]}

        # Score each observation in this band with its calibration gap
        if not dry_run:
            for obs_id, trace_id, conf, is_correct in d["obs_ids"]:
                if obs_id and trace_id:
                    post_score(
                        trace_id=trace_id,
                        observation_id=obs_id,
                        name="gate.calibration_gap",
                        value=round(gap, 4),
                        comment=f"Band {band}: conf={conf:.2f}, actual_acc={actual_acc:.2f}",
                    )
            time.sleep(0.1)

    return results


# ── Eval 3: Explanation Quality ───────────────────────
def run_explanation_eval(rows: list[dict], sample_rate: float = 0.1, dry_run: bool = False) -> dict:
    """
    Explanation quality: Claude judges the plain-English explanation.
    Sampled at sample_rate of all prediction traces.
    """
    print(f"\n── Eval 3: Explanation Quality (sample={sample_rate:.0%}) ──")

    if not OPENAI_API_KEY:
        print("  OPENAI_API_KEY not set — skipping explanation eval")
        return {}

    pred_rows = [r for r in rows if r.get("name") == "invoice_prediction"]
    sample_size = max(1, int(len(pred_rows) * sample_rate))
    sample = pred_rows[:sample_size]

    scores = []
    for i, row in enumerate(sample):
        obs_id = row.get("observation_id", "")
        trace_id = row.get("trace_id", "")

        # Fetch full observation to get routing_reason (used as explanation)
        detail = fetch_observation_detail(obs_id) if obs_id else {}
        metadata = detail.get("metadata") or {}
        output = detail.get("output") or {}

        explanation = metadata.get("routing_reason", "")
        predicted_gl = output.get("predicted_gl", row.get("predicted_gl", ""))
        routing = output.get("routing_decision", row.get("routing_decision", ""))
        confidence = output.get("weighted_confidence", row.get("weighted_confidence", ""))
        evidence_count = output.get("evidence_count", row.get("evidence_count", ""))

        if not explanation:
            continue

        prompt = EXPLANATION_JUDGE_PROMPT.format(
            explanation=explanation,
            predicted_gl=predicted_gl,
            routing_decision=routing,
            confidence=confidence,
            evidence_count=evidence_count,
        )

        try:
            import urllib.request
            req_data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"].strip()
            score_data = json.loads(text)
            overall = float(score_data.get("overall", 0))
            comment = score_data.get("comment", "")
            scores.append(overall)
            print(f"  [{i+1}/{sample_size}] Score: {overall:.1f}/5 — {comment[:60]}")

            if not dry_run and obs_id and trace_id:
                post_score(
                    trace_id=trace_id,
                    observation_id=obs_id,
                    name="agentic.explanation_quality",
                    value=overall,
                    comment=comment,
                )
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] Error: {e}")

        time.sleep(0.5)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n  Avg explanation quality: {avg:.2f}/5.0 ({len(scores)} scored)")
    return {"avg_explanation_quality": avg, "scored": len(scores)}


# ── Eval 4: False Positive Rate ───────────────────────
def run_false_positive_eval(rows: list[dict], dry_run: bool = False) -> dict:
    """
    False positive rate: auto-posted invoices that were incorrect.
    Checks routing=auto_post AND correct=False.
    """
    print("\n── Eval 4: False Positive Rate ────────────────────")
    pred_rows = [r for r in rows if r.get("name") == "invoice_prediction"]

    auto_posts = [r for r in pred_rows if r.get("routing_decision") == "auto_post"]
    false_positives = [
        r for r in auto_posts
        if r.get("correct", "").lower() in ("false", "0", "no") and r.get("correct", "") != ""
    ]

    rate = len(false_positives) / len(auto_posts) if auto_posts else 0
    print(f"  Auto-posts: {len(auto_posts)} | False positives: {len(false_positives)} | Rate: {rate:.1%}")

    if false_positives:
        print("  False positive details:")
        for r in false_positives[:5]:
            print(f"    Vendor: {r.get('vendor_raw','')[:30]} | GL: {r.get('predicted_gl')} | Conf: {r.get('weighted_confidence')}")

    # Post summary score — one score per run (use first obs as anchor)
    if not dry_run and auto_posts:
        first = auto_posts[0]
        post_score(
            trace_id=first.get("trace_id", ""),
            observation_id=first.get("observation_id", ""),
            name="gate.false_positive_rate",
            value=round(rate, 4),
            comment=f"{len(false_positives)} false positives out of {len(auto_posts)} auto-posts",
        )

    return {"false_positive_rate": rate, "auto_posts": len(auto_posts), "false_positives": len(false_positives)}


# ── Main ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run custom Langfuse evals for AP Automation")
    parser.add_argument("--export-path", default="data/langfuse_analysis/all_observations.csv", help="Path to Langfuse CSV export")
    parser.add_argument("--dry-run", action="store_true", help="Run evals without posting scores to Langfuse")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sample rate for explanation eval (default 0.1 = 10%%)")
    parser.add_argument("--skip-explanation", action="store_true", help="Skip explanation quality eval (requires OpenAI API)")
    args = parser.parse_args()

    print("AP Automation — Custom Langfuse Evaluators")
    print(f"Export: {args.export_path}")
    print(f"Dry run: {args.dry_run}")
    print()

    rows = load_observations(args.export_path)
    if not rows:
        return

    results = {}
    results["recall"] = run_recall_eval(rows, dry_run=args.dry_run)
    results["calibration"] = run_calibration_eval(rows, dry_run=args.dry_run)
    if not args.skip_explanation:
        results["explanation"] = run_explanation_eval(rows, sample_rate=args.sample_rate, dry_run=args.dry_run)
    results["false_positive"] = run_false_positive_eval(rows, dry_run=args.dry_run)

    print("\n" + "=" * 55)
    print("EVAL SUMMARY")
    print("=" * 55)
    if "recall" in results:
        print(f"Recall@10:              {results['recall']['recall_at_10']:.1%}")
    if "false_positive" in results:
        print(f"False positive rate:    {results['false_positive']['false_positive_rate']:.1%}")
    if "explanation" in results and results["explanation"]:
        print(f"Explanation quality:    {results['explanation']['avg_explanation_quality']:.2f}/5.0")
    print()
    if args.dry_run:
        print("DRY RUN — no scores posted to Langfuse")
    else:
        print("Scores posted to Langfuse — check cloud.langfuse.com → Scores")
    print("=" * 55)

    out_path = Path("data/langfuse_analysis/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
