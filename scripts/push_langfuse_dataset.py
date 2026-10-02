"""
Push confirmed invoice predictions to a Langfuse dataset for experiment tracking.
Creates a dataset called 'ap-automation-confirmed' and populates it with
invoice predictions that have ground truth GL codes.

Usage:
    cd C:\\Users\\Miles\\Desktop\\Projects\\ap-automation
    python scripts/push_langfuse_dataset.py --export-path "C:\\Users\\Miles\\Downloads\\<latest-lf-events-export>.csv"

After running:
    - Go to Langfuse â†’ Datasets â†’ ap-automation-confirmed
    - Run experiments against it to track accuracy over time
    - Compare batches side by side
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

PUBLIC_KEY = "pk-lf-beb692d4-53fe-4246-8922-a6347ba30b55"
SECRET_KEY = "sk-lf-deef0ed8-59cb-45f7-92f8-807ea0c2502e"
HOST = "https://cloud.langfuse.com"
AUTH = HTTPBasicAuth(PUBLIC_KEY, SECRET_KEY)
DATASET_NAME = "ap-automation-confirmed"


def get_or_create_dataset() -> str:
    """Get existing dataset or create it. Returns dataset name."""
    # Try to get existing
    r = requests.get(
        f"{HOST}/api/public/datasets/{DATASET_NAME}",
        auth=AUTH,
        timeout=15,
    )
    if r.status_code == 200:
        print(f"Dataset '{DATASET_NAME}' already exists.")
        return DATASET_NAME

    # Create it
    r = requests.post(
        f"{HOST}/api/public/datasets",
        auth=AUTH,
        json={
            "name": DATASET_NAME,
            "description": "Confirmed AP invoice predictions with ground truth GL codes. Used for Recall@10, calibration and accuracy tracking across batches.",
            "metadata": {"project": "ap-automation", "version": "1.0"},
        },
        timeout=15,
    )
    if r.status_code in (200, 201):
        print(f"Created dataset '{DATASET_NAME}'.")
        return DATASET_NAME
    else:
        print(f"Error creating dataset: {r.status_code} {r.text[:200]}")
        return ""


def push_items(export_path: str, limit: int = 500, dry_run: bool = False) -> int:
    """Push confirmed invoice predictions as dataset items."""
    path = Path(export_path)
    if not path.exists():
        print(f"Export file not found: {export_path}")
        return 0

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    pred_rows = [r for r in rows if r["name"] == "invoice_prediction"]
    print(f"Loaded {len(pred_rows)} prediction rows from export.")

    # Filter to rows with ground truth GL
    scoreable = []
    for r in pred_rows:
        inp = json.loads(r["input"]) if r["input"] else {}
        out = json.loads(r["output"]) if r["output"] else {}
        gt = inp.get("ground_truth_gl", "")
        pred_gl = out.get("predicted_gl", "")
        conf = out.get("weighted_confidence", 0)
        routing = out.get("routing_decision", "")
        if gt and pred_gl:
            scoreable.append({
                "trace_id": r["traceId"],
                "observation_id": r["id"],
                "input": {
                    "invoice_id": inp.get("invoice_id", ""),
                    "vendor_raw": inp.get("vendor_raw", ""),
                    "description": inp.get("description", ""),
                    "amount": inp.get("amount", ""),
                    "currency": inp.get("currency", "GBP"),
                    "entity_id": inp.get("entity_id", "UK001"),
                },
                "expected_output": {
                    "gl_account": gt,
                },
                "metadata": {
                    "predicted_gl": pred_gl,
                    "weighted_confidence": conf,
                    "routing_decision": routing,
                    "correct": pred_gl == gt,
                    "source_observation_id": r["id"],
                },
            })

    print(f"Found {len(scoreable)} rows with ground truth GL.")
    to_push = scoreable[:limit]
    print(f"Pushing {len(to_push)} items to dataset '{DATASET_NAME}'...")

    if dry_run:
        print("DRY RUN â€” no items pushed.")
        return len(to_push)

    pushed = 0
    errors = 0
    for i, item in enumerate(to_push):
        r = requests.post(
            f"{HOST}/api/public/dataset-items",
            auth=AUTH,
            json={
                "datasetName": DATASET_NAME,
                "input": item["input"],
                "expectedOutput": item["expected_output"],
                "metadata": item["metadata"],
                "sourceTraceId": item["trace_id"],
                "sourceObservationId": item["observation_id"],
            },
            timeout=15,
        )
        if r.status_code in (200, 201):
            pushed += 1
        else:
            errors += 1
            if errors <= 3:
                print(f"  Error on item {i}: {r.status_code} {r.text[:100]}")
        if i % 50 == 0 and i > 0:
            print(f"  Progress: {i}/{len(to_push)}...")
        time.sleep(0.05)

    print(f"\nPushed: {pushed} | Errors: {errors}")
    return pushed


def print_summary(pushed: int) -> None:
    print("\n" + "=" * 55)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Items pushed: {pushed}")
    print(f"\nNext steps:")
    print(f"  1. Go to cloud.langfuse.com â†’ Datasets â†’ {DATASET_NAME}")
    print(f"  2. Click 'Run experiment' to score a new batch against it")
    print(f"  3. Compare experiment runs over time to track accuracy")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Push confirmed invoices to Langfuse dataset")
    parser.add_argument(
        "--export-path",
        default=r"C:\Users\Miles\Downloads\1790935341278-lf-events-export-cmupfurad010pad0dt7xvbusr.csv",
        help="Path to Langfuse events CSV export",
    )
    parser.add_argument("--limit", type=int, default=500, help="Max items to push (default 500)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without pushing")
    args = parser.parse_args()

    print("AP Automation â€” Push Langfuse Dataset")
    print(f"Export: {args.export_path}")
    print(f"Dry run: {args.dry_run}")
    print()

    dataset_name = get_or_create_dataset()
    if not dataset_name:
        return

    pushed = push_items(args.export_path, limit=args.limit, dry_run=args.dry_run)
    print_summary(pushed)


if __name__ == "__main__":
    main()

