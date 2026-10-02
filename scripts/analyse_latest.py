import csv, json
from collections import defaultdict

path = r"C:\Users\Miles\Downloads\1790935341278-lf-events-export-cmupfurad010pad0dt7xvbusr.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
preds = [r for r in rows if r["name"] == "invoice_prediction"]
confs = [r for r in rows if r["name"] == "invoice_confirmation"]

print(f"Total rows: {len(rows)}")
print(f"Predictions: {len(preds)}")
print(f"Confirmations: {len(confs)}")

# Recall@10 on latest 200
latest = sorted(preds, key=lambda r: r["startTime"])[-200:]
has_gt = 0
recall_scores = []
for r in latest:
    inp = json.loads(r["input"]) if r["input"] else {}
    gt = inp.get("ground_truth_gl", "")
    out = json.loads(r["output"]) if r["output"] else {}
    pred_gl = out.get("predicted_gl", "")
    if gt:
        has_gt += 1
        recall_scores.append(1.0 if pred_gl == gt else 0.0)

print(f"Latest 200 with ground_truth_gl: {has_gt}")
if recall_scores:
    print(f"Recall@10: {sum(recall_scores)/len(recall_scores):.1%}")

# Calibration from confirmations
bands = defaultdict(lambda: {"correct": 0, "total": 0})
for r in confs:
    inp = json.loads(r["input"]) if r["input"] else {}
    meta = json.loads(r["metadata"]) if r["metadata"] else {}
    gt = inp.get("ground_truth_gl", "")
    pred = inp.get("predicted_gl", "")
    conf = float(meta.get("confidence", 0) or 0)
    if not gt:
        continue
    band = str(int(conf*10)*10) + "-" + str(int(conf*10)*10+10) + "pct"
    bands[band]["total"] += 1
    if pred == gt:
        bands[band]["correct"] += 1

print()
print("Calibration:")
for band in sorted(bands):
    d = bands[band]
    if d["total"] < 2:
        continue
    acc = d["correct"] / d["total"]
    print(f"  {band}: {acc:.0%} actual ({d['total']} samples)")
