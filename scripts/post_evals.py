"""Run custom evals against the Langfuse export and post scores."""
import csv, json, requests, time
from collections import defaultdict
from requests.auth import HTTPBasicAuth

EXPORT_PATH = r"C:\Users\Miles\Downloads\1790863343494-lf-events-export-cmupfurad010pad0dt7xvbusr.csv"
AUTH = HTTPBasicAuth("pk-lf-beb692d4-53fe-4246-8922-a6347ba30b55", "sk-lf-deef0ed8-59cb-45f7-92f8-807ea0c2502e")
HOST = "https://cloud.langfuse.com"

rows = list(csv.DictReader(open(EXPORT_PATH, encoding="utf-8")))
pred = [r for r in rows if r["name"] == "invoice_prediction"]
print(f"Loaded {len(rows)} rows, {len(pred)} predictions")

def post_score(trace_id, obs_id, name, value, comment=""):
    r = requests.post(f"{HOST}/api/public/scores", auth=AUTH, json={
        "name": name, "value": value, "traceId": trace_id,
        "observationId": obs_id, "comment": comment
    }, timeout=15)
    return r.status_code in (200, 201)

# ── Eval 1: Recall@10 ─────────────────────────────────
print("\n=== Eval 1: Recall@10 ===")
scored = correct = 0
for r in pred[:500]:
    gate = r.get("gate.correct", "")
    if gate == "": continue
    out = json.loads(r["output"]) if r["output"] else {}
    recall = 1.0 if gate in ("1", "1.0") else 0.0
    scored += 1
    if recall == 1.0: correct += 1
    gl = out.get("predicted_gl", "")
    ok = post_score(r["traceId"], r["id"], "retrieval.recall_at_10", recall, f"GL:{gl} correct:{gate}")
    time.sleep(0.05)

print(f"  Scored: {scored} | Recall@10: {correct/scored:.1%}" if scored else "  No scored rows")

# ── Eval 2: False Positive Rate ───────────────────────
print("\n=== Eval 2: False Positive Rate ===")
auto = []
for r in pred:
    out = json.loads(r["output"]) if r["output"] else {}
    if out.get("routing_decision") == "auto_post":
        auto.append((r, out))

fp = [(r, out) for r, out in auto if r.get("gate.correct", "") in ("0", "0.0")]
rate = len(fp) / len(auto) if auto else 0
print(f"  Auto-posts: {len(auto)} | False positives: {len(fp)} | Rate: {rate:.1%}")
if auto:
    r0, _ = auto[0]
    post_score(r0["traceId"], r0["id"], "gate.false_positive_rate", round(rate, 4),
               f"{len(fp)} FP out of {len(auto)} auto-posts")

# ── Eval 3: Calibration ───────────────────────────────
print("\n=== Eval 3: Calibration ===")
bands = defaultdict(lambda: {"correct": 0, "total": 0})
for r in pred:
    out = json.loads(r["output"]) if r["output"] else {}
    gate = r.get("gate.correct", "")
    if gate == "": continue
    conf = float(out.get("weighted_confidence", 0))
    band = f"{int(conf*10)*10}-{int(conf*10)*10+10}pct"
    bands[band]["total"] += 1
    if gate in ("1", "1.0"): bands[band]["correct"] += 1

print(f"  {'Band':<20} {'Conf':>8} {'Actual':>8} {'Gap':>8} {'n':>6}")
for band in sorted(bands):
    d = bands[band]
    if d["total"] < 3: continue
    mid = int(band.split("-")[0]) / 100 + 0.05
    acc = d["correct"] / d["total"]
    gap = abs(mid - acc)
    print(f"  {band:<20} {mid:>7.0%} {acc:>7.0%} {gap:>7.0%} {d['total']:>6}")

print("\n=== DONE — scores posted to Langfuse ===")
print("Check cloud.langfuse.com → Scores tab")
