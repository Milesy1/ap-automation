# AP Automation — Self-Learning Non-PO Invoice Coding

POC demonstrating self-learning GL code prediction via RAG, confidence gating, and a write-back loop.

## What this proves

- **No PS bottleneck** — automation rate improves across the batch through write-back alone
- **Novelty handled** — cold-start and hard-edge invoices route to review correctly
- **Governance proportional** — confidence gate routes uncertain predictions to human review
- **Compliance-grade audit trail** — full provenance JSON per prediction

## Quick start

### Prerequisites
- Python 3.12+
- Docker Desktop running
- OpenAI API key

### 1. Install dependencies
```bash
cd C:\Users\Miles\Desktop\Projects\ap-automation
pip install -e .
```

### 2. Add your OpenAI API key
Edit `.env` and replace `your-key-here` with your actual key.

### 3. Qdrant is already running
```bash
docker ps  # confirm qdrant container is up on port 6333
```

### 4. Start the mock ERP
```bash
python -m ap_automation.api.erp_mock
```

### 5. Start the Streamlit UI
```bash
streamlit run ap_automation/ui/app.py
```

### 6. In the UI
1. Go to **Setup** tab → Generate dataset → Seed corpus
2. Go to **Single Invoice** tab → submit invoices and confirm
3. Go to **Batch Processing** tab → upload `data/invoices.csv` → run batch
4. Watch the automation rate climb in **Automation Rate** tab

## Project structure

```
ap_automation/
  core/
    models.py          # Pydantic data models
    config.py          # Settings from .env
    mdm.py             # Vendor resolution (MDM stub)
    preprocessing.py   # Description cleaning pipeline
    embeddings.py      # OpenAI embedding wrapper
    retrieval.py       # Hybrid retrieval (dense + BM25 + RRF + re-rank)
    confidence.py      # Confidence scoring and routing gate
    pipeline.py        # Main prediction pipeline
    database.py        # SQLite persistence
    dataset.py         # Synthetic dataset generator
    sheets.py          # Google Sheets mirror
  api/
    erp_mock.py        # FastAPI mock ERP endpoint
  ui/
    app.py             # Streamlit demo UI
data/
  invoices.csv         # Generated on first run
  vendor_map.csv       # Vendor → canonical ID mapping
tests/
notebooks/
```

## Google Sheets (optional)
1. Create a Google Cloud project and enable the Sheets API
2. Create a service account and download the JSON key
3. Place it at `credentials/google_service_account.json`
4. Create a Google Sheet and share it with the service account email
5. Add the Sheet ID to `.env` → `GOOGLE_SHEET_ID`

## Running tests
```bash
pytest tests/
```
