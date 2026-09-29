"""
Mock ERP endpoint — accepts confirmed invoice coding payloads,
writes to SQLite ledger, and mirrors to Google Sheets.
"""

from __future__ import annotations

from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ap_automation.core.database import init_db, post_to_erp_ledger, write_audit
from ap_automation.core.sheets import post_to_sheet

app = FastAPI(
    title="AP Automation — Mock ERP",
    description="Simulates an ERP invoice posting endpoint for the POC.",
    version="0.1.0",
)


class PostingRequest(BaseModel):
    invoice_id: str
    vendor_id: str
    entity_id: str
    gl_account: str
    cost_centre: str
    tax_code: str
    amount: float
    currency: str
    description: str
    posted_by: str
    posted_at: str
    trace_id: str


class AuditRequest(BaseModel):
    prediction_id: str
    invoice_id: str
    vendor_id: str
    predicted_gl: str
    confirmed_gl: str
    confidence: float
    routing_decision: str
    routing_reason: str
    source: str
    confirming_user: str
    provenance: dict


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/post-invoice", status_code=201)
def post_invoice(payload: PostingRequest) -> dict:
    """
    Accept a confirmed invoice coding and write it to the ERP ledger.
    Also mirrors to Google Sheets if configured.
    """
    try:
        data = payload.model_dump()
        post_to_erp_ledger(data)

        # Mirror to Google Sheets (best-effort — does not fail the posting)
        try:
            post_to_sheet(data)
        except Exception:
            pass

        return {
            "status": "posted",
            "invoice_id": payload.invoice_id,
            "trace_id": payload.trace_id,
            "posted_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audit", status_code=201)
def write_audit_record(record: AuditRequest) -> dict:
    """Write a prediction audit record."""
    try:
        write_audit(record.model_dump())
        return {"status": "recorded", "prediction_id": record.prediction_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("ap_automation.api.erp_mock:app", host="0.0.0.0", port=8001, reload=True)
