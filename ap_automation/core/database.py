"""
SQLite persistence — ERP ledger and audit trail.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ap_automation.core.config import settings


def _get_conn(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise both databases with schema."""
    # ERP ledger
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS erp_postings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT NOT NULL,
                vendor_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                gl_account TEXT NOT NULL,
                cost_centre TEXT NOT NULL,
                tax_code TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                description TEXT NOT NULL,
                posted_by TEXT NOT NULL,
                posted_at TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

    # Audit trail
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                invoice_id TEXT NOT NULL,
                vendor_id TEXT NOT NULL,
                predicted_gl TEXT NOT NULL,
                confirmed_gl TEXT NOT NULL,
                confidence REAL NOT NULL,
                routing_decision TEXT NOT NULL,
                routing_reason TEXT NOT NULL,
                source TEXT NOT NULL,
                confirming_user TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def post_to_erp_ledger(payload: dict) -> None:
    """Write a confirmed invoice posting to the ERP ledger."""
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute("""
            INSERT INTO erp_postings
            (invoice_id, vendor_id, entity_id, gl_account, cost_centre,
             tax_code, amount, currency, description, posted_by, posted_at, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["invoice_id"],
            payload["vendor_id"],
            payload["entity_id"],
            payload["gl_account"],
            payload["cost_centre"],
            payload["tax_code"],
            payload["amount"],
            payload["currency"],
            payload["description"],
            payload["posted_by"],
            payload["posted_at"],
            payload["trace_id"],
        ))
        conn.commit()


def write_audit(record: dict) -> None:
    """Write a prediction audit record."""
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("""
            INSERT INTO audit_trail
            (prediction_id, invoice_id, vendor_id, predicted_gl, confirmed_gl,
             confidence, routing_decision, routing_reason, source, confirming_user, provenance_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["prediction_id"],
            record["invoice_id"],
            record["vendor_id"],
            record["predicted_gl"],
            record["confirmed_gl"],
            record["confidence"],
            record["routing_decision"],
            record["routing_reason"],
            record["source"],
            record["confirming_user"],
            json.dumps(record["provenance"]),
        ))
        conn.commit()


def get_postings(limit: int = 100) -> list[dict]:
    """Fetch recent ERP postings."""
    with _get_conn(settings.erp_db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM erp_postings ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_audit_records(limit: int = 100) -> list[dict]:
    """Fetch recent audit trail records."""
    with _get_conn(settings.audit_db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_trail ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
