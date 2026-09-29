"""
Google Sheets integration — mirrors confirmed ERP postings to a live Sheet.
"""

from __future__ import annotations

from pathlib import Path

from ap_automation.core.config import settings


def _get_service():
    """Build the Google Sheets API service."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds_path = Path(settings.google_sheets_credentials)
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Google service account credentials not found at {creds_path}. "
            "Add credentials/google_service_account.json to enable Sheets mirroring."
        )

    creds = Credentials.from_service_account_file(
        str(creds_path),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds)


def ensure_sheet_headers() -> None:
    """Write column headers to the Sheet if not already present."""
    if not settings.google_sheet_id:
        return

    headers = [
        "Invoice ID", "Vendor ID", "Entity", "GL Account",
        "Cost Centre", "Tax Code", "Amount", "Currency",
        "Description", "Posted By", "Posted At", "Trace ID",
    ]
    service = _get_service()
    service.spreadsheets().values().update(
        spreadsheetId=settings.google_sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        body={"values": [headers]},
    ).execute()


def post_to_sheet(payload: dict) -> None:
    """Append a confirmed invoice posting to the Google Sheet."""
    if not settings.google_sheet_id:
        return

    row = [
        payload.get("invoice_id", ""),
        payload.get("vendor_id", ""),
        payload.get("entity_id", ""),
        payload.get("gl_account", ""),
        payload.get("cost_centre", ""),
        payload.get("tax_code", ""),
        payload.get("amount", ""),
        payload.get("currency", ""),
        payload.get("description", ""),
        payload.get("posted_by", ""),
        payload.get("posted_at", ""),
        payload.get("trace_id", ""),
    ]

    service = _get_service()
    service.spreadsheets().values().append(
        spreadsheetId=settings.google_sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()
