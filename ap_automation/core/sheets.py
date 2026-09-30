"""
Google Sheets integration — mirrors confirmed ERP postings to a live Sheet.
Reads credentials from Streamlit secrets (hosted) or local JSON file (local dev).
"""

from __future__ import annotations

import json
from pathlib import Path

from ap_automation.core.config import settings


def _get_service():
    """Build the Google Sheets API service."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    # Try Streamlit secrets first (hosted)
    try:
        import streamlit as st
        creds_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if creds_json:
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return build("sheets", "v4", credentials=creds)
    except Exception:
        pass

    # Fall back to local JSON file
    creds_path = Path(settings.google_sheets_credentials)
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Google credentials not found at {creds_path}. "
            "Add GOOGLE_SERVICE_ACCOUNT_JSON to Streamlit secrets."
        )
    creds = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
    return build("sheets", "v4", credentials=creds)


def _get_sheet_id() -> str:
    """Get sheet ID from Streamlit secrets or config."""
    try:
        import streamlit as st
        sheet_id = st.secrets.get("GOOGLE_SHEET_ID", "")
        if sheet_id:
            return sheet_id
    except Exception:
        pass
    return settings.google_sheet_id


def ensure_sheet_headers() -> None:
    sheet_id = _get_sheet_id()
    if not sheet_id:
        return
    headers = [
        "Invoice ID", "Vendor ID", "Entity", "GL Account",
        "Cost Centre", "Tax Code", "Amount", "Currency",
        "Description", "Posted By", "Posted At", "Trace ID",
    ]
    try:
        service = _get_service()
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="Sheet1!A1",
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute()
    except Exception:
        pass


def post_to_sheet(payload: dict) -> None:
    sheet_id = _get_sheet_id()
    if not sheet_id:
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
        str(payload.get("posted_at", "")),
        payload.get("trace_id", ""),
    ]
    service = _get_service()
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [row]},
    ).execute()
