"""
Configuration — loaded from .env via pydantic-settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "invoice_corpus"
    erp_db_path: str = "data/erp_ledger.db"
    audit_db_path: str = "data/audit_trail.db"
    google_sheets_credentials: str = "credentials/google_service_account.json"
    google_sheet_id: str = ""

    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    top_k: int = 10
    rrf_k: int = 60
    min_evidence_lines: int = 3
    default_confidence_threshold: float = 0.50
    auto_post_user_id: str = "system"


settings = Settings()
