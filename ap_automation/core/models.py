"""
Pydantic models — shared data structures across the entire pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class InvoiceCategory(str, Enum):
    ROUTINE = "routine"
    NEAR_EDGE = "near_edge"
    HARD_EDGE = "hard_edge"


class RoutingDecision(str, Enum):
    AUTO_POST = "auto_post"
    HUMAN_REVIEW = "human_review"


class ConfirmationSource(str, Enum):
    AUTO_POST = "auto_post"
    HUMAN_CONFIRM = "human_confirm"
    HUMAN_CORRECT = "human_correct"


class AmountBand(str, Enum):
    MICRO = "micro"        # < 100
    SMALL = "small"        # 100–999
    MEDIUM = "medium"      # 1000–9999
    LARGE = "large"        # 10000–49999
    CAPEX = "capex"        # >= 50000


def amount_to_band(amount: float) -> AmountBand:
    if amount < 100:
        return AmountBand.MICRO
    elif amount < 1000:
        return AmountBand.SMALL
    elif amount < 10000:
        return AmountBand.MEDIUM
    elif amount < 50000:
        return AmountBand.LARGE
    else:
        return AmountBand.CAPEX


class InvoiceLine(BaseModel):
    """Raw invoice line as it arrives from the extraction pipeline."""
    invoice_id: UUID = Field(default_factory=uuid4)
    raw_vendor_name: str
    description: str
    amount: float
    currency: str = "GBP"
    entity_id: str
    cost_centre: str | None = None
    category: InvoiceCategory = InvoiceCategory.ROUTINE
    ground_truth_gl: str | None = None  # for POC evaluation only


class ResolvedInvoiceLine(BaseModel):
    """Invoice line after MDM vendor resolution."""
    invoice_line: InvoiceLine
    canonical_vendor_id: str
    mdm_confidence: float
    amount_band: AmountBand


class RetrievedEvidence(BaseModel):
    """A single historically-coded line retrieved from Qdrant."""
    invoice_id: str
    description: str
    gl_account: str
    cost_centre: str
    vendor_id: str
    similarity_score: float
    dense_rank: int
    sparse_rank: int
    rrf_score: float
    confirmation_timestamp: datetime
    source: ConfirmationSource


class Prediction(BaseModel):
    """Full prediction output for one invoice line."""
    prediction_id: UUID = Field(default_factory=uuid4)
    resolved_line: ResolvedInvoiceLine
    evidence: list[RetrievedEvidence]
    predicted_gl: str
    predicted_cost_centre: str
    agreement_ratio: float
    weighted_confidence: float
    routing_decision: RoutingDecision
    routing_reason: str
    provenance: dict  # full JSON audit record


class ConfirmedOutcome(BaseModel):
    """A confirmed invoice coding — either auto-posted or human-confirmed/corrected."""
    prediction_id: UUID
    invoice_id: UUID
    canonical_vendor_id: str
    entity_id: str
    amount_band: AmountBand
    currency: str
    gl_account: str
    cost_centre: str
    tax_code: str = "STANDARD"
    confirming_user_id: str
    confirmation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_confidence: float
    source: ConfirmationSource
    description: str


class ERPPostingPayload(BaseModel):
    """Payload sent to the mock ERP endpoint."""
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
    posted_at: datetime
    trace_id: str
