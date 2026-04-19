"""
UDMS API Schemas — Pydantic v2 request/response models.
"""

from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Primary prediction with review flag."""

    category: str
    category_label: str
    confidence: float
    requires_review: bool


class AlternativeResult(BaseModel):
    """Runner-up prediction (category + confidence only)."""

    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    """Full classification response returned by POST /api/v1/classify."""

    prediction: PredictionResult
    alternatives: list[AlternativeResult]
    model_version: str
    inference_time_ms: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str

