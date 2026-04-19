"""
UDMS Health Check Endpoint.
"""

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_classifier
from src.inference.classifier import UDMSClassifier

router = APIRouter()


@router.get("/health")
def health_check(classifier: UDMSClassifier = Depends(get_classifier)) -> dict:
    """Return service health status and model metadata."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": settings.MODEL_VERSION,
    }

