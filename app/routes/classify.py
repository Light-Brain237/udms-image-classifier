"""
UDMS Classification Endpoint — POST /api/v1/classify.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import settings
from app.dependencies import get_classifier
from app.schemas import ClassifyResponse, ErrorResponse
from src.inference.classifier import UDMSClassifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def classify_image(
    file: UploadFile = File(...),
    classifier: UDMSClassifier = Depends(get_classifier),
) -> ClassifyResponse:
    """Classify an uploaded image into a UDMS category."""

    # --- validate content type ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. "
            f"Allowed: jpg, jpeg, png, webp.",
        )

    # --- read and validate size ---
    image_bytes = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size {len(image_bytes)} bytes exceeds "
            f"limit of {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # --- run inference ---
    try:
        result = classifier.predict(image_bytes)
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Model inference error: {exc}",
        )

    result["model_version"] = settings.MODEL_VERSION
    return ClassifyResponse(**result)

