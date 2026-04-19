"""
UDMS Dependency Injection — Singleton classifier loaded once at startup.
"""

import logging

import numpy as np

from src.inference.classifier import UDMSClassifier
from app.config import settings

logger = logging.getLogger(__name__)

_classifier: UDMSClassifier | None = None


def load_classifier() -> None:
    """Instantiate the classifier singleton. Called once during app startup."""
    global _classifier
    _classifier = UDMSClassifier(
        model_path=settings.MODEL_PATH,
        label_map_path=settings.LABEL_MAP_PATH,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
    )
    _warmup(_classifier)


def _warmup(classifier: UDMSClassifier) -> None:
    """Run one dummy prediction so TFLite fully initialises its internals."""
    dummy = np.zeros((224, 224, 3), dtype=np.float32)
    dummy_bytes = _numpy_to_png_bytes(dummy)
    classifier.predict(dummy_bytes)
    logger.info("Warmup inference completed")


def _numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy image array to PNG bytes for the warmup call."""
    from PIL import Image
    import io

    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_classifier() -> UDMSClassifier:
    """FastAPI dependency — returns the pre-loaded classifier instance."""
    if _classifier is None:
        raise RuntimeError("Classifier not loaded. Was startup event skipped?")
    return _classifier

