"""
UDMS Image Classifier — Shared test fixtures.

Provides mock model, label map, sample images, and classifier instance
so tests can run before a real trained model exists.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mock_tflite_model():
    """Return path to the real TFLite model for testing.

    Uses the trained model artifact so tests don't require a live
    Keras/TensorFlow build environment.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "classifier.tflite")


@pytest.fixture
def mock_label_map(tmp_path):
    """Create a valid label_map.json for testing."""
    label_map = {
        str(i): {"category": cat, "label": f"Test {cat}"}
        for i, cat in enumerate([
            "bad_drainage", "damaged_signage", "illegal_dumping",
            "potholes", "vegetation_overgrowth",
        ])
    }
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps(label_map))
    return str(path)


@pytest.fixture
def sample_image_bytes():
    """Create a valid JPEG image as bytes for testing."""
    img = Image.new("RGB", (640, 480), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def corrupted_bytes():
    """Bytes that are NOT a valid image."""
    return b"this is not an image file at all"


@pytest.fixture
def classifier_instance(mock_tflite_model, mock_label_map):
    """UDMSClassifier loaded with mock model for testing."""
    from src.inference.classifier import UDMSClassifier

    return UDMSClassifier(
        model_path=mock_tflite_model,
        label_map_path=mock_label_map,
        confidence_threshold=0.6,
    )
 
