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
def mock_tflite_model(tmp_path):
    """Create a minimal valid TFLite model for testing.

    This model outputs random 7-class probabilities.
    It validates that the classifier wrapper works correctly
    WITHOUT needing a trained model.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    model_path = tmp_path / "test_model.tflite"
    model_path.write_bytes(tflite_bytes)
    return str(model_path)


@pytest.fixture
def mock_label_map(tmp_path):
    """Create a valid label_map.json for testing."""
    label_map = {
        str(i): {"category": cat, "label": f"Test {cat}"}
        for i, cat in enumerate([
            "illegal_dumping", "pothole_road", "broken_lighting",
            "water_sewage", "damaged_signage", "vegetation", "other",
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
 
