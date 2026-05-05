"""Tests for src.inference.classifier.UDMSClassifier."""
import io

import pytest
from PIL import Image

VALID_CATEGORIES = {
    "bad_drainage", "damaged_signage", "illegal_dumping",
    "potholes", "vegetation_overgrowth",
}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color="blue").save(buf, format="JPEG")
    return buf.getvalue()


class TestPredict:
    def test_returns_dict(self, classifier_instance, sample_image_bytes):
        assert isinstance(classifier_instance.predict(sample_image_bytes), dict)

    def test_top_level_keys(self, classifier_instance, sample_image_bytes):
        result = classifier_instance.predict(sample_image_bytes)
        assert {"prediction", "alternatives", "model_version", "inference_time_ms"} <= result.keys()

    def test_prediction_keys(self, classifier_instance, sample_image_bytes):
        pred = classifier_instance.predict(sample_image_bytes)["prediction"]
        assert {"category", "category_label", "confidence", "requires_review"} <= pred.keys()

    def test_confidence_range(self, classifier_instance, sample_image_bytes):
        c = classifier_instance.predict(sample_image_bytes)["prediction"]["confidence"]
        assert isinstance(c, float) and 0.0 <= c <= 1.0

    def test_category_is_valid(self, classifier_instance, sample_image_bytes):
        cat = classifier_instance.predict(sample_image_bytes)["prediction"]["category"]
        assert cat in VALID_CATEGORIES

    def test_requires_review_is_bool(self, classifier_instance, sample_image_bytes):
        assert isinstance(
            classifier_instance.predict(sample_image_bytes)["prediction"]["requires_review"],
            bool,
        )

    def test_alternatives_is_list_of_3(self, classifier_instance, sample_image_bytes):
        alts = classifier_instance.predict(sample_image_bytes)["alternatives"]
        assert isinstance(alts, list) and len(alts) == 3

    def test_alternatives_have_keys(self, classifier_instance, sample_image_bytes):
        alt = classifier_instance.predict(sample_image_bytes)["alternatives"][0]
        assert "category" in alt and "confidence" in alt

    def test_inference_time_positive(self, classifier_instance, sample_image_bytes):
        assert classifier_instance.predict(sample_image_bytes)["inference_time_ms"] > 0

    def test_requires_review_true_when_threshold_high(
        self, mock_tflite_model, mock_label_map
    ):
        from src.inference.classifier import UDMSClassifier
        clf = UDMSClassifier(mock_tflite_model, mock_label_map, confidence_threshold=0.99)
        result = clf.predict(_jpeg())
        assert result["prediction"]["requires_review"] is True

    def test_requires_review_false_when_threshold_zero(
        self, mock_tflite_model, mock_label_map
    ):
        from src.inference.classifier import UDMSClassifier
        clf = UDMSClassifier(mock_tflite_model, mock_label_map, confidence_threshold=0.0)
        result = clf.predict(_jpeg())
        assert result["prediction"]["requires_review"] is False


class TestModelInfo:
    def test_has_categories(self, classifier_instance):
        assert "categories" in classifier_instance.model_info

    def test_five_categories(self, classifier_instance):
        assert len(classifier_instance.model_info["categories"]) == 5

    def test_input_shape(self, classifier_instance):
        assert classifier_instance.model_info["input_shape"] == [224, 224, 3]
