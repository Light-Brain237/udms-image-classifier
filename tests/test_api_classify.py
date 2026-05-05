"""Tests for POST /api/v1/classify endpoint."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

VALID_CATEGORIES = {
    "bad_drainage", "damaged_signage", "illegal_dumping",
    "potholes", "vegetation_overgrowth",
}


def _jpeg(w=224, h=224, color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _png(w=224, h=224, color="blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(classifier_instance):
    from app.main import app
    from app.dependencies import get_classifier

    app.dependency_overrides[get_classifier] = lambda: classifier_instance
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestClassifySuccess:
    def test_jpeg_returns_200(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 200

    def test_png_returns_200(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.png", _png(), "image/png")})
        assert r.status_code == 200

    def test_webp_returns_200(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), "purple").save(buf, format="WEBP")
        r = client.post("/api/v1/classify", files={"file": ("t.webp", buf.getvalue(), "image/webp")})
        assert r.status_code == 200

    def test_response_has_prediction(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert "prediction" in r.json()

    def test_response_has_alternatives(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert "alternatives" in r.json()

    def test_response_has_model_version(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert "model_version" in r.json()

    def test_response_has_inference_time(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert "inference_time_ms" in r.json()

    def test_prediction_fields(self, client):
        pred = client.post(
            "/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")}
        ).json()["prediction"]
        assert {"category", "category_label", "confidence", "requires_review"} <= pred.keys()

    def test_category_is_valid(self, client):
        cat = client.post(
            "/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")}
        ).json()["prediction"]["category"]
        assert cat in VALID_CATEGORIES

    def test_confidence_between_0_and_1(self, client):
        conf = client.post(
            "/api/v1/classify", files={"file": ("t.jpg", _jpeg(), "image/jpeg")}
        ).json()["prediction"]["confidence"]
        assert 0.0 <= conf <= 1.0


class TestClassifyErrors:
    def test_invalid_content_type_returns_400(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.pdf", b"pdf", "application/pdf")})
        assert r.status_code == 400

    def test_400_detail_mentions_file_type(self, client):
        r = client.post("/api/v1/classify", files={"file": ("t.pdf", b"pdf", "application/pdf")})
        assert "file type" in r.json()["detail"].lower()
