"""Edge case tests for preprocessing, classifier, and API."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _jpeg(w=224, h=224, color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(classifier_instance):
    from app.main import app
    from app.dependencies import get_classifier

    app.dependency_overrides[get_classifier] = lambda: classifier_instance
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


class TestPreprocessingEdgeCases:
    def test_tiny_image(self):
        from src.data.preprocessing import preprocess_bytes
        buf = io.BytesIO()
        Image.new("RGB", (5, 5), "red").save(buf, format="JPEG")
        assert preprocess_bytes(buf.getvalue()).shape == (224, 224, 3)

    def test_wide_image(self):
        from src.data.preprocessing import preprocess_bytes
        buf = io.BytesIO()
        Image.new("RGB", (2000, 50), "green").save(buf, format="JPEG")
        assert preprocess_bytes(buf.getvalue()).shape == (224, 224, 3)

    def test_tall_image(self):
        from src.data.preprocessing import preprocess_bytes
        buf = io.BytesIO()
        Image.new("RGB", (50, 2000), "blue").save(buf, format="JPEG")
        assert preprocess_bytes(buf.getvalue()).shape == (224, 224, 3)

    def test_square_224_unchanged(self):
        from src.data.preprocessing import preprocess_bytes
        import numpy as np
        arr = preprocess_bytes(_jpeg(224, 224))
        assert arr.shape == (224, 224, 3)


class TestAPIEdgeCases:
    def test_corrupted_bytes_returns_500(self, client):
        r = client.post(
            "/api/v1/classify",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert r.status_code == 500

    def test_empty_bytes_returns_500(self, client):
        r = client.post(
            "/api/v1/classify",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 500

    def test_no_file_returns_422(self, client):
        r = client.post("/api/v1/classify")
        assert r.status_code == 422

    def test_unsupported_type_400(self, client):
        r = client.post(
            "/api/v1/classify",
            files={"file": ("x.bmp", b"bmpdata", "image/bmp")},
        )
        assert r.status_code == 400


class TestCategoryMapping:
    def test_num_classes(self):
        from src.data.category_mapping import NUM_CLASSES
        assert NUM_CLASSES == 5

    def test_categories_count(self):
        from src.data.category_mapping import UDMS_CATEGORIES
        assert len(UDMS_CATEGORIES) == 5

    def test_expected_categories(self):
        from src.data.category_mapping import UDMS_CATEGORIES
        expected = {
            "bad_drainage", "damaged_signage", "illegal_dumping",
            "potholes", "vegetation_overgrowth",
        }
        assert set(UDMS_CATEGORIES) == expected

    def test_labels_match_categories(self):
        from src.data.category_mapping import CATEGORY_LABELS, UDMS_CATEGORIES
        assert set(CATEGORY_LABELS.keys()) == set(UDMS_CATEGORIES)

    def test_get_label_index_roundtrip(self):
        from src.data.category_mapping import UDMS_CATEGORIES, get_label_index, get_category_from_index
        for i, cat in enumerate(UDMS_CATEGORIES):
            assert get_label_index(cat) == i
            assert get_category_from_index(i) == cat
