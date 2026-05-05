"""Tests for GET /health and GET / endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(classifier_instance):
    from app.main import app
    from app.dependencies import get_classifier

    app.dependency_overrides[get_classifier] = lambda: classifier_instance
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_status_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_has_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data and isinstance(data["model_version"], str)


class TestRootEndpoint:
    def test_status_200(self, client):
        assert client.get("/").status_code == 200

    def test_has_service_key(self, client):
        assert "service" in client.get("/").json()

    def test_has_version_key(self, client):
        assert "version" in client.get("/").json()
