from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simulate_endpoint(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={"text": "sample narrative about a trending topic", "steps": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert "graph_stats" in body
    assert "propagation_metrics" in body
    assert body["graph_stats"]["nodes"] > 0


def test_predict_endpoint(client: TestClient) -> None:
    response = client.post("/predict", json={"features": [0.5, 0.3, 0.4, 2.0]})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["threshold_used"] <= 1.0


def test_predict_endpoint_rejects_wrong_length_feature_vector(client: TestClient) -> None:
    response = client.post("/predict", json={"features": [0.5, 0.3, 0.4]})
    assert response.status_code == 422


def test_explain_endpoint(client: TestClient) -> None:
    response = client.post("/explain", json={"features": [0.5, 0.3, 0.4, 2.0]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_features"]) == 4
    assert len(body["importance_scores"]) == 4
