
import pytest
from fastapi.testclient import TestClient

from backend.rl.inference_api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert data["observations"] == 15
    assert data["actions"] == 5


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"observation": [0.0] * 15},
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["action"], int)
    assert 0 <= data["action"] <= 4
    assert isinstance(data["decision"], str)


@pytest.mark.parametrize(
    "observation",
    [
        [],
        [0.0] * 14,
        [0.0] * 16,
        [2.0] + [0.0] * 14,
        [-2.0] + [0.0] * 14,
        ["invalid"] + [0.0] * 14,
    ],
)
def test_invalid_observations_rejected(observation):
    response = client.post(
        "/predict",
        json={"observation": observation},
    )
    assert response.status_code == 422


def test_missing_observation_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
