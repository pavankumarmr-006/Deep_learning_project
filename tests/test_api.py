
from fastapi.testclient import TestClient
import app.main as main


class FakePredictor:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def predict(self, text):
        return self.label, self.confidence


def fake_load_models():
    return {
        "category": FakePredictor("Billing", 0.96),
        "priority": FakePredictor("High", 0.90),
    }


def make_client(monkeypatch):
    monkeypatch.setattr(main, "load_models", fake_load_models)
    return TestClient(main.app)


def test_health(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_predict(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/predict", json={"text": "I was charged twice"}
        )
    assert response.status_code == 200
    assert response.json()["category"] == "Billing"
    assert response.json()["priority"] == "High"
    assert 0 <= response.json()["confidence"] <= 1


def test_empty_text(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_whitespace_text(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_missing_text(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422


def test_batch(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/predict-batch",
            json={
                "tickets": [
                    {"text": "I was charged twice"},
                    {"text": "I cannot log in"},
                ]
            }
        )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2

