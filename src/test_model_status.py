"""GET /api/model: the client's view of the live model and online learning (version, feedback, last retrain).

Run from the repo root:  PYTHONPATH=src:. .venv/bin/python -m pytest src/test_model_status.py -q
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import main, model_client
from app.api.model_client import ModelServiceError

HEALTH = {"status": "ok", "model_version": "v3", "schema_version": "1", "horizon_minutes": 60,
          "feedback_samples": 10, "new_feedback": 4, "min_new_feedback": 50, "retraining": True, "last_retrain": {"status": "promoted", "reason": "", "version": "v3",
                                                   "new_samples": 10, "at": "2026-11-26T12:50:00+00:00"}}


def test_status_passes_on_the_model_health():
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json=HEALTH)

    status = model_client.get_model_status(transport=httpx.MockTransport(handler))
    assert status == {"model_version": "v3", "horizon_minutes": 60, "feedback_samples": 10, "new_feedback": 4,
                      "min_new_feedback": 50, "retraining": True, "last_retrain": {"status": "promoted", "reason": "", "version": "v3",
                                       "at": "2026-11-26T12:50:00+00:00"}}


def test_status_before_any_retrain_has_no_report():
    health = {**HEALTH, "model_version": "v2", "feedback_samples": 0, "last_retrain": None}
    status = model_client.get_model_status(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=health)))
    assert status["last_retrain"] is None and status["model_version"] == "v2"


def test_status_raises_when_the_model_service_fails():
    with pytest.raises(ModelServiceError):
        model_client.get_model_status(transport=httpx.MockTransport(lambda r: httpx.Response(500)))


def test_endpoint_returns_the_status(monkeypatch):
    monkeypatch.setattr(main, "get_model_status", lambda: {"model_version": "v2", "horizon_minutes": 60,
                                                           "feedback_samples": 0, "last_retrain": None})
    response = TestClient(main.app).get("/api/model")
    assert response.status_code == 200 and response.json()["model_version"] == "v2"


def test_endpoint_is_503_when_the_model_service_is_down(monkeypatch):
    def down():
        raise ModelServiceError("Model service is unavailable")

    monkeypatch.setattr(main, "get_model_status", down)
    response = TestClient(main.app).get("/api/model")
    assert response.status_code == 503 and "unavailable" in response.json()["detail"]


def test_start_retrain_asks_the_model_service():
    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1/retrain/start"
        return httpx.Response(202, json={"started": True})

    assert model_client.start_retrain(transport=httpx.MockTransport(handler)) is True


def test_start_retrain_reports_a_running_one():
    with pytest.raises(model_client.RetrainRunning):
        model_client.start_retrain(transport=httpx.MockTransport(lambda r: httpx.Response(409)))


def test_retrain_endpoint(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "start_retrain", lambda: True)
    assert client.post("/api/model/retrain").json() == {"started": True}

    def running():
        raise model_client.RetrainRunning("a retrain is already running")

    monkeypatch.setattr(main, "start_retrain", running)
    assert client.post("/api/model/retrain").status_code == 409

    def down():
        raise ModelServiceError("Model service is unavailable")

    monkeypatch.setattr(main, "start_retrain", down)
    assert client.post("/api/model/retrain").status_code == 503
