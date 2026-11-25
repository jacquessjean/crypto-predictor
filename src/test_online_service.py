"""The model service's online-learning endpoints, against a temporary models folder and synthetic data.
Run as a script (the service reads its settings from the environment when it is imported)."""
import importlib.util
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from cleanup import FEATURES
from test_online import samples, setup

TMP = Path(tempfile.mkdtemp())
BASE, ROOT, _ = setup(TMP)
_saved_env = dict(os.environ)
os.environ.update({"MODELS_ROOT": str(ROOT), "MODEL_DIR": str(ROOT / "v2"), "BASE_DATA": str(TMP / "features.parquet"),
                   "RETRAIN_INTERVAL_HOURS": "0", "MIN_NEW_FEEDBACK": "50",
                   "GATE_TOLERANCE": "1.0"})  # the gate itself is tested in test_online.py; here: promotion and hot swap
# a private copy of the service under its own name, so this temporary model never leaks into test_model_service
_spec = importlib.util.spec_from_file_location(
    "model_service_online_test", Path(__file__).resolve().parents[1] / "service" / "model_service.py")
model_service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_service)  # the service reads its settings now, at import
os.environ.clear()
os.environ.update(_saved_env)  # and the next test file sees the real environment again
client = TestClient(model_service.app)


def test_feedback_validates_and_dedupes():
    batch = samples(BASE, 3, offset=500)
    assert client.post("/v1/feedback", json={"samples": batch}).json() == {"accepted": 3, "duplicates": 0}
    assert client.post("/v1/feedback", json={"samples": batch}).json() == {"accepted": 0, "duplicates": 3}
    broken = samples(BASE, 1, offset=600)[0]
    del broken["features"]["ret_4"]
    assert client.post("/v1/feedback", json={"samples": [broken]}).status_code == 422
    unknown = {**samples(BASE, 1, offset=601)[0], "pair": "FAKEUSDT"}
    assert client.post("/v1/feedback", json={"samples": [unknown]}).status_code == 422
    assert client.get("/health").json()["feedback_samples"] == 3  # nothing malformed was stored


def test_retrain_promotes_and_hot_swaps():
    client.post("/v1/feedback", json={"samples": samples(BASE, 60)})
    report = client.post("/v1/retrain").json()
    assert report["status"] == "promoted" and report["version"] == "v3", report
    assert client.get("/health").json()["model_version"] == "v3"
    row = BASE.iloc[0]
    out = client.post("/v1/predict", json={"features": {f: float(row[f]) for f in FEATURES}}).json()
    assert out["model_version"] == "v3" and out["action"] in {"BUY", "SELL", "WAIT"}


def test_second_retrain_at_once_is_refused():
    with model_service.RETRAIN_LOCK:
        response = client.post("/v1/retrain")
    assert response.status_code == 409


def test_bar_time_must_be_a_timezone_aware_iso_time():
    for bad in ["garbage", "2026-11-20T10:00:00"]:  # unparseable, and without a timezone
        sample = {**samples(BASE, 1, offset=700)[0], "bar_time": bad}
        assert client.post("/v1/feedback", json={"samples": [sample]}).status_code == 422
    z = {**samples(BASE, 1, offset=701)[0], "bar_time": "2026-11-20T10:00:00Z"}
    plus = {**z, "bar_time": "2026-11-20T10:00:00+00:00"}  # the same instant spelled differently
    assert client.post("/v1/feedback", json={"samples": [z]}).json()["accepted"] == 1
    assert client.post("/v1/feedback", json={"samples": [plus]}).json() == {"accepted": 0, "duplicates": 1}


def test_a_failing_retrain_is_reported_and_does_not_stop_the_schedule():
    original = model_service.retrain
    model_service.retrain = lambda *a, **k: (_ for _ in ()).throw(OSError("training data missing"))
    try:
        model_service.scheduled_tick()  # must not raise: the schedule thread keeps running
        last = client.get("/health").json()["last_retrain"]
        assert last["status"] == "error" and "training data missing" in last["reason"]
    finally:
        model_service.retrain = original


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
