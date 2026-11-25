"""Checks the real model service against the contract app/api/model_client.py expects (skips without an artifact)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from final_model import FEATURES, MODEL_DIR, apply_scaler, load_artifact, load_splits

os.environ.setdefault("MODEL_DIR", str(MODEL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))


def _client():
    if not (MODEL_DIR / "model.pkl").exists():
        print("skipped: no artifact in", MODEL_DIR)
        return None
    from fastapi.testclient import TestClient
    import model_service

    return TestClient(model_service.app)


def _row():
    return load_splits()["val_b"].iloc[0]


def test_predict_matches_artifact():
    client = _client()
    if client is None:
        return
    row = _row()
    body = client.post("/v1/predict", json={"pair": row["symbol"], "bar_time": str(row["open_time"]),
                                            "features": {f: float(row[f]) for f in FEATURES}})
    assert body.status_code == 200, body.text
    out = body.json()
    assert out["action"] in {"BUY", "SELL", "WAIT"}
    assert math.isclose(out["p_up"] + out["p_flat"] + out["p_down"], 1.0, abs_tol=1e-9)
    model, scaler, _, _ = load_artifact(MODEL_DIR)
    p = dict(zip(model.classes_, model.predict_proba(apply_scaler(pd.DataFrame([row]), scaler))[0]))
    assert np.allclose([out["p_up"], out["p_flat"], out["p_down"]], [p[1], p[-1], p[0]], atol=1e-12)


def test_missing_and_nonfinite_features_are_rejected():
    client = _client()
    if client is None:
        return
    features = {f: float(_row()[f]) for f in FEATURES}
    missing = {k: v for k, v in features.items() if k != "ret_4"}
    assert client.post("/v1/predict", json={"features": missing}).status_code == 422
    assert client.post("/v1/predict", json={"features": {**features, "ret_4": "nan"}}).status_code == 422


def test_health_reports_versions():
    client = _client()
    if client is None:
        return
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["schema_version"] == "1" and body["model_version"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
