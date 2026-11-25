"""Model service: 36 features in, the three class probabilities and Buy / Sell / Wait out (replaces stub.py)."""
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from final_model import apply_scaler, decide, load_artifact

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/v1"))  # mounted read-only in the model container
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
model, scaler, schema, decision = load_artifact(MODEL_DIR)
FEATURES = schema["features"]
CLASSES = list(model.classes_)

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION, "schema_version": schema["schema_version"]}


@app.post("/v1/predict")
def predict(payload: dict[str, Any]) -> dict[str, float | str]:
    features = payload.get("features")
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="features must be an object")
    missing = [name for name in FEATURES if name not in features]
    invalid = [name for name in FEATURES if name in features and (
        isinstance(features[name], bool) or not isinstance(features[name], (int, float)) or not math.isfinite(features[name]))]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {', '.join(missing)}")
    if invalid:
        raise HTTPException(status_code=422, detail=f"invalid features: {', '.join(invalid)}")

    proba = model.predict_proba(apply_scaler(pd.DataFrame([features], columns=FEATURES), scaler))[0]
    p = dict(zip(CLASSES, (float(v) for v in proba)))
    return {"p_up": p[1], "p_flat": p[-1], "p_down": p[0], "action": decide(proba, CLASSES, decision),
            "model_version": MODEL_VERSION, "schema_version": schema["schema_version"]}
