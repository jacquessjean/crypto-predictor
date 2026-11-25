"""Model service: 36 features in, the three class probabilities and Buy / Sell / Wait out.

Online learning: the application sends each resolved prediction back (POST /v1/feedback); the service retrains on a
schedule (and on POST /v1/retrain), and a new version replaces the live one only if it passes the safety gate.
"""
import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from cleanup import SYMBOLS
from final_model import FeedbackStore, apply_scaler, decide, load_artifact, retrain

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/v2"))  # used when no retrain has been promoted yet
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", MODEL_DIR.parent))  # where versions and current.json live
BASE_DATA = Path(os.environ.get("BASE_DATA", "/data/features.parquet"))
FEEDBACK_DB = Path(os.environ.get("FEEDBACK_DB", MODELS_ROOT / "feedback.db"))
RETRAIN_INTERVAL_HOURS = float(os.environ.get("RETRAIN_INTERVAL_HOURS", "24"))  # 0 disables the schedule
MIN_NEW_FEEDBACK = int(os.environ.get("MIN_NEW_FEEDBACK", "50"))
GATE_TOLERANCE = float(os.environ.get("GATE_TOLERANCE", "0.0005"))  # allowed val-B log-loss increase

logger = logging.getLogger("model_service")
RETRAIN_LOCK = threading.Lock()  # one retrain at a time
_LIVE_LOCK = threading.Lock()  # guards the swap of the live model
_store = None
live: dict = {}
last_retrain: dict = {}


def store() -> FeedbackStore:
    global _store
    if _store is None:  # created on first use, so a predict-only service never writes files
        _store = FeedbackStore(FEEDBACK_DB)
    return _store


def load_live() -> None:
    """Loads the promoted version (current.json), or MODEL_DIR before any promotion, and swaps it in."""
    pointer = MODELS_ROOT / "current.json"
    folder = MODELS_ROOT / json.loads(pointer.read_text())["version"] if pointer.exists() else MODEL_DIR
    model, scaler, schema, decision = load_artifact(folder)
    with _LIVE_LOCK:
        live.update(dir=folder, version=folder.name, model=model, scaler=scaler, schema=schema, decision=decision,
                    classes=list(model.classes_), horizon=int(decision.get("horizon_minutes", 60)))


load_live()
# kept for callers and tests that read them as module attributes
FEATURES = live["schema"]["features"]
HORIZON_MINUTES = live["horizon"]
decision = live["decision"]


def invalid_features(features: Any) -> list[str] | None:
    """None when features is a complete dict of finite numbers, otherwise the bad or missing names."""
    if not isinstance(features, dict):
        return ["features"]
    return [name for name in FEATURES if name not in features or isinstance(features[name], bool)
            or not isinstance(features[name], (int, float)) or not math.isfinite(features[name])] or None


def run_retrain() -> dict:
    """One online-learning step; refuses to start while another is running."""
    if not RETRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a retrain is already running")
    try:
        report = retrain(BASE_DATA, store(), live["dir"], MODELS_ROOT, min_new=MIN_NEW_FEEDBACK, tolerance=GATE_TOLERANCE)
        if report["status"] == "promoted":
            load_live()
    except Exception as exc:  # recorded so /health shows it, then raised for the caller
        report = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        last_retrain.clear()
        last_retrain.update({**report, "at": pd.Timestamp.now(tz="UTC").isoformat()})
        RETRAIN_LOCK.release()
    return report


def scheduled_tick() -> None:
    """One scheduled retrain. Never raises, so a failure (missing data, a full disk) cannot stop the schedule."""
    try:
        run_retrain()
    except HTTPException:
        pass  # a manual retrain is running: skip this tick
    except Exception:
        logger.exception("scheduled retrain failed (reported in /health; the next tick will try again)")


def _schedule(stop: threading.Event) -> None:
    while not stop.wait(RETRAIN_INTERVAL_HOURS * 3600):
        scheduled_tick()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = threading.Event()
    if RETRAIN_INTERVAL_HOURS > 0:
        threading.Thread(target=_schedule, args=(stop,), daemon=True, name="retrain-schedule").start()
    yield
    stop.set()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_version": live["version"], "schema_version": live["schema"]["schema_version"],
            "horizon_minutes": live["horizon"],
            "feedback_samples": store().count_since(None) if _store or FEEDBACK_DB.exists() else 0,  # never creates the file
            "last_retrain": last_retrain or None}


@app.post("/v1/predict")
def predict(payload: dict[str, Any]) -> dict[str, float | str | int]:
    features = payload.get("features")
    bad = invalid_features(features)
    if bad:
        missing = [n for n in bad if not isinstance(features, dict) or n not in features]
        raise HTTPException(status_code=422, detail=f"missing features: {', '.join(missing)}" if missing
                            else f"invalid features: {', '.join(bad)}")
    current = dict(live)  # one consistent snapshot even if a retrain swaps the model meanwhile
    proba = current["model"].predict_proba(apply_scaler(pd.DataFrame([features], columns=FEATURES), current["scaler"]))[0]
    p = dict(zip(current["classes"], (float(v) for v in proba)))
    return {"p_up": p[1], "p_flat": p[-1], "p_down": p[0], "action": decide(proba, current["classes"], current["decision"]),
            "model_version": current["version"], "schema_version": current["schema"]["schema_version"],
            "horizon_minutes": current["horizon"]}


@app.post("/v1/feedback")
def feedback(payload: dict[str, Any]) -> dict[str, int]:
    """Resolved predictions sent back as ground truth: pair, bar_time, the 36 features and the realized return."""
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise HTTPException(status_code=422, detail="samples must be a non-empty list")
    clean = []
    for i, s in enumerate(samples):
        ret = s.get("realized_return") if isinstance(s, dict) else None
        when = utc_bar_time(s.get("bar_time")) if isinstance(s, dict) else None
        if (not isinstance(s, dict) or s.get("pair") not in SYMBOLS or when is None
                or invalid_features(s.get("features")) or isinstance(ret, bool) or not isinstance(ret, (int, float))
                or not math.isfinite(ret)):
            raise HTTPException(status_code=422, detail=f"sample {i} is malformed; nothing was stored")
        clean.append({**s, "bar_time": when})
    accepted, duplicates = store().add(clean)
    return {"accepted": accepted, "duplicates": duplicates}


def utc_bar_time(value: Any) -> str | None:
    """A timezone-aware ISO time in one canonical UTC spelling (so 'Z' and '+00:00' dedupe), or None."""
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return when.astimezone(timezone.utc).isoformat() if when.tzinfo else None


@app.post("/v1/retrain")
def retrain_now() -> dict[str, Any]:
    return run_retrain()
