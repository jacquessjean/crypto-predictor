import hashlib
import math
from typing import Any

from fastapi import FastAPI, HTTPException

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
           "DOGEUSDT", "SOLUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
NUMERIC = ["returns", "volatility", "upper_wick", "lower_wick", "body_ratio",
           "taker_buy_ratio", "taker_buy_ratio_16", "relative_volume",
           "normalized_volatility", "rel_trade_size", "ret_4", "ret_16",
           "ret_96", "ret_672", "vol_96", "rv_15", "rv_60", "rv_ratio",
           "idle_minutes", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
           "btc_ret_4", "btc_ret_16", "rel_ret_4"]
FEATURES = [f"symbol={symbol}" for symbol in SYMBOLS] + NUMERIC

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": "v1", "schema_version": "1"}


@app.post("/v1/predict")
def predict(payload: dict[str, Any]) -> dict[str, float | str]:
    features = payload.get("features")
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="features must be an object")
    missing = [name for name in FEATURES if name not in features]
    invalid = [name for name, value in features.items()
               if name in FEATURES and (isinstance(value, bool) or not isinstance(value, (int, float))
                                        or not math.isfinite(value))]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {', '.join(missing)}")
    if invalid:
        raise HTTPException(status_code=422, detail=f"invalid features: {', '.join(invalid)}")

    seed = hashlib.sha256(str([features[name] for name in FEATURES]).encode()).digest()[0]
    action, probabilities = [
        ("WAIT", (0.38, 0.30, 0.32)),
        ("BUY", (0.55, 0.20, 0.25)),
        ("SELL", (0.20, 0.25, 0.55)),
    ][seed % 3]
    p_up, p_flat, p_down = probabilities
    return {"p_up": p_up, "p_flat": p_flat, "p_down": p_down,
            "action": action, "model_version": "v1"}
