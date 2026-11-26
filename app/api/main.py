import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from .binance import (
    BinanceRestError,
    fetch_chart_candles,
    fetch_pair_klines,
    get_latest_closed_15m_bar,
)
from .config import BAR_SECONDS, HORIZON_SECONDS, SUPPORTED_PAIRS, settings
from .db import (
    get_prediction_by_bar,
    get_stats,
    init_db,
    list_predictions,
    insert_prediction,
)
from .features_live import LiveFeatureError, build_live_feature_row
from .model_client import ModelServiceError, RetrainRunning, get_model_status, predict_from_features, start_retrain
from .resolver import resolve_pending_predictions
from .schemas import (
    CandleItem,
    ModelStatusResponse,
    PredictionResponse,
    PredictRequest,
    StatsResponse,
)

scheduler = BackgroundScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.add_job(
        resolve_pending_predictions,
        trigger="interval",
        seconds=settings.resolver_interval_seconds,
        id="outcome_resolver",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Cryptocurrency Market Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


def _validate_pair(pair: str) -> str:
    normalized = pair.strip().upper()
    if normalized not in SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported pair '{pair}'. Supported pairs: {', '.join(SUPPORTED_PAIRS)}",
        )
    return normalized


@app.get("/api/pairs", response_model=list[str])
def get_pairs() -> list[str]:
    """Return the 10 supported USDT trading pairs."""
    return list(SUPPORTED_PAIRS)


@app.get("/api/candles", response_model=list[CandleItem])
def get_candles(
    pair: str = Query(..., description="Trading pair, e.g. ETHUSDT"),
    interval: str = Query("15m", pattern="^(1m|5m|15m|1h|4h)$"),
    limit: int = Query(96, ge=10, le=500),
) -> list[dict]:
    """Fetch recent candles for the frontend candlestick chart."""
    valid_pair = _validate_pair(pair)
    try:
        return fetch_chart_candles(valid_pair, interval=interval, limit=limit)
    except BinanceRestError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict", response_model=PredictionResponse)
def post_predict(body: PredictRequest) -> dict:
    """
    Validate pair, deduplicate by closed 15m bar, build live features,
    call MODEL_URL/v1/predict, store in SQLite, and return the result.
    """
    pair = _validate_pair(body.pair)
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    # The bar we score is the latest CLOSED 15m bar, known from the clock alone. Check the
    # database first so repeat requests within the same bar never touch Binance.
    bar_ms = BAR_SECONDS * 1000
    boundary_ms = now_ms // bar_ms * bar_ms          # open of the current (still forming) bar
    bar_time_iso = datetime.fromtimestamp(
        (boundary_ms - bar_ms) / 1000, tz=timezone.utc
    ).isoformat()
    cached = get_prediction_by_bar(pair, bar_time_iso)
    if cached is not None:
        return cached

    # Cache miss: fetch klines only up to the bar boundary, so every request in the same bar
    # asks Binance for the same window (and the forming bar is never included).
    try:
        klines_map = fetch_pair_klines(pair, end_time=boundary_ms)
        bar_time_iso, bar_close_ms, entry_price = get_latest_closed_15m_bar(
            klines_map[pair], pair, now_ms=now_ms
        )
    except (BinanceRestError, LiveFeatureError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Compute the 36-column feature vector using the shared training pipeline
    try:
        features = build_live_feature_row(
            pair=pair,
            pair_klines=klines_map[pair],
            btc_klines=klines_map.get("BTCUSDT"),
            now=now_ms,
        )
    except LiveFeatureError as exc:
        raise HTTPException(status_code=503, detail=exc.reason) from exc

    # Call the separate model container
    try:
        model_out = predict_from_features(pair=pair, bar_time=bar_time_iso, features=features)
    except ModelServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    expires_at_ms = bar_close_ms + (HORIZON_SECONDS * 1000)
    expires_at_iso = datetime.fromtimestamp(expires_at_ms / 1000, tz=timezone.utc).isoformat()

    record = {
        "pair": pair,
        "request_time": now.isoformat(),
        "bar_time": bar_time_iso,
        "bar_close_ms": bar_close_ms,
        "expires_at": expires_at_iso,
        "expires_at_ms": expires_at_ms,
        "p_up": model_out["p_up"],
        "p_flat": model_out["p_flat"],
        "p_down": model_out["p_down"],
        "action": model_out["action"],
        "entry_price": entry_price,
        "model_version": model_out["model_version"],
        "schema_version": model_out["schema_version"],
        # kept so resolved predictions can later be used as new training samples
        "features_json": json.dumps(features),
        "expected_net": model_out.get("expected_net"),
        "buy_margin": model_out.get("buy_margin"),
    }
    return insert_prediction(record)


@app.get("/api/predictions", response_model=list[PredictionResponse])
def get_predictions(
    pair: str | None = Query(None, description="Optional pair filter"),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict]:
    """Return prediction history with resolved outcomes."""
    valid_pair = _validate_pair(pair) if pair else None
    return list_predictions(pair=valid_pair, limit=limit)


@app.get("/api/stats", response_model=StatsResponse)
def get_prediction_stats() -> dict:
    """Return aggregate accuracy and net return after fees across resolved predictions."""
    return get_stats()


BACKTEST = Path(__file__).resolve().parent / "data" / "backtest.json"  # v2's November trades, computed from the saved artifact


@app.get("/api/backtest")
def get_backtest() -> dict:
    """Model v2's Buy trades on November 2026, data no choice had seen: one position per coin, sold after one hour."""
    return json.loads(BACKTEST.read_text())


@app.post("/api/model/retrain")
def post_model_retrain() -> dict:
    """The Retrain now button: starts a retrain in the background on the outcomes received so far. The model service's
    safety gate still decides whether the new version goes live. 409 while one is running.
    ponytail: no login, so anyone who can open the app can press it; one retrain at a time bounds the cost."""
    try:
        return {"started": start_retrain()}
    except RetrainRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/model", response_model=ModelStatusResponse)
def get_model() -> dict:
    """The live model version and its online learning: feedback received and the last retrain (read-only)."""
    try:
        return get_model_status()
    except ModelServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Serve the web client from the same origin so `uvicorn` alone gives a working app. This mount is
# registered last so it never shadows /api/*. Behind nginx (Docker) the same folder is served by nginx.
WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"  # the React build (npm run build), never its source
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")