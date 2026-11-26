"""Replay: the predictions the app would have made over the past hours, resolved from the real prices that followed.

Run from the repo root:  PYTHONPATH=src:. .venv/bin/python -m pytest src/test_replay.py -q
Binance and the model service are faked; the feature code is the real live code.
"""
import dataclasses
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import db, main, replay, resolver
from app.api.features_live import build_live_feature_row

MIN = 60_000
BAR = 15 * MIN
NOW = 1_795_000_000_000 // BAR * BAR + 7 * MIN  # 7 minutes into a bar, like a real request


def synthetic_klines(seed: int, days: float = 7.5) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60)
    start = NOW // MIN * MIN - n * MIN
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return [{"open_time": start + i * MIN, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
             "volume": 10.0, "quote_asset_volume": 10.0 * c, "number_of_trades": 5,
             "taker_buy_base_asset_volume": 5.0, "taker_buy_quote_asset_volume": 5.0 * c}
            for i, c in enumerate(close)]


KLINES = {"ETHUSDT": synthetic_klines(1), "BTCUSDT": synthetic_klines(2)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, db_path=str(tmp_path / "t.db")))
    db.init_db()

    def fetch(pair, margin, end_time):
        assert end_time <= NOW  # never asks for the future
        return {s: [k for k in KLINES[s] if k["open_time"] < end_time] for s in (pair, "BTCUSDT")}

    monkeypatch.setattr(replay, "fetch_pair_klines", fetch)
    calls = []

    def model(pair, bar_time, features):
        calls.append(bar_time)
        return {"p_up": 0.5, "p_flat": 0.25, "p_down": 0.25, "action": "BUY" if len(calls) % 2 else "WAIT",
                "model_version": "v2", "schema_version": "1"}

    monkeypatch.setattr(replay, "predict_from_features", model)
    sent = []
    monkeypatch.setattr(resolver, "send_feedback", lambda samples: sent.extend(samples) or len(samples))
    return sent


def test_replay_stores_resolved_predictions_built_only_from_the_past(setup):
    out = replay.replay(hours=3, pairs=["ETHUSDT"], now_ms=NOW)
    rows = sorted(db.list_predictions(pair="ETHUSDT", limit=100), key=lambda r: r["bar_time"])
    # bars whose hour is over and that closed in the last 3 hours: closes in (NOW - 3 h, NOW - 1 h]
    assert out["replayed"] == len(rows) == 8 and all(r["source"] == "replay" for r in rows)
    closes = {k["open_time"]: k["close"] for k in KLINES["ETHUSDT"]}
    for r in rows:
        assert r["bar_close_ms"] + 3_600_000 == r["expires_at_ms"] <= NOW and r["status"] != "pending"
        assert r["exit_price"] == closes[r["expires_at_ms"] - MIN]  # the last 1-minute close before expiry
        assert r["entry_price"] == closes[r["bar_close_ms"] - MIN]
        cut = {s: [k for k in v if k["open_time"] < r["bar_close_ms"]] for s, v in KLINES.items()}
        live = build_live_feature_row("ETHUSDT", cut["ETHUSDT"], cut["BTCUSDT"], now=r["bar_close_ms"] + 1000)
        assert json.loads(r["features_json"]) == live  # exactly what a live request at that time would have built
    assert len(setup) == 8  # every outcome went back to the model service


def test_replay_skips_bars_already_predicted_and_keeps_them(setup):
    replay.replay(hours=2, pairs=["ETHUSDT"], now_ms=NOW)
    first = {r["bar_time"]: r for r in db.list_predictions(pair="ETHUSDT", limit=100)}
    again = replay.replay(hours=3, pairs=["ETHUSDT"], now_ms=NOW)
    rows = {r["bar_time"]: r for r in db.list_predictions(pair="ETHUSDT", limit=100)}
    assert again["skipped"] == len(first) == 4 and again["replayed"] == 4 and len(rows) == 8
    assert all(rows[t] == first[t] for t in first)


def test_history_says_which_rows_were_replayed(setup):
    replay.replay(hours=2, pairs=["ETHUSDT"], now_ms=NOW)
    rows = TestClient(main.app).get("/api/predictions").json()
    assert rows and all(r["source"] == "replay" for r in rows)
