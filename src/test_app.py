"""API tests: cache-before-fetch, stored features, DB migration, WAIT handling.

Run from the repo root:  PYTHONPATH=src:. python -m pytest src/test_app_api.py -q
No network is used: Binance, feature building and the model service are faked.
"""
import dataclasses
import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import db, main, resolver
from cleanup import FEATURES

NOW = datetime(2026, 11, 18, 10, 7, 30, tzinfo=timezone.utc)
BAR_OPEN = "2026-11-18T09:45:00+00:00"          # latest CLOSED 15m bar when it is 10:07
FAKE_FEATURES = {name: 0.5 for name in FEATURES}


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def _minute_klines():
    """Closed 1m klines covering the 09:45-10:00 bar (all that main needs for the entry price)."""
    start = int(datetime(2026, 11, 18, 9, 45, tzinfo=timezone.utc).timestamp() * 1000)
    return [{"open_time": start + i * 60_000, "close": 100.0 + i} for i in range(15)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, db_path=str(tmp_path / "t.db")))
    db.init_db()
    monkeypatch.setattr(main, "datetime", _FrozenDatetime)

    state = {"fetches": 0, "action": "BUY"}

    def fake_fetch(pair, margin=None, end_time=None):
        state["fetches"] += 1
        state["end_time"] = end_time
        return {pair: _minute_klines(), "BTCUSDT": _minute_klines()}

    monkeypatch.setattr(main, "fetch_pair_klines", fake_fetch)
    monkeypatch.setattr(main, "build_live_feature_row", lambda **kwargs: dict(FAKE_FEATURES))
    monkeypatch.setattr(
        main,
        "predict_from_features",
        lambda **kwargs: {"p_up": 0.5, "p_flat": 0.25, "p_down": 0.25, "action": state["action"],
                          "model_version": "v1", "schema_version": "1"},
    )
    return TestClient(main.app), state


# ---- fix 3: repeat requests must not fetch from Binance -------------------------------------

def test_repeat_request_in_same_bar_skips_binance(client):
    c, state = client
    first = c.post("/api/predict", json={"pair": "ETHUSDT"})
    assert first.status_code == 200
    assert first.json()["bar_time"] == BAR_OPEN
    assert state["fetches"] == 1

    second = c.post("/api/predict", json={"pair": "ETHUSDT"})
    assert second.json()["id"] == first.json()["id"]
    assert state["fetches"] == 1, "cached bar must be answered from SQLite without calling Binance"


def test_fetch_window_ends_at_bar_boundary(client):
    c, state = client
    c.post("/api/predict", json={"pair": "ETHUSDT"})
    boundary_ms = int(datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert state["end_time"] == boundary_ms, "the still-forming 15m bar must not be requested"


# ---- fix 4: feature vector is stored, old databases are migrated -----------------------------

def test_features_are_stored_per_prediction(client, tmp_path):
    c, _ = client
    c.post("/api/predict", json={"pair": "ETHUSDT"})
    row = db.get_prediction_by_bar("ETHUSDT", BAR_OPEN)
    assert json.loads(row["features_json"]) == FAKE_FEATURES
    assert "features_json" not in c.get("/api/predictions").json()[0], "large payload stays out of the API"


def test_init_db_migrates_old_database(tmp_path, monkeypatch):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)  # table as created before features_json existed
    conn.execute(
        """CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL, request_time TEXT NOT NULL,
            bar_time TEXT NOT NULL, bar_close_ms INTEGER NOT NULL, expires_at TEXT NOT NULL,
            expires_at_ms INTEGER NOT NULL, p_up REAL NOT NULL, p_flat REAL NOT NULL, p_down REAL NOT NULL,
            action TEXT NOT NULL, entry_price REAL NOT NULL, model_version TEXT NOT NULL,
            schema_version TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', exit_price REAL,
            realized_return REAL, net_return REAL, resolved_at TEXT, UNIQUE(pair, bar_time))"""
    )
    conn.execute(
        "INSERT INTO predictions (pair, request_time, bar_time, bar_close_ms, expires_at, expires_at_ms,"
        " p_up, p_flat, p_down, action, entry_price, model_version, schema_version, status)"
        " VALUES ('ETHUSDT','t','b',0,'e',0,.4,.3,.3,'WAIT',1.0,'v1','1','incorrect')"
    )
    conn.commit()
    conn.close()

    db.init_db(path)
    db.init_db(path)  # idempotent

    with db.get_conn(path) as conn:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
        status = conn.execute("SELECT status FROM predictions").fetchone()["status"]
    assert "features_json" in columns
    assert status == "neutral", "old WAIT rows marked 'incorrect' are corrected"


# ---- fix 6: WAIT is neutral and has no win rate ----------------------------------------------

@pytest.mark.parametrize("exit_price", [100.0, 101.0, 99.0])   # flat, +1%, -1%
def test_wait_is_always_neutral(exit_price):
    realized, net, status = resolver.evaluate_outcome("WAIT", 100.0, exit_price)
    assert status == "neutral"
    assert net == 0.0


def test_buy_and_sell_statuses_unchanged():
    assert resolver.evaluate_outcome("BUY", 100.0, 101.0)[2] == "correct"
    assert resolver.evaluate_outcome("BUY", 100.0, 99.0)[2] == "incorrect"
    assert resolver.evaluate_outcome("BUY", 100.0, 100.1)[2] == "flat"
    assert resolver.evaluate_outcome("SELL", 100.0, 99.0)[2] == "correct"
    assert resolver.evaluate_outcome("SELL", 100.0, 101.0)[2] == "incorrect"


def test_stats_have_no_win_rate_for_wait(client, monkeypatch):
    c, state = client
    state["action"] = "WAIT"
    c.post("/api/predict", json={"pair": "ETHUSDT"})
    monkeypatch.setattr(resolver, "fetch_close_at", lambda pair, ms: 110.0)   # big move after a WAIT
    monkeypatch.setattr(resolver, "datetime", _FrozenDatetime)
    monkeypatch.setattr(_FrozenDatetime, "now", classmethod(lambda cls, tz=None: datetime(2026, 11, 18, 12, tzinfo=timezone.utc)))
    assert resolver.resolve_pending_predictions() == 1

    history = c.get("/api/predictions").json()
    assert history[0]["status"] == "neutral"
    wait = next(a for a in c.get("/api/stats").json()["by_action"] if a["action"] == "WAIT")
    assert wait["resolved"] == 1
    assert wait["win_rate"] is None


# ---- web client is served from the same origin without shadowing the API ---------------------

def test_web_client_is_served_and_api_still_works(client):
    c, _ = client
    page = c.get("/")
    assert page.status_code == 200 and "Crypto Predictor" in page.text
    assert c.get("/app.js").status_code == 200
    assert c.get("/style.css").status_code == 200
    assert c.get("/api/pairs").json()[0] == "BTCUSDT"      # an API route, not the static mount
    assert c.get("/api/does-not-exist").status_code == 404