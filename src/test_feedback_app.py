"""The application sends every resolved prediction back to the model service as a labeled sample, with retry.

Run from the repo root:  PYTHONPATH=src:. .venv/bin/python -m pytest src/test_feedback_app.py -q
"""
import dataclasses
import json
import sqlite3

import pytest

from app.api import db, resolver
from app.api.model_client import ModelServiceError
from cleanup import FEATURES

FEATURES_JSON = json.dumps({name: 0.5 for name in FEATURES})


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, db_path=str(tmp_path / "t.db")))
    db.init_db()


def _prediction(bar_time, action="BUY", resolved=True, features=FEATURES_JSON):
    row = db.insert_prediction({
        "pair": "ETHUSDT", "request_time": "2026-11-18T10:07:30+00:00", "bar_time": bar_time, "bar_close_ms": 0,
        "expires_at": "2026-11-18T11:00:00+00:00", "expires_at_ms": 0, "p_up": 0.5, "p_flat": 0.25, "p_down": 0.25,
        "action": action, "entry_price": 100.0, "model_version": "v2", "schema_version": "1", "features_json": features})
    if resolved:
        db.update_prediction_outcome(row["id"], exit_price=101.0, realized_return=0.01, net_return=0.008,
                                     status="correct" if action == "BUY" else "neutral")
    return row


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(resolver, "send_feedback", lambda samples: sent.append(samples) or len(samples))
    return sent


def test_resolved_prediction_is_sent_once_and_marked(database, monkeypatch):
    _prediction("2026-11-18T09:45:00+00:00")
    sent = _capture(monkeypatch)
    assert resolver.send_pending_feedback() == 1
    sample = sent[0][0]
    assert sample == {"pair": "ETHUSDT", "bar_time": "2026-11-18T09:45:00+00:00",
                      "features": json.loads(FEATURES_JSON), "realized_return": 0.01}
    assert resolver.send_pending_feedback() == 0 and len(sent) == 1  # never sent twice


def test_model_service_down_leaves_rows_for_the_next_run(database, monkeypatch):
    _prediction("2026-11-18T09:45:00+00:00")

    def down(samples):
        raise ModelServiceError("Model service is unavailable")

    monkeypatch.setattr(resolver, "send_feedback", down)
    assert resolver.send_pending_feedback() == 0
    sent = _capture(monkeypatch)
    assert resolver.send_pending_feedback() == 1 and len(sent) == 1


def test_wait_predictions_are_sent_too(database, monkeypatch):
    _prediction("2026-11-18T09:45:00+00:00", action="WAIT")  # the outcome is ground truth whatever the advice was
    sent = _capture(monkeypatch)
    assert resolver.send_pending_feedback() == 1 and sent[0][0]["realized_return"] == 0.01


def test_pending_or_featureless_predictions_are_not_sent(database, monkeypatch):
    _prediction("2026-11-18T09:45:00+00:00", resolved=False)
    _prediction("2026-11-18T10:00:00+00:00", features=None)
    sent = _capture(monkeypatch)
    assert resolver.send_pending_feedback() == 0 and sent == []


def test_init_db_adds_feedback_column_to_old_database(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, pair TEXT, bar_time TEXT, action TEXT, status TEXT, "
                 "expires_at_ms INTEGER, features_json TEXT, UNIQUE(pair, bar_time))")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, db_path=str(path)))
    db.init_db()
    columns = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(predictions)")}
    assert "feedback_sent" in columns
