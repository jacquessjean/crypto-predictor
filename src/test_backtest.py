"""GET /api/backtest: model v2's Buy trades on November 2026 (data no choice had seen), for the Past signals tab.

Run from the repo root:  PYTHONPATH=src:. .venv/bin/python -m pytest src/test_backtest.py -q
"""
import math

import pandas as pd
from fastapi.testclient import TestClient

from app.api import main

NOVEMBER = pd.read_csv("out/results/horizon_november.csv").iloc[0]  # the recorded one-time check


def test_backtest_matches_the_recorded_november_check():
    body = TestClient(main.app).get("/api/backtest").json()
    s, trades = body["summary"], body["trades"]
    assert s["model_version"] == "v2" and s["trades"] == len(trades) == NOVEMBER.trades
    assert math.isclose(s["mean_net"], NOVEMBER.mean_net, abs_tol=1e-12)
    assert math.isclose(sum(t["net"] for t in trades) / len(trades), s["mean_net"], abs_tol=1e-12)
    assert math.isclose(s["random_mean_net"], NOVEMBER.random_mean_net, abs_tol=1e-12)
    assert math.isclose(s["hit_rate"], NOVEMBER.hit_rate)


def test_each_trade_is_one_hour_and_after_the_fee():
    trades = TestClient(main.app).get("/api/backtest").json()["trades"]
    for t in trades:
        hours = (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 3600
        assert hours == 1 and math.isclose(t["net"], t["exit"] / t["entry"] - 1 - 0.002, abs_tol=1e-12)
    assert [t["entry_time"] for t in trades] == sorted(t["entry_time"] for t in trades)  # oldest first
