"""Checks for the decision rule, the signal metrics and the exit simulation of the final model."""
import math

import numpy as np
import pandas as pd

from final_model import FEE, decide, embargo, horizon_return, random_baseline, three_class, tune_margin_trades, expected_net, recency_weights, signal_stats, simulate_exit, tune_margin

PROBA = np.array([[0.2, 0.3, 0.5]])
CLASSES = [1, -1, 0]  # deliberately not the sorted order: columns must be read through classes
MEANS = {1: 0.01, -1: 0.0, 0: -0.01}


def test_expected_net_uses_classes_order():
    assert math.isclose(expected_net(PROBA, CLASSES, MEANS, "buy")[0], 0.2 * 0.01 + 0.5 * -0.01 - FEE, abs_tol=1e-12)
    assert math.isclose(expected_net(PROBA, CLASSES, MEANS, "sell")[0], -(0.2 * 0.01 + 0.5 * -0.01) - FEE, abs_tol=1e-12)


def test_signal_stats_skips_nan_and_empty():
    score, fwd = np.array([1.0, 1.0, 0.0]), np.array([0.01, np.nan, 0.05])
    s = signal_stats(score, fwd, "buy", 0.5)
    assert s["n"] == 1 and math.isclose(s["mean_net"], 0.01 - FEE) and s["hit_rate"] == 1.0
    empty = signal_stats(score, fwd, "buy", 2.0)
    assert empty["n"] == 0 and math.isnan(empty["mean_net"])


def test_tune_margin_respects_min_share():
    score = np.arange(100, dtype=float)
    fwd = np.where(score >= 98, 0.05, -0.01)
    margin = tune_margin(score, fwd, "buy", min_share=0.1)
    assert (score > margin).sum() >= 10


def test_simulate_exit_no_overlap_and_early_exit():
    df = pd.DataFrame({"symbol": "BTCUSDT", "open_time": pd.date_range("2025-03-01", periods=8, freq="15min"),
                       "close": [100.0, 101, 102, 103, 104, 105, 106, 107]})
    buy = np.array([1.0, 1, 0, 0, 0, 0, 0, 0])
    sell = np.array([0.0, 0, 1, 0, 0, 0, 0, 0])
    hold = simulate_exit(df, buy, sell, 0.5, 0.5, early_exit=False)
    assert hold["trades"] == 1 and math.isclose(hold["mean_net"], 104 / 100 - 1 - FEE)
    early = simulate_exit(df, buy, sell, 0.5, 0.5, early_exit=True)
    assert early["trades"] == 1 and math.isclose(early["mean_net"], 102 / 100 - 1 - FEE)


def test_simulate_exit_uses_time_when_a_bar_is_missing():
    # 00:15 is missing: the Sell at 00:45 is the second row after entry but 30 minutes later
    df = pd.DataFrame({"symbol": "BTCUSDT", "close": [100.0, 110, 90, 95],
                       "open_time": pd.to_datetime(["2025-03-01 00:00", "2025-03-01 00:30", "2025-03-01 00:45", "2025-03-01 01:00"])})
    buy, sell = np.array([1.0, 0, 0, 0]), np.array([0.0, 0, 1, 0])
    early = simulate_exit(df, buy, sell, 0.5, 0.5, early_exit=True)
    assert early["trades"] == 1 and math.isclose(early["mean_net"], 90 / 100 - 1 - FEE)
    hold = simulate_exit(df, buy, sell, 0.5, 0.5, early_exit=False)
    assert hold["trades"] == 1 and math.isclose(hold["mean_net"], 95 / 100 - 1 - FEE)


def test_random_baseline():
    # acting at every bar: buy earns the average move minus the fee, sell avoids it minus the fee; NaN rows skipped
    fwd = np.array([0.01, -0.03, np.nan])
    assert math.isclose(random_baseline(fwd, "buy"), -0.01 - FEE)
    assert math.isclose(random_baseline(fwd, "sell"), 0.01 - FEE)


def test_horizon_return_needs_exact_bar():
    df = pd.DataFrame({"symbol": "ETHUSDT", "close": [100.0, 110, 121],
                       "open_time": pd.to_datetime(["2025-03-01 00:00", "2025-03-01 00:15", "2025-03-01 00:45"])})
    r = horizon_return(df, 2)
    assert math.isnan(r.iloc[0]) and math.isclose(r.iloc[1], 121 / 110 - 1) and math.isnan(r.iloc[2])
    assert list(np.nan_to_num(three_class(np.array([0.01, 0.0, -0.01, np.nan])), nan=9)) == [1, -1, 0, 9]


def test_embargo_drops_rows_whose_label_crosses():
    t = pd.to_datetime(["2025-02-28 23:45", "2025-03-01 00:00", "2025-03-01 00:15", "2025-03-01 00:30", "2025-03-01 00:45"])
    kept = embargo(pd.DataFrame({"open_time": t}), pd.Timestamp("2025-03-01 01:00"), 4)
    assert list(kept["open_time"]) == [t[0]]


def test_simulate_exit_horizon():
    df = pd.DataFrame({"symbol": "BTCUSDT", "open_time": pd.date_range("2025-03-01", periods=8, freq="15min"),
                       "close": [100.0, 101, 102, 103, 104, 105, 106, 107]})
    buy = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    out = simulate_exit(df, buy, np.zeros(8), 0.5, 0.5, early_exit=False, horizon_bars=2)
    assert out["trades"] == 1 and math.isclose(out["mean_net"], 102 / 100 - 1 - FEE)
    nets = simulate_exit(df, buy, np.zeros(8), 0.5, 0.5, early_exit=False, horizon_bars=2, with_trades=True)["nets"]
    assert len(nets) == 1 and math.isclose(nets[0], 102 / 100 - 1 - FEE)


def test_simulate_exit_lists_each_trade():
    df = pd.DataFrame({"symbol": "BTCUSDT", "open_time": pd.date_range("2025-03-01", periods=8, freq="15min"),
                       "close": [100.0, 101, 102, 103, 104, 105, 106, 107]})
    buy = np.array([1.0, 0, 1, 0, 0, 0, 0, 0])  # the second Buy fires at the first trade's exit bar
    out = simulate_exit(df, buy, np.zeros(8), 0.5, 0.5, early_exit=False, horizon_bars=2, with_trades=True)
    assert [(t["symbol"], str(t["entry_time"]), str(t["exit_time"]), t["entry"], t["exit"]) for t in out["details"]] == [
        ("BTCUSDT", "2025-03-01 00:00:00", "2025-03-01 00:30:00", 100.0, 102.0),
        ("BTCUSDT", "2025-03-01 00:30:00", "2025-03-01 01:00:00", 102.0, 104.0)]
    assert np.allclose([t["net"] for t in out["details"]], out["nets"])


def test_tune_margin_trades_respects_min_trades():
    df = pd.DataFrame({"symbol": "BTCUSDT", "open_time": pd.date_range("2025-03-01", periods=400, freq="15min"),
                       "close": 100 + np.sin(np.arange(400) / 5)})
    score = np.cos(np.arange(400) / 5)
    margin = tune_margin_trades(df, score, 4, min_trades=30)
    assert simulate_exit(df, score, np.zeros(400), margin, 1.0, early_exit=False, horizon_bars=4)["trades"] >= 30


def test_recency_weights():
    t = pd.Series(pd.to_datetime(["2023-01-01", "2024-01-01"]))
    w = recency_weights(t, half_life_days=365)
    assert math.isclose(w[1] / w[0], 2.0, rel_tol=1e-9)


def test_decide():
    loose = {"classes": CLASSES, "means": MEANS, "fee": FEE, "buy_margin": 0.0, "sell_margin": 0.0}
    assert decide(PROBA[0], CLASSES, loose) == "SELL"
    assert decide(PROBA[0], CLASSES, {**loose, "buy_margin": 1.0, "sell_margin": 1.0}) == "WAIT"
    assert decide(PROBA[0], CLASSES, {**loose, "sell_enabled": False}) == "WAIT"  # Sell switched off: never emitted


def test_artifact_roundtrip():
    from final_model import MODEL_DIR, apply_scaler, load_artifact, load_splits

    for folder in [MODEL_DIR, MODEL_DIR.parent / "v2"]:
        if not (folder / "model.pkl").exists():
            print("skipped: no artifact in", folder)
            continue
        model, scaler, schema, decision = load_artifact(folder)
        proba = model.predict_proba(apply_scaler(load_splits()["val_b"].iloc[:100], scaler))
        assert np.allclose(proba, np.load(folder / "check.npz")["proba"], atol=1e-12), folder
        assert schema["schema_version"] == "1" and set(decision) >= {"classes", "means", "fee", "buy_margin", "sell_margin"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
