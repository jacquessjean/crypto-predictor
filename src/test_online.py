"""Online learning: the feedback store, the retrain recipe and its safety gate, on a small synthetic dataset."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cleanup import FEATURES, NUMERIC_FEATURES, SYMBOLS
from final_model import FEE, FeedbackStore, fit_v2, retrain, save_artifact, three_class


def synthetic_base(path: Path, n: int = 3000) -> pd.DataFrame:
    """A features.parquet look-alike: train / val rows with a learnable link between ret_4 and the outcome."""
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", "2025-06-29", periods=n).floor("15min")
    df = pd.DataFrame(rng.normal(size=(n, len(NUMERIC_FEATURES))), columns=NUMERIC_FEATURES)
    sym = rng.choice(SYMBOLS, n)
    for s in SYMBOLS:
        df[f"symbol={s}"] = (sym == s).astype("int8")
    df["fwd_ret"] = 0.006 * df["ret_4"] + rng.normal(0, 0.006, n)
    df = df.assign(open_time=t, symbol=sym, close=100.0, label=three_class(df["fwd_ret"]).astype(int),
                   split=np.where(t < pd.Timestamp("2024-10-01"), "train", "val"))
    df[["open_time", "symbol", "close", "fwd_ret", "label", "split"] + FEATURES].to_parquet(path, index=False)
    return df


def samples(df: pd.DataFrame, k: int, offset: int = 0) -> list[dict]:
    rows = df.iloc[offset:offset + k]
    return [{"pair": r.symbol, "bar_time": f"2026-11-{1 + i // 96:02d}T{(i % 96) // 4:02d}:{15 * (i % 4):02d}:00+00:00",
             "features": {f: float(r[f]) for f in FEATURES},
             "realized_return": float(r.fwd_ret)} for i, (_, r) in enumerate(rows.iterrows(), start=offset)]


def setup(tmp: Path):
    base = synthetic_base(tmp / "features.parquet")
    root = tmp / "models"
    from final_model import load_splits

    splits = load_splits(tmp / "features.parquet")
    model, scaler, means = fit_v2(splits["train"], splits["val_a"])
    save_artifact(root / "v2", model, scaler, {"classes": [-1, 0, 1], "means": {str(k): v for k, v in means.items()},
                                               "fee": FEE, "buy_margin": 0.0005, "sell_margin": 0.0,
                                               "sell_enabled": False, "horizon_minutes": 60})
    return base, root, FeedbackStore(root / "feedback.db")


def test_feedback_store_dedupes():
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    batch = samples(base, 5)
    assert store.add(batch) == (5, 0)
    assert store.add(batch[:2] + samples(base, 1, offset=5)) == (1, 2)
    loaded = store.load()
    assert len(loaded) == 6 and set(FEATURES) <= set(loaded.columns) and store.count_since(None) == 6


def test_retrain_skips_below_min_new():
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    store.add(samples(base, 10))
    report = retrain(tmp / "features.parquet", store, root / "v2", root, min_new=50)
    assert report["status"] == "skipped" and report["new_samples"] == 10
    assert not (root / "v3").exists() and not (root / "current.json").exists()


def test_retrain_promotes_with_enough_feedback():
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    store.add(samples(base, 60))
    report = retrain(tmp / "features.parquet", store, root / "v2", root, min_new=50)
    assert report["status"] == "promoted" and report["version"] == "v3", report
    assert json.loads((root / "current.json").read_text()) == {"version": "v3"}
    trained_on = json.loads((root / "v3" / "trained_on.json").read_text())
    assert trained_on["feedback_rows"] == 60 and trained_on["gate"]["passed"]
    decision = json.loads((root / "v3" / "decision.json").read_text())
    assert decision["buy_margin"] == 0.0005 and decision["sell_enabled"] is False  # copied from the live model
    # the same feedback does not trigger another version
    again = retrain(tmp / "features.parquet", store, root / "v3", root, min_new=50)
    assert again["status"] == "skipped" and again["new_samples"] == 0


def test_retrain_rejected_by_gate_leaves_live_model():
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    store.add(samples(base, 60))
    report = retrain(tmp / "features.parquet", store, root / "v2", root, min_new=50, tolerance=-1.0)
    assert report["status"] == "rejected" and "log loss" in report["reason"]
    assert not (root / "v3").exists() and not (root / "current.json").exists()


def test_gate_also_holds_against_the_first_baseline():
    # each promotion may lose up to the tolerance against the live model; the fixed baseline stops a slow drift
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    (root / "v2" / "trained_on.json").write_text(json.dumps({"feedback_received_until": None, "baseline_log_loss": 0.1}))
    store.add(samples(base, 60))
    report = retrain(tmp / "features.parquet", store, root / "v2", root, min_new=50, tolerance=0.0005)
    assert report["status"] == "rejected" and "baseline" in report["reason"], report  # beats live, not the baseline


def test_promotion_leaves_no_partial_pointer_file():
    tmp = Path(tempfile.mkdtemp())
    base, root, store = setup(tmp)
    store.add(samples(base, 60))
    retrain(tmp / "features.parquet", store, root / "v2", root, min_new=50, tolerance=1.0)
    assert sorted(p.name for p in root.iterdir() if "current" in p.name) == ["current.json"]
    trained_on = json.loads((root / "v3" / "trained_on.json").read_text())
    assert "baseline_log_loss" in trained_on  # carried forward to the next retrain


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
