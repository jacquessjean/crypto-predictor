"""The fresh-data path must give the same features as the training build (checked on October 2026, where both exist)."""
import numpy as np
import pandas as pd

from cleanup import FEATURES
from final_model import horizon_return
from fresh_data import PROJECT_ROOT, _read, build_features, cached_fresh

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _monthly(sym):
    return _read(PROJECT_ROOT / "out" / "raw" / sym / f"{sym}-1m-2026-10.zip")


def test_build_features_matches_training():
    fresh = build_features({s: _monthly(s) for s in COINS})
    fresh = fresh[fresh["open_time"] >= "2026-10-20"]
    train = pd.read_parquet(PROJECT_ROOT / "out" / "processed" / "features.parquet",
                            columns=["open_time", "symbol"] + FEATURES)
    merged = fresh.merge(train, on=["open_time", "symbol"], suffixes=("_fresh", "_train"))
    assert len(merged) > 1000
    for f in FEATURES:
        assert np.allclose(merged[f + "_fresh"], merged[f + "_train"], rtol=1e-10, atol=1e-12), f


def test_labels_past_the_last_bar_are_missing():
    # November ends at the last downloaded bar: the final H bars have no outcome yet and must drop out
    df = pd.DataFrame({"symbol": "BTCUSDT", "close": np.arange(10, dtype=float) + 100,
                       "open_time": pd.date_range("2026-11-24 21:30", periods=10, freq="15min")})
    r = horizon_return(df, 4)
    assert r.iloc[:6].notna().all() and r.iloc[6:].isna().all()


def test_cached_fresh_builds_once(tmp_path=None):
    import tempfile
    from pathlib import Path

    path = Path(tmp_path or tempfile.mkdtemp()) / "fresh.parquet"
    calls = []
    build = lambda: calls.append(1) or pd.DataFrame({"a": [1, 2]})
    first, second = cached_fresh(path, build), cached_fresh(path, build)
    assert len(calls) == 1 and first.equals(second) and path.exists()


if __name__ == "__main__":
    test_build_features_matches_training()
    test_labels_past_the_last_bar_are_missing()
    test_cached_fresh_builds_once()
    print("ok")
