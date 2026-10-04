"""Leakage check: features at time t must not change when data after t is removed."""
import numpy as np
import pandas as pd

from cleanup import FEATURES, HORIZON, add_features, add_market_features, clean_symbol


def test_no_lookahead():
    rng = np.random.default_rng(0)
    n = 2000 * 15  # minutes
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    v = rng.uniform(0, 10, n)
    v[rng.random(n) < 0.05] = 0  # filler candles
    d = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=n, freq="1min"),
                      "open": c, "high": c * 1.002, "low": c * 0.998, "close": c, "volume": v,
                      "quote_asset_volume": v * c, "number_of_trades": (v * 10).astype(int),
                      "taker_buy_base_asset_volume": v / 2, "taker_buy_quote_asset_volume": v * c / 2, "symbol": "BTCUSDT"})
    feats = lambda x: add_market_features(add_features(clean_symbol(x)[0]).assign(symbol="BTCUSDT")
                                          .rename_axis("open_time").reset_index())
    full, cut = feats(d), feats(d.iloc[:1500 * 15])
    pd.testing.assert_frame_equal(full[FEATURES].iloc[:1500], cut[FEATURES])
    assert cut["fwd_ret"].iloc[-HORIZON:].isna().all()  # no label without a real future candle


if __name__ == "__main__":
    test_no_lookahead()
    print("ok")
