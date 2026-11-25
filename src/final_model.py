import json
from pathlib import Path

import numpy as np
import pandas as pd

from cleanup import FEATURES, HORIZON, NUMERIC_FEATURES


DELTA = 0.002
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "out" / "processed" / "features.parquet"
# The final model ships its own scaler and schema: out/processed/scaler.json stays the Phase III one (built by cleanup.py)
MODEL_DIR = PROJECT_ROOT / "out" / "model" / "v1"
SCALER_PATH = MODEL_DIR / "scaler.json"
SCHEMA_PATH = MODEL_DIR / "schema.json"
SCHEMA_VERSION = "1"


def fit_scaler(df: pd.DataFrame) -> dict:
    """Fit clipping bounds and standardization statistics on every training row."""
    train = df.loc[df["split"] == "train", NUMERIC_FEATURES]
    if train.empty:
        raise ValueError("No rows have split='train'")

    clip_low = train.quantile(0.001)
    clip_high = train.quantile(0.999)
    clipped = train.clip(clip_low, clip_high, axis=1)
    mean = clipped.mean()
    std = clipped.std()

    if (std == 0).any():
        zero_variance = std.index[std == 0].tolist()
        raise ValueError(f"Zero-variance numeric features: {zero_variance}")

    return {
        "clip_low": clip_low.to_dict(),
        "clip_high": clip_high.to_dict(),
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "horizon": HORIZON,
        "delta": DELTA,
    }


CLASSES = [-1, 0, 1]  # flat, down, up (as in cleanup.py)
FEE = DELTA  # the round-trip spot taker fee is the same 0.2% that defines the classes
VAL_B_START = pd.Timestamp("2025-03-01")
WEAK_FEATURES = ["lower_wick", "vol_96", "volatility", "rv_15", "normalized_volatility"]  # Phase III importance <= 0


def load_splits(path=FEATURES_PATH) -> dict[str, pd.DataFrame]:
    """train / val_a / val_b / test frames with the 36 features, the label, the forward return and the close."""
    df = pd.read_parquet(path, columns=["open_time", "symbol", "close", "fwd_ret", "label", "split"] + FEATURES)
    val = df[df["split"] == "val"]
    return {"train": df[df["split"] == "train"], "val_a": val[val["open_time"] < VAL_B_START],
            "val_b": val[val["open_time"] >= VAL_B_START], "test": df[df["split"] == "test"]}


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """The 36 features in schema order: numeric ones clipped then standardized, one-hot columns untouched."""
    x = df[FEATURES].astype(float).copy()
    num = scaler["numeric_features"]
    lo, hi = pd.Series(scaler["clip_low"]), pd.Series(scaler["clip_high"])
    x[num] = (x[num].clip(lo[num], hi[num], axis=1) - pd.Series(scaler["mean"])[num]) / pd.Series(scaler["std"])[num]
    return x


def size_weights(fwd_ret) -> np.ndarray:
    w = np.abs(np.asarray(fwd_ret, dtype=float))
    return w / w.mean()


def recency_weights(open_time, half_life_days: float) -> np.ndarray:
    t = pd.to_datetime(pd.Series(open_time))
    age = (t.max() - t).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / half_life_days)
    return w / w.mean()


def class_means(train: pd.DataFrame) -> dict[int, float]:
    return {int(k): float(v) for k, v in train.groupby("label")["fwd_ret"].mean().items()}


def expected_net(proba, classes, means, side: str) -> np.ndarray:
    """Expected return after the fee of acting now: buy gains the move, sell avoids it."""
    m = {int(k): float(v) for k, v in means.items()}  # JSON turns the class keys into strings
    move = np.asarray(proba, dtype=float) @ np.array([m[int(c)] for c in classes])
    return (move if side == "buy" else -move) - FEE


def signal_stats(score, fwd_ret, side: str, margin: float) -> dict:
    score, fwd = np.asarray(score, dtype=float), np.asarray(fwd_ret, dtype=float)
    fire = (score > margin) & ~np.isnan(fwd)
    gain = fwd[fire] if side == "buy" else -fwd[fire]
    n = int(fire.sum())
    return {"n": n, "share": n / max(int((~np.isnan(fwd)).sum()), 1),
            "mean_net": float(np.mean(gain - FEE)) if n else float("nan"),
            "hit_rate": float(np.mean(gain > FEE)) if n else float("nan")}


def random_baseline(fwd_ret, side: str) -> float:
    """Mean net return of acting at every bar: what a signal must beat to be better than acting at random."""
    fwd = np.asarray(fwd_ret, dtype=float)
    fwd = fwd[~np.isnan(fwd)]
    return float(np.mean(fwd if side == "buy" else -fwd) - FEE)


def signal_curve(score, fwd_ret, side: str, shares=(0.05, 0.1, 0.25, 0.5)) -> pd.DataFrame:
    """For each share of bars, the margin that fires on that share and what those signals earned."""
    score = np.asarray(score, dtype=float)
    rows = []
    for share in shares:
        margin = float(np.quantile(score, 1 - share))
        rows.append({"side": side, "target_share": share, "margin": margin, **signal_stats(score, fwd_ret, side, margin)})
    return pd.DataFrame(rows)


def tune_margin(score, fwd_ret, side: str, min_share: float) -> float:
    """The score quantile (50% to 99.5%) with the best mean net return among those firing on >= min_share of bars."""
    score = np.asarray(score, dtype=float)
    best, best_margin = -np.inf, float(np.quantile(score, 1 - min_share))
    for q in np.arange(0.5, 0.9951, 0.005):
        margin = float(np.quantile(score, q))
        s = signal_stats(score, fwd_ret, side, margin)
        if s["share"] >= min_share and s["n"] and s["mean_net"] > best:
            best, best_margin = s["mean_net"], margin
    return best_margin


def simulate_exit(df: pd.DataFrame, buy_score, sell_score, buy_margin: float, sell_margin: float,
                  early_exit: bool) -> dict:
    """A user following the Buy signals: one position per coin, sold after 4 bars or, with early_exit, at the first Sell."""
    frame = df[["symbol", "open_time", "close"]].assign(buy=np.asarray(buy_score), sell=np.asarray(sell_score))
    nets = []
    for _, g in frame.sort_values("open_time").groupby("symbol", sort=False):
        t, c = g["open_time"].to_numpy(), g["close"].to_numpy()
        buy, sell = g["buy"].to_numpy() > buy_margin, g["sell"].to_numpy() > sell_margin
        i = 0
        while i < len(g):
            if not buy[i]:
                i += 1
                continue
            # Exits are found by time, not row offset: bars can be missing, so the 4th row may be more than an hour away
            deadline = t[i] + np.timedelta64(60, "m")
            k = None
            if early_exit:
                for m in range(i + 1, len(g)):
                    if t[m] > deadline:
                        break
                    if sell[m]:
                        k = m
                        break
            if k is None:
                m = int(np.searchsorted(t, deadline))
                k = m if m < len(g) and t[m] == deadline else None
            if k is not None and not np.isnan(c[k]):
                nets.append(c[k] / c[i] - 1 - FEE)
                i = k  # the position is closed at bar k; a new Buy can start there
            else:
                i += 1  # exit bar missing: skip this trade
    nets = np.array(nets)
    return {"trades": len(nets), "mean_net": float(nets.mean()) if len(nets) else float("nan"),
            "hit_rate": float((nets > 0).mean()) if len(nets) else float("nan")}


def decide(proba_row, classes, decision: dict) -> str:
    """BUY, SELL or WAIT for one bar, from its class probabilities and the shipped margins."""
    p = np.asarray(proba_row, dtype=float).reshape(1, -1)
    buy = expected_net(p, classes, decision["means"], "buy")[0]
    sell = expected_net(p, classes, decision["means"], "sell")[0]
    if buy > decision["buy_margin"] and buy >= sell:
        return "BUY"
    if decision.get("sell_enabled", True) and sell > decision["sell_margin"]:  # Sell can be switched off in decision.json
        return "SELL"
    return "WAIT"


def save_artifact(dirpath, model, scaler: dict, decision: dict) -> None:
    import joblib
    import sklearn

    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.pkl")
    files = {"scaler.json": scaler, "schema.json": {"schema_version": SCHEMA_VERSION, "features": FEATURES},
             "decision.json": decision,
             "versions.json": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    for name, value in files.items():
        (d / name).write_text(json.dumps(value, indent=1))


def load_artifact(dirpath):
    # model.pkl is a joblib pickle, which can run code when loaded: only ever load the artifact our own training wrote
    # (the model container mounts it read-only), never a file supplied by a user
    import joblib

    d = Path(dirpath)
    read = lambda name: json.loads((d / name).read_text())
    return joblib.load(d / "model.pkl"), read("scaler.json"), read("schema.json"), read("decision.json")


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    scaler = fit_scaler(df)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCALER_PATH.open("w") as file:
        json.dump(scaler, file, indent=1)
    with SCHEMA_PATH.open("w") as file:
        json.dump({"schema_version": SCHEMA_VERSION, "features": FEATURES}, file, indent=2)

    print(df["label"].value_counts().sort_index().rename(index={-1: "flat", 0: "down", 1: "up"}).to_string())


if __name__ == "__main__":
    main()