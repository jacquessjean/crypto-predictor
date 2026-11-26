import json
import os
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


def load_splits(path=FEATURES_PATH, include_test: bool = True) -> dict[str, pd.DataFrame]:
    """train / val_a / val_b / test frames with the 36 features, the label, the forward return and the close.
    include_test=False skips reading the test rows (retraining never uses them, and it halves the memory)."""
    filters = None if include_test else [("split", "in", ["train", "val"])]
    df = pd.read_parquet(path, columns=["open_time", "symbol", "close", "fwd_ret", "label", "split"] + FEATURES,
                         filters=filters)
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
                  early_exit: bool, horizon_bars: int = 4, with_trades: bool = False) -> dict:
    """A user following the Buy signals: one position per coin, sold after horizon_bars bars or, with early_exit, at the
    first Sell."""
    frame = df[["symbol", "open_time", "close"]].assign(buy=np.asarray(buy_score), sell=np.asarray(sell_score))
    nets, details = [], []
    for sym, g in frame.sort_values("open_time").groupby("symbol", sort=False):
        t, c = g["open_time"].to_numpy(), g["close"].to_numpy()
        buy, sell = g["buy"].to_numpy() > buy_margin, g["sell"].to_numpy() > sell_margin
        i = 0
        while i < len(g):
            if not buy[i]:
                i += 1
                continue
            # Exits are found by time, not row offset: bars can be missing, so the 4th row may be more than an hour away
            deadline = t[i] + np.timedelta64(15 * horizon_bars, "m")
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
                details.append({"symbol": sym, "entry_time": pd.Timestamp(t[i]), "exit_time": pd.Timestamp(t[k]),
                                "entry": float(c[i]), "exit": float(c[k]), "net": nets[-1]})
                i = k  # the position is closed at bar k; a new Buy can start there
            else:
                i += 1  # exit bar missing: skip this trade
    nets = np.array(nets)
    out = {"trades": len(nets), "mean_net": float(nets.mean()) if len(nets) else float("nan"),
           "hit_rate": float((nets > 0).mean()) if len(nets) else float("nan")}
    return {**out, "nets": nets, "details": details} if with_trades else out  # per trade, e.g. for a bootstrap


def horizon_return(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """close[t+H] / close[t] - 1 per coin; NaN unless the bar exactly H x 15 minutes later exists."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("symbol"):
        later = g.set_index("open_time")["close"].reindex(g["open_time"] + pd.Timedelta(minutes=15 * horizon_bars))
        out.loc[g.index] = later.to_numpy() / g["close"].to_numpy() - 1
    return out


def three_class(fwd) -> np.ndarray:
    """1 up / 0 down / -1 flat at the fee, as in cleanup.py; NaN stays NaN."""
    fwd = np.asarray(fwd, dtype=float)
    return np.where(np.isnan(fwd), np.nan, np.select([fwd > DELTA, fwd < -DELTA], [1, 0], -1))


def embargo(df: pd.DataFrame, next_start: pd.Timestamp, horizon_bars: int) -> pd.DataFrame:
    """Drops the rows whose label would look into the next split."""
    return df[df["open_time"] + pd.Timedelta(minutes=15 * horizon_bars) < next_start]


def random_buy(df: pd.DataFrame, horizon_bars: int) -> dict:
    """The baseline a Buy rule must beat: buying whenever no position is open, selling H bars later."""
    ones = np.ones(len(df))
    return simulate_exit(df, ones, np.zeros(len(df)), 0.0, 1.0, early_exit=False, horizon_bars=horizon_bars)


def tune_margin_trades(df: pd.DataFrame, score, horizon_bars: int, min_trades: int) -> float:
    """The score quantile (50% to 99.5%) with the best mean net per non-overlapping trade among those with at least
    min_trades trades."""
    score = np.asarray(score, dtype=float)
    zeros = np.zeros(len(df))
    best, best_margin = -np.inf, float(np.quantile(score, 0.5))
    for q in np.arange(0.5, 0.9951, 0.005):
        margin = float(np.quantile(score, q))
        res = simulate_exit(df, score, zeros, margin, 1.0, early_exit=False, horizon_bars=horizon_bars)
        if res["trades"] >= min_trades and res["mean_net"] > best:
            best, best_margin = res["mean_net"], margin
    return best_margin


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

# ---------------------------------------------------------------------------- online learning
class FeedbackStore:
    """Labeled samples sent back by the application (features + realized return), one per pair and bar."""

    def __init__(self, path):
        import sqlite3

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connect = lambda: sqlite3.connect(self.path, timeout=10.0)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS feedback (pair TEXT NOT NULL, bar_time TEXT NOT NULL,
                            features_json TEXT NOT NULL, realized_return REAL NOT NULL, received_at TEXT NOT NULL,
                            UNIQUE(pair, bar_time))""")

    def add(self, samples: list[dict]) -> tuple[int, int]:
        """Stores new samples; returns (accepted, duplicates)."""
        now = pd.Timestamp.now(tz="UTC").isoformat()
        accepted = 0
        with self._connect() as conn:
            for s in samples:
                cur = conn.execute("INSERT OR IGNORE INTO feedback VALUES (?, ?, ?, ?, ?)",
                                   (s["pair"], s["bar_time"], json.dumps(s["features"]), float(s["realized_return"]), now))
                accepted += cur.rowcount
        return accepted, len(samples) - accepted

    def load(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT pair, bar_time, features_json, realized_return, received_at FROM feedback").fetchall()
        feats = pd.DataFrame([json.loads(r[2]) for r in rows], columns=FEATURES)
        meta = pd.DataFrame([(r[0], r[1], r[3], r[4]) for r in rows], columns=["pair", "bar_time", "realized_return", "received_at"])
        return pd.concat([meta, feats], axis=1)

    def count_since(self, received_after) -> int:
        """Samples received after a time (ISO string); all of them when received_after is None."""
        with self._connect() as conn:
            if received_after is None:
                return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM feedback WHERE received_at > ?", (received_after,)).fetchone()[0]


def fit_v2(train: pd.DataFrame, val_a: pd.DataFrame):
    """The v2 recipe: scaler on train, Random Forest with move-size weights and the weak features dropped, sigmoid
    calibration on val-A. Returns (pipeline taking the 36 scaled features, scaler, class means)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.pipeline import Pipeline

    cols = [c for c in FEATURES if c not in WEAK_FEATURES]
    scaler = fit_scaler(train)
    Xtr, Xa = apply_scaler(train, scaler), apply_scaler(val_a, scaler)
    forest = RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", criterion="log_loss",
                                    n_jobs=-1, random_state=0)
    forest.fit(Xtr[cols], train["label"].astype(int), sample_weight=size_weights(train["fwd_ret"]))
    calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid").fit(Xa[cols], val_a["label"].astype(int))
    select = (ColumnTransformer([("keep", "passthrough", cols)], remainder="drop", verbose_feature_names_out=False)
              .set_output(transform="pandas").fit(Xa))
    return Pipeline([("select", select), ("model", calibrated)]), scaler, class_means(train)


def _gate(live, new, val_b: pd.DataFrame, tolerance: float, baseline: float | None = None) -> dict:
    """The safety gate: on val-B the new model must not be worse than the live one, nor than the first model of the
    chain (`baseline`, so promotions cannot drift down by a tolerance at a time), by more than `tolerance` in log loss,
    and its probabilities must be valid and cover all three classes."""
    from sklearn.metrics import log_loss

    (lm, ls), (nm, ns) = live, new
    pl, pn = lm.predict_proba(apply_scaler(val_b, ls)), nm.predict_proba(apply_scaler(val_b, ns))
    y = val_b["label"].astype(int)
    live_ll, new_ll = log_loss(y, pl, labels=CLASSES), log_loss(y, pn, labels=CLASSES)
    baseline = live_ll if baseline is None else baseline
    valid = bool(np.isfinite(pn).all() and np.allclose(pn.sum(axis=1), 1.0, atol=1e-9) and set(nm.classes_) == set(CLASSES))
    reason = ("invalid probabilities or missing class" if not valid else
              f"log loss on val-B {new_ll:.5f} is worse than the live model's {live_ll:.5f} by more than {tolerance}"
              if new_ll > live_ll + tolerance else
              f"log loss on val-B {new_ll:.5f} is worse than the baseline's {baseline:.5f} by more than {tolerance}"
              if new_ll > baseline + tolerance else "")
    return {"passed": not reason, "reason": reason, "live_log_loss": live_ll, "new_log_loss": new_ll,
            "baseline_log_loss": baseline}


def new_feedback(store: "FeedbackStore", live_dir) -> int:
    """Feedback samples received after the live model was trained (all of them for a model never retrained)."""
    trained_on_path = Path(live_dir) / "trained_on.json"
    trained_on = json.loads(trained_on_path.read_text()) if trained_on_path.exists() else {}
    return store.count_since(trained_on.get("feedback_received_until"))


def retrain(base_path, store: "FeedbackStore", live_dir, models_root, min_new: int = 50,
            tolerance: float = 0.0005) -> dict:
    """One online-learning step: retrain on the training data plus all feedback, gate it against the live model, and
    promote it as the next version (writing current.json) only if it passes."""
    live_dir, models_root = Path(live_dir), Path(models_root)
    trained_on_path = live_dir / "trained_on.json"
    trained_on = json.loads(trained_on_path.read_text()) if trained_on_path.exists() else {}
    new = new_feedback(store, live_dir)
    report = {"status": "skipped", "reason": "", "version": None, "new_samples": new}
    if new < min_new:
        return {**report, "reason": f"{new} new feedback samples, {min_new} needed"}

    splits, fb = load_splits(base_path, include_test=False), store.load()
    fb_rows = fb.assign(symbol=fb["pair"], open_time=pd.to_datetime(fb["bar_time"], utc=True).dt.tz_localize(None),
                        fwd_ret=fb["realized_return"], label=three_class(fb["realized_return"]).astype(int), split="train",
                        close=np.nan)  # close is only used by the exit simulation, never for training
    train = pd.concat([splits["train"], fb_rows[splits["train"].columns]], ignore_index=True)
    model, scaler, means = fit_v2(train, splits["val_a"])
    live_model, live_scaler, _, live_decision = load_artifact(live_dir)
    gate = _gate((live_model, live_scaler), (model, scaler), splits["val_b"], tolerance,
                 baseline=trained_on.get("baseline_log_loss"))
    report.update(live_log_loss=gate["live_log_loss"], new_log_loss=gate["new_log_loss"])
    if not gate["passed"]:
        return {**report, "status": "rejected", "reason": gate["reason"]}

    numbers = [int(p.name[1:]) for p in models_root.glob("v*") if p.name[1:].isdigit()]
    version = f"v{max(numbers) + 1}"
    decision = {**live_decision, "means": {str(k): v for k, v in means.items()}}  # margin, Sell switch, horizon kept
    save_artifact(models_root / version, model, scaler, decision)
    (models_root / version / "trained_on.json").write_text(json.dumps({
        "base_rows": int(len(splits["train"])), "feedback_rows": int(len(fb)),
        "feedback_received_until": fb["received_at"].max(), "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "previous_version": live_dir.name, "baseline_log_loss": gate["baseline_log_loss"], "gate": gate},
        indent=1, default=float))
    pointer = models_root / "current.json"
    tmp = pointer.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": version}))
    os.replace(tmp, pointer)  # atomic: a crash never leaves a half-written pointer that would stop the service
    return {**report, "status": "promoted", "version": version}
