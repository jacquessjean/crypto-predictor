"""Writes src/horizons.ipynb (trustworthy-Buy plan, Tasks 3-4). Run from the repo root."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Trustworthy Buy signals: horizon and model comparison

Design: docs/superpowers/specs/2026-11-25-trustworthy-buy-design.md. Compares horizons of 1 hour, 4 hours and 1 day,
each with a Random Forest and a HistGradientBoosting model, by what a user following the Buy signals would get on
val-B. The criterion and the November pass rule were fixed before any result was seen."""),
    code('''import ast
import os
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss

from cleanup import FEATURES, TEST_START, VAL_START
from fresh_data import cached_fresh
from final_model import (CLASSES, FEE, MODEL_DIR, VAL_B_START, WEAK_FEATURES, apply_scaler, embargo, expected_net,
                         fit_scaler, horizon_return, random_buy, save_artifact, simulate_exit, size_weights, three_class,
                         tune_margin_trades)

RESULTS = os.path.join("..", "out", "results")
SEARCH = False  # True re-runs the comparison; False reloads out/results/horizon_*.csv
HORIZONS = {4: "1 hour", 16: "4 hours", 96: "1 day"}
SEEDS = [0, 1, 2]
COLS = [c for c in FEATURES if c not in WEAK_FEATURES]  # v1's feature selection

def save(results, name):
    results.to_csv(os.path.join(RESULTS, f"{name}.csv"), index=False)

def parse(v):
    if v == "":
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v

def search(name, run):
    path = os.path.join(RESULTS, f"{name}.csv")
    if not SEARCH and os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False, dtype=str).apply(lambda col: col.map(parse))
    results = run()
    save(results, name)
    return results

df = pd.read_parquet(os.path.join("..", "out", "processed", "features.parquet"),
                     columns=["open_time", "symbol", "close", "split"] + FEATURES)
df = df[df["split"].isin(["train", "val"])]

def horizon_data(H):
    """Splits with labels for horizon H, each embargoed so no label looks into the next split."""
    d = df.assign(fwd=horizon_return(df, H))
    d = d[d["fwd"].notna()].assign(label=lambda x: three_class(x["fwd"]).astype(int))
    tr = embargo(d[d["split"] == "train"], VAL_START, H)
    val = d[d["split"] == "val"]
    va = embargo(val[val["open_time"] < VAL_B_START], VAL_B_START, H)
    vb = embargo(val[val["open_time"] >= VAL_B_START], TEST_START, H)
    return tr, va, vb

def make_model(kind, seed, iters=None):
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", criterion="log_loss",
                                      n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.05, max_iter=iters, max_features=0.8,
                                          early_stopping=False, random_state=seed)'''),
    md("## Candidates on val-A (tuning) and val-B (judging)"),
    code('''def hgb_iterations(Xtr, ytr, w, Xa, ya):
    """Iterations by val-A log loss (seed 0), growing one model with warm_start."""
    m = make_model("gradient_boosting", 0, 100).set_params(warm_start=True)
    rows = []
    for iters in [100, 200, 400, 800]:
        m.set_params(max_iter=iters).fit(Xtr, ytr, sample_weight=w)
        rows.append({"iters": iters, "val_a_log_loss": log_loss(ya, m.predict_proba(Xa), labels=CLASSES)})
    return pd.DataFrame(rows)

def run_candidates():
    rows, iter_rows = [], []
    for H, name in HORIZONS.items():
        tr, va, vb = horizon_data(H)
        scaler = fit_scaler(tr)
        Xtr, Xa, Xb = (apply_scaler(d, scaler)[COLS] for d in (tr, va, vb))
        ytr, ya, yb = tr["label"].to_numpy(), va["label"].to_numpy(), vb["label"].to_numpy()
        w = size_weights(tr["fwd"])
        means = {int(k): float(v) for k, v in tr.groupby("label")["fwd"].mean().items()}
        rand_b = random_buy(vb, H)
        it = hgb_iterations(Xtr, ytr, w, Xa, ya).assign(horizon=name)
        iter_rows.append(it)
        best_iters = int(it.sort_values("val_a_log_loss").iloc[0]["iters"])
        for kind in ["random_forest", "gradient_boosting"]:
            for seed in SEEDS:
                t0 = time.time()
                base = make_model(kind, seed, best_iters).fit(Xtr, ytr, sample_weight=w)
                model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(Xa, ya)
                pa, pb, cls = model.predict_proba(Xa), model.predict_proba(Xb), list(model.classes_)
                score_a, score_b = expected_net(pa, cls, means, "buy"), expected_net(pb, cls, means, "buy")
                margin = tune_margin_trades(va, score_a, H, min_trades=100)
                in_a = simulate_exit(va, score_a, np.zeros(len(va)), margin, 1.0, early_exit=False, horizon_bars=H)
                out_b = simulate_exit(vb, score_b, np.zeros(len(vb)), margin, 1.0, early_exit=False, horizon_bars=H)
                rows.append({"horizon": name, "H": H, "model": kind, "seed": seed,
                             "iters": best_iters if kind == "gradient_boosting" else None,
                             "val_a_log_loss": log_loss(ya, pa, labels=CLASSES), "val_b_log_loss": log_loss(yb, pb, labels=CLASSES),
                             "margin": margin, "val_a_trades": in_a["trades"], "val_a_mean_net": in_a["mean_net"],
                             "val_b_trades": out_b["trades"], "val_b_mean_net": out_b["mean_net"], "val_b_hit": out_b["hit_rate"],
                             "val_b_random_mean_net": rand_b["mean_net"], "val_b_random_trades": rand_b["trades"],
                             "seconds": time.time() - t0})
                print(rows[-1]["horizon"], kind, seed, f"val-B trades {out_b['trades']}, mean net {100 * out_b['mean_net']:+.3f}%")
    save(pd.concat(iter_rows, ignore_index=True), "horizon_hgb_iters")
    return pd.DataFrame(rows)

candidates = search("horizon_candidates", run_candidates)'''),
    code('''# Criterion fixed before any result was seen (design, "Pre-registered criterion")
summary = (candidates.groupby(["H", "horizon", "model"])
           .agg(trades=("val_b_trades", "mean"), mean_net=("val_b_mean_net", "mean"), std=("val_b_mean_net", "std"),
                hit=("val_b_hit", "mean"), random=("val_b_random_mean_net", "mean"), val_a_mean_net=("val_a_mean_net", "mean"))
           .reset_index())
summary["trustworthy"] = (summary.trades >= 100) & (summary.mean_net > 0) & (summary.mean_net > summary.random)
print(summary.assign(**{c: summary[c].map(lambda v: f"{100 * v:+.3f}%") for c in ["mean_net", "std", "random", "val_a_mean_net"]}).to_string())
ok = summary[summary.trustworthy].sort_values("mean_net", ascending=False)
if ok.empty:
    chosen = None
    print("No candidate is trustworthy on val-B: v1 stays, and November is not used.")
else:
    top = ok.iloc[0]
    tied = ok[ok.mean_net >= top.mean_net - top["std"]]  # within the best candidate's seed std
    # ties go to the shorter horizon, then to the Random Forest (simpler, already integrated)
    chosen = tied.assign(rf_first=(tied.model != "random_forest")).sort_values(["H", "rf_first"]).iloc[0]
    print(f"chosen: {chosen.horizon}, {chosen.model} (val-B mean net {100 * chosen.mean_net:+.3f}% over {chosen.trades:.0f} trades)")
save(summary, "horizon_summary")'''),
    md("""## November 2026: one check on data nobody has seen

Pass rule, fixed before November was loaded: at least 30 trades, mean net return per trade above zero, and better
than buying at random in November. Fewer than 30 trades is inconclusive. The model is the chosen candidate with
seed 0, trained on the training period only."""),
    code('''assert chosen is not None, "no trustworthy candidate on val-B: nothing to confirm"
H, kind = int(chosen.H), chosen.model
seed0 = candidates[(candidates.H == H) & (candidates.model == kind) & (candidates.seed == 0)].iloc[0]
tr, va, vb = horizon_data(H)
scaler = fit_scaler(tr)
Xtr, Xa = (apply_scaler(d, scaler)[COLS] for d in (tr, va))
means = {int(k): float(v) for k, v in tr.groupby("label")["fwd"].mean().items()}
base = make_model(kind, 0, None if kind == "random_forest" else int(seed0.iters)).fit(Xtr, tr["label"], sample_weight=size_weights(tr["fwd"]))
model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(Xa, va["label"])
cls = list(model.classes_)
margin = tune_margin_trades(va, expected_net(model.predict_proba(Xa), cls, means, "buy"), H, min_trades=100)
assert np.isclose(margin, seed0.margin), "the retrained seed-0 model must reproduce the selected candidate"

def score_november():
    """The single November check. Once horizon_november.csv exists, search() reloads it and November is not scored again."""
    path = os.path.join("..", "out", "processed", "fresh_november.parquet")  # downloaded and built on first use
    nov = cached_fresh(path).dropna(subset=FEATURES)
    nov = nov.assign(fwd=horizon_return(nov, H))
    nov = nov[(nov["open_time"] >= "2026-11-01") & nov["fwd"].notna()]  # complete labels only
    score = expected_net(model.predict_proba(apply_scaler(nov, scaler)[COLS]), cls, means, "buy")
    res = simulate_exit(nov, score, np.zeros(len(nov)), margin, 1.0, early_exit=False, horizon_bars=H, with_trades=True)
    rand = random_buy(nov, H)
    rng = np.random.default_rng(0)
    boot = [rng.choice(res["nets"], len(res["nets"])).mean() for _ in range(2000)] if res["trades"] else [np.nan]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = ("inconclusive" if res["trades"] < 30 else
               "pass" if res["mean_net"] > 0 and res["mean_net"] > rand["mean_net"] else "fail")
    return pd.DataFrame([{"horizon": chosen.horizon, "model": kind, "bars": len(nov), "trades": res["trades"],
                          "mean_net": res["mean_net"], "ci_low": lo, "ci_high": hi, "hit_rate": res["hit_rate"],
                          "random_mean_net": rand["mean_net"], "random_trades": rand["trades"], "verdict": verdict}])

november = search("horizon_november", score_november)
verdict = november["verdict"].iloc[0]
print(november.T.to_string())'''),
    code('''from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

V2 = MODEL_DIR.parent / "v2"
if verdict == "pass":
    select = (ColumnTransformer([("keep", "passthrough", COLS)], remainder="drop", verbose_feature_names_out=False)
              .set_output(transform="pandas").fit(apply_scaler(va, scaler)))
    shipped = Pipeline([("select", select), ("model", model)])
    decision = {"classes": CLASSES, "means": {str(k): v for k, v in means.items()}, "fee": FEE, "buy_margin": float(margin),
                "sell_margin": 0.0, "sell_enabled": False, "early_exit": False, "horizon_minutes": 15 * H,
                "model": kind, "criterion": "log_loss", "variant": "drop_weak+size", "calibration": "sigmoid",
                "margin_rule": "tuned on non-overlapping val-A trades, at least 100 trades"}
    save_artifact(V2, shipped, scaler, decision)
    np.savez(V2 / "check.npz", proba=shipped.predict_proba(apply_scaler(vb.iloc[:100], scaler)))
    print("v2 saved:", sorted(p.name for p in V2.iterdir()))
else:
    print(f"November verdict: {verdict}. v1 stays; no v2 is written.")'''),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "src/horizons.ipynb")
print("written", len(cells), "cells")
