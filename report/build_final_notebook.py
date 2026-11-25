"""Writes src/final_model.ipynb (Tasks 3-5 of the final-model plan). Run from the repo root."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Final model (Phase V)

Implements the model committed in the Phase IV design (report/ML - Project Phase IV.pdf): a 3-class Random Forest
(up / flat / down beyond the 0.2% fee over the next hour), a fee-aware Buy / Sell / Wait rule, and a protocol that tunes
on val-A (Oct 2024 - Feb 2025), confirms on val-B (Mar - Jun 2025) and evaluates the test period once. Every choice is
made by a rule fixed in the code before the results are seen."""),
    code('''import os
import time
import ast

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from final_model import (CLASSES, FEATURES, FEE, MODEL_DIR, WEAK_FEATURES, apply_scaler, class_means, expected_net,
                         fit_scaler, load_splits, recency_weights, save_artifact, signal_curve, signal_stats,
                         simulate_exit, size_weights, tune_margin)

RESULTS = os.path.join("..", "out", "results")
SEARCH = False  # True re-runs every experiment; False reloads the saved out/results/final_*.csv

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
    """Run an experiment and save it, or reload the saved one when SEARCH is False."""
    path = os.path.join(RESULTS, f"{name}.csv")
    if not SEARCH and os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False, dtype=str).apply(lambda col: col.map(parse))
    results = run()
    save(results, name)
    return results

splits = load_splits()
train, val_a, val_b, test = (splits[k] for k in ["train", "val_a", "val_b", "test"])
scaler = fit_scaler(train)  # all training rows, flat ones included
Xtr, Xa, Xb = (apply_scaler(d, scaler) for d in (train, val_a, val_b))
ytr, ya, yb = (d["label"].to_numpy() for d in (train, val_a, val_b))
means = class_means(train)
prior = train["label"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
prior_ll = log_loss(ya, np.tile(prior, (len(ya), 1)), labels=CLASSES)
print({k: len(v) for k, v in splits.items()})
print("class mean returns (train):", {k: f"{100 * v:+.3f}%" for k, v in means.items()})
print(f"val-A log loss of the class prior: {prior_ll:.5f}")'''),
    code('''def brier(y, proba, classes):
    onehot = (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(((proba - onehot) ** 2).sum(axis=1).mean())

def evaluate(model, X, df):
    """Probability quality and what Buy / Sell signals at fixed signal rates would have earned."""
    p, cls = model.predict_proba(X), list(model.classes_)
    buy = signal_curve(expected_net(p, cls, means, "buy"), df["fwd_ret"], "buy").set_index("target_share")
    sell = signal_curve(expected_net(p, cls, means, "sell"), df["fwd_ret"], "sell").set_index("target_share")
    return {"log_loss": log_loss(df["label"], p, labels=cls), "brier": brier(df["label"], p, cls),
            "buy_mean_net@5%": buy.loc[0.05, "mean_net"], "buy_mean_net@10%": buy.loc[0.1, "mean_net"],
            "buy_mean_net@25%": buy.loc[0.25, "mean_net"], "buy_hit@10%": buy.loc[0.1, "hit_rate"],
            "sell_mean_net@10%": sell.loc[0.1, "mean_net"]}

def forest(seed, criterion="log_loss"):
    return RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", criterion=criterion,
                                  n_jobs=-1, random_state=seed)

SEEDS = [0, 1, 2]
METRICS = ["log_loss", "brier", "buy_mean_net@5%", "buy_mean_net@10%", "buy_mean_net@25%", "buy_hit@10%", "sell_mean_net@10%"]'''),
    md("## Baseline: log-loss criterion against the default Gini (val-A, three seeds)"),
    code('''def run_baseline():
    rows = []
    for criterion in ["log_loss", "gini"]:
        for seed in SEEDS:
            t0 = time.time()
            m = forest(seed, criterion).fit(Xtr, ytr)
            rows.append({"criterion": criterion, "seed": seed, **evaluate(m, Xa, val_a), "fit_time": time.time() - t0})
    return pd.DataFrame(rows)

baseline = search("final_baseline", run_baseline)
summary = baseline.groupby("criterion")[METRICS].agg(["mean", "std"])
print(summary.T.to_string())
# Rule fixed before running: keep the design's log-loss criterion unless Gini beats it on the Buy metric by more than
# the seed std of log loss
ll, gi = summary.loc["log_loss"], summary.loc["gini"]
criterion = "gini" if gi[("buy_mean_net@10%", "mean")] - ll[("buy_mean_net@10%", "mean")] > ll[("buy_mean_net@10%", "std")] else "log_loss"
print("criterion:", criterion, "| every log loss below the prior:", bool((baseline["log_loss"] < prior_ll).all()))'''),
    md("## Sample weighting and feature selection (val-A, three seeds)"),
    code('''def spec(variant):
    """Sample weights and input columns of a variant; '+' combines two variants."""
    w, cols = np.ones(len(train)), list(FEATURES)
    for part in variant.split("+"):
        if part == "size":
            w = w * size_weights(train["fwd_ret"])
        elif part.startswith("recency_"):
            w = w * recency_weights(train["open_time"], int(part.split("_")[1]))
        elif part == "drop_weak":
            cols = [c for c in cols if c not in WEAK_FEATURES]
    return w / w.mean(), cols

def run_variant(variant):
    w, cols = spec(variant)
    return [{"variant": variant, "seed": seed, **evaluate(forest(seed, criterion).fit(Xtr[cols], ytr, sample_weight=w), Xa[cols], val_a)}
            for seed in SEEDS]

def run_weighting():
    rows = [r for v in ["none", "size", "recency_365", "recency_730", "drop_weak"] for r in run_variant(v)]
    df = pd.DataFrame(rows)
    singles = df[df.variant != "none"].groupby("variant")["buy_mean_net@10%"].mean().sort_values(ascending=False)
    pair = sorted(singles.index[:2])
    if not (pair[0].startswith("recency") and pair[1].startswith("recency")):  # two half-lives do not combine
        df = pd.concat([df, pd.DataFrame(run_variant("+".join(pair)))], ignore_index=True)
    return df

weighting = search("final_weighting", run_weighting)
wsum = weighting.groupby("variant")[["log_loss", "buy_mean_net@10%", "sell_mean_net@10%"]].agg(["mean", "std"])
print(wsum.sort_values(("buy_mean_net@10%", "mean"), ascending=False).to_string())
# Rule fixed before running: a variant must beat 'none' on the Buy metric by more than its own seed std, else 'none'
base = wsum.loc["none", ("buy_mean_net@10%", "mean")]
gain = wsum[("buy_mean_net@10%", "mean")] - base - wsum[("buy_mean_net@10%", "std")]
variant = gain.drop("none").idxmax() if (gain.drop("none") > 0).any() else "none"
print("variant:", variant)'''),
    md("## Calibration, decision margins and the exit rule (tune on val-A, confirm on val-B)"),
    code('''weights, cols = spec(variant)
model = forest(0, criterion).fit(Xtr[cols], ytr, sample_weight=weights)
select = ColumnTransformer([("keep", "passthrough", cols)], remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas").fit(Xa)  # 36 inputs in, the kept ones out, names kept
candidates = {"none": model}
for method in ["sigmoid", "isotonic"]:
    candidates[method] = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(Xa[cols], ya)
cal_rows = [{"calibration": k, "val_a_log_loss": log_loss(ya, c.predict_proba(Xa[cols]), labels=CLASSES),
             "val_b_log_loss": log_loss(yb, c.predict_proba(Xb[cols]), labels=CLASSES),
             "val_b_brier": brier(yb, c.predict_proba(Xb[cols]), list(c.classes_))} for k, c in candidates.items()]
cal = pd.DataFrame(cal_rows)
print(cal.to_string())
# The design calibrates: pick sigmoid or isotonic by val-B log loss (the uncalibrated row is for reference)
method = cal[cal.calibration != "none"].sort_values("val_b_log_loss").iloc[0]["calibration"]
chosen = Pipeline([("select", select), ("model", candidates[method])])
print("calibration:", method)

pa, pb, cls = chosen.predict_proba(Xa), chosen.predict_proba(Xb), list(chosen.classes_)
score = {(s, k): expected_net(p, cls, means, s) for s in ["buy", "sell"] for k, p in [("a", pa), ("b", pb)]}
margins = {s: tune_margin(score[(s, "a")], val_a["fwd_ret"], s, min_share=0.01) for s in ["buy", "sell"]}
rows = []
for s in ["buy", "sell"]:
    stats_b = signal_stats(score[(s, "b")], val_b["fwd_ret"], s, margins[s])
    if not stats_b["mean_net"] > 0:
        print(f"No setting beats the fee on val-B ({s}): firing only when the expected net return is positive")
        margins[s] = max(margins[s], 0.0)
    for split, df, k in [("val_a", val_a, "a"), ("val_b", val_b, "b")]:
        rows.append({"side": s, "split": split, "margin": margins[s], **signal_stats(score[(s, k)], df["fwd_ret"], s, margins[s])})
decision_table = pd.DataFrame(rows)
curve_b = pd.concat([signal_curve(score[(s, "b")], val_b["fwd_ret"], s) for s in ["buy", "sell"]])
decision = {"classes": CLASSES, "means": {str(k): v for k, v in means.items()}, "fee": FEE,
            "buy_margin": margins["buy"], "sell_margin": margins["sell"], "criterion": criterion,
            "variant": variant, "calibration": method}
save(cal.assign(chosen=cal.calibration == method), "final_calibration")
save(decision_table, "final_decision")
save(curve_b, "final_curve_val_b")
print(decision_table.to_string())
print(curve_b.to_string())'''),
    code('''exit_rows = [{"early_exit": e, **simulate_exit(val_b, score[("buy", "b")], score[("sell", "b")], margins["buy"], margins["sell"], early_exit=e)}
             for e in [False, True]]
exit_table = pd.DataFrame(exit_rows)
save(exit_table, "final_exit")
early_exit = bool(exit_table.sort_values("mean_net", ascending=False).iloc[0]["early_exit"]) if exit_table["trades"].min() > 0 else False
decision["early_exit"] = early_exit
# Decided after seeing val-B, and disclosed as such: Sell signals lost far more than selling at random on val-B
# (see docs/final-model-results.md), so the shipped rule never emits Sell. Buy and the exit rule do not depend on it.
decision["sell_enabled"] = False
print(exit_table.to_string())
print("exit rule:", "sell at the first Sell signal within the hour" if early_exit else "hold the full hour")'''),
    md("""## Test period, evaluated once

This cell runs once, after every choice above was fixed by its rule (criterion, weighting, calibration, margins, exit
rule). Phase III evaluated binary models on this period; the 3-class model and its decision rule never were."""),
    code('''Xte, yt = apply_scaler(test, scaler), test["label"].to_numpy()
pt, cls = chosen.predict_proba(Xte), list(chosen.classes_)
bt, st = expected_net(pt, cls, means, "buy"), expected_net(pt, cls, means, "sell")
test_row = {"rows": len(test), "log_loss": log_loss(yt, pt, labels=CLASSES),
            "prior_log_loss": log_loss(yt, np.tile(prior, (len(yt), 1)), labels=CLASSES), "brier": brier(yt, pt, cls),
            **{f"buy_{k}": v for k, v in signal_stats(bt, test["fwd_ret"], "buy", decision["buy_margin"]).items()},
            **{f"sell_{k}": v for k, v in signal_stats(st, test["fwd_ret"], "sell", decision["sell_margin"]).items()},
            **{f"trade_{k}": v for k, v in simulate_exit(test, bt, st, decision["buy_margin"], decision["sell_margin"], decision["early_exit"]).items()}}
final_test = pd.DataFrame([test_row])
save(final_test, "final_test")
print(final_test.T.to_string())'''),
    md("## Model artifact for the model service"),
    code('''save_artifact(MODEL_DIR, chosen, scaler, decision)
check = apply_scaler(val_b.iloc[:100], scaler)
np.savez(MODEL_DIR / "check.npz", proba=chosen.predict_proba(check))  # lets a test prove the saved model reloads identically
print(sorted(p.name for p in MODEL_DIR.iterdir()))
print(open(MODEL_DIR / "decision.json").read())'''),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "src/final_model.ipynb")
print("written", len(cells), "cells")
