from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleanup import FEATURES, OUT as PROCESSED, SYMBOLS, load_symbol

OUT = Path("out/report")
# One small figure per feature group for the report (time features are uniform by construction, so no figure).
GROUPS = {
    "candle": ["returns", "body_ratio", "volatility", "upper_wick", "lower_wick"],
    "momentum": ["ret_4", "ret_16", "ret_96", "ret_672"],
    "volatility": ["rv_15", "rv_60", "rv_ratio", "vol_96", "normalized_volatility"],
    "activity": ["taker_buy_ratio", "taker_buy_ratio_16", "relative_volume", "rel_trade_size", "idle_minutes"],
    "market": ["btc_ret_4", "btc_ret_16", "rel_ret_4"],
}


def group_figures(train: pd.DataFrame) -> None:
    for name, cols in GROUPS.items():
        rows = (len(cols) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(5, 1.9 * rows))
        for ax in axes.flat[len(cols):]:
            ax.axis("off")
        for ax, f in zip(axes.flat, cols):
            x = train[f]
            lo, hi = x.quantile([0.005, 0.995])
            ax.hist(x[x.between(lo, hi)], bins=60, color="#4c72b0")  # hide the 1% extremes for readability
            ax.set_title(f, fontsize=10)
            ax.tick_params(labelsize=7)
            ax.locator_params(axis="x", nbins=4)
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"dist_{name}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Raw 1m data, one pair at a time (all of it does not fit in memory).
    raw_stats = sum(pd.DataFrame({"rows": len(r), "nan": r.isna().sum(), "zeros": (r == 0).sum(), "duplicates": r.duplicated("open_time").sum()}) for r in map(load_symbol, SYMBOLS))
    raw_stats.to_csv(OUT / "raw_columns.csv")

    df = pd.read_parquet(PROCESSED / "features.parquet")
    train = df[df["split"] == "train"]
    labelled = train[train["label"] != -1]

    train[FEATURES + ["fwd_ret"]].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T \
        .assign(skew=train[FEATURES].skew(), kurtosis=train[FEATURES].kurt()) \
        .to_csv(OUT / "feature_stats.csv")

    corr = pd.DataFrame({
        "pearson_label": labelled[FEATURES].corrwith(labelled["label"]),
        "spearman_label": labelled[FEATURES].corrwith(labelled["label"], method="spearman"),
        "spearman_fwd_ret": train[FEATURES].corrwith(train["fwd_ret"], method="spearman"),
    }).sort_values("spearman_label", key=abs, ascending=False)
    corr.to_csv(OUT / "label_correlation.csv")

    # Up-rate by feature decile: shows non-linear relations that a correlation coefficient misses.
    decile_rates = {f: labelled.groupby(pd.qcut(labelled[f], 10, labels=False, duplicates="drop"))["label"].mean() for f in FEATURES}
    pd.DataFrame(decile_rates).T.to_csv(OUT / "up_rate_by_decile.csv")

    pd.crosstab([df["split"]], df["label"].map({1: "up", 0: "down", -1: "flat"})).to_csv(OUT / "label_balance.csv")
    pd.crosstab(df["symbol"], df["split"]).to_csv(OUT / "rows_per_symbol.csv")

    group_figures(train)

    fig, ax = plt.subplots(figsize=(11, 9))
    cm = train[FEATURES].corr(method="spearman")
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "feature_correlation.png", dpi=110)
    cm.to_csv(OUT / "feature_correlation.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    corr["spearman_label"].plot.barh(ax=ax, color=np.where(corr["spearman_label"] > 0, "#55a868", "#c44e52"))
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation with label (train)")
    fig.tight_layout()
    fig.savefig(OUT / "label_correlation.png", dpi=110)

    print(corr.round(4).to_string())


if __name__ == "__main__":
    main()
