"""Builds the Phase IV design document from the Phase I docx template: python report/make_phase4.py"""
import copy
import shutil

import docx
import matplotlib
import numpy as np
import pandas as pd
from docx.shared import Inches, Pt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

FIG = "out/report/"
RES = "out/results/"

# ---------------------------------------------------------------------------- template (same as make_docx.py)
d = docx.Document("report/ML - Project Phase I.docx")
P = d.paragraphs
TITLE, HEAD, BODY, ABS, REF = P[0], P[10], P[11], P[7], P[40]

r = TITLE.runs
r[0].text = "Final Model and Utility Application Design"
for x in r[1:]:
    x.text = ""

templates = {k: copy.deepcopy(v._p) for k, v in dict(head=HEAD, body=BODY, abs=ABS, ref=REF).items()}
for p in P[7:]:
    p._p.getparent().remove(p._p)
anchor = P[6]._p


def place(el):
    global anchor
    anchor.addnext(el)
    anchor = el


def para(kind, *parts):
    """parts: str for plain text, ('bold text',) tuple for bold."""
    el = copy.deepcopy(templates[kind])
    p = docx.text.paragraph.Paragraph(el, d._body)
    rpr = copy.deepcopy(p.runs[0]._r.rPr)
    for child in list(el):
        if not child.tag.endswith("}pPr"):  # drop runs and hyperlinks copied from the template
            el.remove(child)
    for part in parts:
        bold = isinstance(part, tuple)
        run = p.add_run(part[0] if bold else part)
        run._r.insert(0, copy.deepcopy(rpr))
        run.bold = True if bold else None
    place(el)
    return p


def figure(img, caption, width=3.2):
    p = para("body")
    p.alignment = 1
    p.add_run().add_picture(FIG + img, width=Inches(width))
    para("body", (caption.split(":")[0] + ":",), caption.split(":", 1)[1]).alignment = 1


def table(caption, rows):
    para("body", (caption.split(":")[0] + ":",), caption.split(":", 1)[1]).alignment = 1
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = d.styles["Table Grid"]  # bordered
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j).paragraphs[0]
            cell.alignment = 0  # left: the template justifies text, which stretches short cell lines
            run = cell.add_run(str(val))
            run.font.size, run.font.name, run.bold = Pt(9), "Times New Roman", i == 0
    place(t._tbl)
    para("body")


# ---------------------------------------------------------------------------- numbers from the data and Phase III
f = pd.read_parquet("out/processed/features.parquet", columns=["label", "split", "fwd_ret"])
share = f.groupby("split").label.value_counts(normalize=True).unstack()
means = f[f.split == "train"].groupby("label").fwd_ret.mean()  # -1 flat, 0 down, 1 up
UP, DOWN, FLAT, FEE = means[1], means[0], means[-1], 0.002
gap = (FEE - FLAT) / UP  # P(up) - P(down) a Buy needs, taking |down mean| ~ up mean
seeds = pd.read_csv(RES + "top3_seeds.csv")
rf_seed = seeds[(seeds.model == "Random Forest") & (seeds.seed != "search")].log_loss
act = pd.read_csv(RES + "mlp_activation.csv")
mlp_best = act.groupby(["activation", "hidden_layer_sizes", "alpha"]).val_log_loss.agg(["mean", "std"]).sort_values("mean").iloc[0]
comp = pd.read_csv(RES + "comparison.csv").set_index("model")
by_setting = act.groupby(["activation", "hidden_layer_sizes", "alpha"]).agg(mean=("val_log_loss", "mean"), std=("val_log_loss", "std"), epochs=("n_iter", "mean")).reset_index()
best_act = by_setting.sort_values("mean").groupby("activation").head(1).sort_values("mean")
test = pd.read_csv(RES + "test.csv").iloc[0]
imp = pd.read_csv(RES + "feature_importance.csv")
weak = list(imp[imp.importance <= 0].feature)

# ---------------------------------------------------------------------------- figures (Mermaid-like style)
BOX, EDGE, AREA, AREA_EDGE, INK = "#ECECFF", "#9370DB", "#FFFFDE", "#AAAA33", "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 100)
    ax.set_ylim(0, h * 100)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fill=BOX, edge=EDGE, size=9, bold=False, round_=True):
    style = "round,pad=0,rounding_size=8" if round_ else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, fc=fill, ec=edge, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def area(ax, x0, y0, w, h, label):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="square,pad=0", fc=AREA, ec=AREA_EDGE, lw=1, zorder=0))
    ax.text(x0 + w / 2, y0 + h - 14, label, ha="center", va="center", fontsize=9, color=INK)


def arrow(ax, a, b, label=None, rad=0.0, lpos=0.5, loff=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        x = a[0] + (b[0] - a[0]) * lpos + loff[0]
        y = a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=INK, bbox=dict(fc="#E8E8E8", ec="none", pad=1.5))


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)], closed=True, fc=BOX, ec=EDGE, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9, color=INK, linespacing=1.3)


def save(fig, name):
    fig.savefig(FIG + name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Model decision flow
fig, ax = canvas(7.8, 5.4)
box(ax, 430, 505, 440, 46, "36 inputs for the latest closed 15-minute bar\n(26 features + pair one-hot)")
box(ax, 430, 420, 320, 40, "Clip and standardize (scaler.json)")
box(ax, 430, 335, 440, 46, "Random Forest, 3 classes\n(log-loss criterion, weighted samples)", bold=True)
box(ax, 430, 250, 320, 40, "Probability calibration")
box(ax, 430, 165, 440, 46, "Expected net return of Buy and of Sell\n(class mean returns, 0.2% fee)")
box(ax, 190, 55, 150, 40, "Buy", bold=True)
box(ax, 430, 55, 150, 40, "Wait", bold=True)
box(ax, 670, 55, 150, 40, "Sell", bold=True)
arrow(ax, (430, 482), (430, 440))
arrow(ax, (430, 400), (430, 358))
arrow(ax, (430, 312), (430, 270), "P(up), P(flat), P(down)", lpos=0.5, loff=(110, 0))
arrow(ax, (430, 230), (430, 188))
arrow(ax, (330, 142), (210, 75), "Buy margin\nexceeded", lpos=0.45, loff=(-55, 8))
arrow(ax, (430, 142), (430, 75), "neither", lpos=0.5, loff=(40, 0))
arrow(ax, (530, 142), (650, 75), "Sell margin\nexceeded", lpos=0.45, loff=(55, 8))
ax.plot([115, 60, 60, 210], [55, 55, 505, 505], color=INK, lw=1.1, ls="--")
arrow(ax, (195, 505), (210, 505))
ax.text(50, 290, "after a Buy:\nre-check every\n15 minutes,\nsell at the latest\nafter 1 hour", fontsize=8, color=INK, ha="right", va="center")
ax.set_xlim(-60, 780)
save(fig, "phase4_decision.png")

# Data periods and their role
fig, ax = plt.subplots(figsize=(7.6, 1.9))
num = matplotlib.dates.date2num
periods = [("Train", "2020-01-01", "2024-10-01"), ("A", "2024-10-01", "2025-03-01"), ("B", "2025-03-01", "2025-07-01"), ("Test", "2025-07-01", "2026-11-01")]
for name, a, b in periods:
    x0, x1 = pd.Timestamp(a), pd.Timestamp(b)
    ax.barh(0, (x1 - x0).days, left=num(x0), height=0.5, color=BOX if name != "Test" else "#FFF3D6", edgecolor=EDGE, lw=1.2)
    ax.text(num(x0 + (x1 - x0) / 2), 0.0, name, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
ax.text(num(pd.Timestamp("2022-05-15")), -0.42, "fit the model", ha="center", va="top", fontsize=8, color=INK)
ax.text(num(pd.Timestamp("2025-03-01")), -0.42, "val-A: calibrate, tune margins", ha="right", va="top", fontsize=8, color=INK)
ax.text(num(pd.Timestamp("2025-03-01")), -0.72, "val-B: confirm", ha="left", va="top", fontsize=8, color=INK)
ax.text(num(pd.Timestamp("2026-03-01")), -0.42, "evaluate once", ha="center", va="top", fontsize=8, color=INK)
ax.xaxis_date()
ax.set_ylim(-1.05, 0.4)
ax.set_yticks([])
for side in ["left", "right", "top"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="x", labelsize=8, colors="#555555")
fig.tight_layout()
save(fig, "phase4_periods.png")

# Application architecture (redrawn from the application design, model response updated)
fig, ax = canvas(9.0, 7.4)
box(ax, 560, 700, 100, 34, "User")
area(ax, 420, 545, 280, 125, "Client tier")
box(ax, 560, 590, 250, 62, "Web App (React SPA)\npair selector, chart,\naction card, history")
area(ax, 300, 40, 590, 450, "Application tier")
box(ax, 560, 410, 270, 50, "App API (FastAPI)\nREST endpoints, validation")
box(ax, 765, 285, 230, 50, "Feature Pipeline\nshared with training code")
box(ax, 540, 285, 200, 50, "Outcome Resolver\nscheduled job")
box(ax, 430, 120, 210, 50, "SQLite\npredictions, outcomes")
box(ax, 770, 120, 210, 50, "Binance Client\nretry, rate limit, cache")
area(ax, 0, 180, 270, 310, "Model tier (another host)")
box(ax, 135, 390, 230, 56, "Model Service (FastAPI)\nPOST /v1/predict")
box(ax, 135, 255, 250, 84, "Model artifact\nmodel.pkl, scaler.json,\nschema.json, decision.json")
box(ax, 770, -40, 250, 40, "Binance public REST API")
arrow(ax, (560, 683), (560, 621))
arrow(ax, (560, 559), (560, 435), "HTTPS / JSON", lpos=0.4)
arrow(ax, (425, 420), (250, 400), "features", lpos=0.5, loff=(0, 16))
arrow(ax, (250, 375), (425, 398), "3 probabilities + action", lpos=0.5, loff=(5, -30))
arrow(ax, (135, 362), (135, 297))
arrow(ax, (660, 385), (750, 310))
arrow(ax, (520, 260), (450, 145))
arrow(ax, (470, 385), (410, 145))
arrow(ax, (600, 260), (720, 145))
arrow(ax, (770, 260), (770, 145))
arrow(ax, (770, 95), (770, -20), "HTTPS", lpos=0.5)
ax.set_xlim(-10, 900)
ax.set_ylim(-70, 730)
save(fig, "phase4_architecture.png")

# Feature pipeline (corrected fetch sizes and model output)
fig, ax = canvas(7.6, 8.0)
box(ax, 380, 770, 150, 36, "Request: pair")
box(ax, 200, 680, 310, 50, "Fetch 1-minute klines, pair\n7 days + margin (~11 calls)")
box(ax, 560, 680, 310, 50, "Fetch 1-minute klines, BTCUSDT\n7 days + margin (~11 calls)")
box(ax, 380, 590, 260, 36, "Drop the still-open bar")
diamond(ax, 380, 490, 260, 110, "Data OK?\nno long gaps, fresh,\n7 days of history")
box(ax, 130, 385, 230, 36, "Return 503 with reason")
box(ax, 520, 385, 420, 50, "clean_symbol + add_features\n(the same code that built the training set)")
box(ax, 520, 295, 420, 50, "26 features + pair one-hot\nordered by schema.json")
box(ax, 520, 195, 420, 64, "Model Service: clip and standardize\n(scaler.json), Random Forest\n+ calibration")
box(ax, 520, 85, 420, 64, "decision.json: expected net return\n-> P(up), P(flat), P(down)\nand Buy / Sell / Wait")
arrow(ax, (340, 760), (230, 705))
arrow(ax, (420, 760), (530, 705))
arrow(ax, (200, 655), (340, 608))
arrow(ax, (560, 655), (420, 608))
arrow(ax, (380, 572), (380, 545))
arrow(ax, (305, 455), (160, 403), "no", lpos=0.5, loff=(-10, 12))
arrow(ax, (455, 455), (500, 410), "yes", lpos=0.5, loff=(20, 6))
arrow(ax, (520, 360), (520, 320))
arrow(ax, (520, 270), (520, 227))
arrow(ax, (520, 163), (520, 117))
ax.set_ylim(40, 800)
save(fig, "phase4_pipeline.png")

# Interface states and deployment are unchanged from the application design (extracted from its PDF)
for src, dst in [("doc_fig_states.png", "phase4_states.png"), ("doc_fig_deployment.png", "phase4_deployment.png")]:
    shutil.copy(FIG + src, FIG + dst)

# ---------------------------------------------------------------------------- text
para("abs", f"This document designs the final model and the utility application of our cryptocurrency predictor. Based on the Phase III pilot study, we keep the random forest: it tied the best model on validation, gave the same result with every random seed, trains in under half a minute on 1.1 million rows and can explain its predictions. The main design change is the objective. Instead of separating upward from downward moves, the model predicts whether the price will rise above, stay within or fall below the 0.2% trading fee over the next hour, and a fee-aware rule turns these three probabilities into one of three actions for the user: Buy, Sell or Wait. We describe the training loss, the sample weighting, the feature selection, probability calibration and the decision rule, and an evaluation protocol based on the net return a user would make after fees. The application is a web client, an application server and a separate model service that communicate over HTTP, so the model can run on another machine.")
para("head", ("KEYWORDS",))
para("abs", "Model design, Random forest, Cost-sensitive learning, Probability calibration, Client/server architecture")

para("head", ("1.  Introduction",))
para("body", f"The goal of the project is an application that tells a user what to do with a cryptocurrency right now. Phase II built the dataset: 15-minute bars for 10 large-cap USDT pairs from the official Binance data [1], with 26 engineered numeric features and a one-hot encoding of the pair. Phase III compared eleven supervised models on predicting whether the price rises or falls by more than 0.2% over the next hour. All of them reached between 51.6% and 53.0% validation accuracy; we selected the random forest, which reached {100 * test['accuracy']:.1f}% accuracy and an AUC of {test['roc_auc']:.3f} on the test period.")
para("body", "Phase III answered how well a model separates large upward moves from large downward ones. A user needs a different answer: whether acting now will make money after the trading fee. Section 2 designs the model around that question, and Section 3 designs the application that serves it.")

para("head", ("2.  Model Design",))
para("body", ("2.1  Choice of Algorithm",), " Table 1 compares the strongest models of Phase III on the criteria that matter for the final model. The validation log loss is averaged over several random seeds, since Phase III showed that a single seed can make a model look better than it is.")
table("Table 1: Candidates from the Phase III pilot study (validation set; log loss averaged over seeds).", [
    ["Model", "Log loss", "Seed std", "Fit", "Predict"],
    ["Random forest", f"{rf_seed.mean():.5f}", f"{rf_seed.std():.5f}", f"{comp.loc['Random Forest', 'fit_time']:.0f} s", f"{comp.loc['Random Forest', 'predict_time']:.1f} s"],
    [f"MLP ({mlp_best.name[0]}, 16)", f"{mlp_best['mean']:.5f}", f"{mlp_best['std']:.5f}", f"{comp.loc['MLP (full data)', 'fit_time']:.0f} s", "< 0.1 s"],
    ["KNN", f"{comp.loc['KNN', 'log_loss']:.5f}", "0", "-", f"{comp.loc['KNN', 'predict_time']:.0f} s"],
    ["Logistic reg.", f"{comp.loc['Logistic Regression', 'log_loss']:.5f}", "0", f"{comp.loc['Logistic Regression', 'fit_time']:.1f} s", "< 0.1 s"],
    ["AdaBoost", f"{comp.loc['AdaBoost', 'log_loss']:.5f}", "-", f"{comp.loc['AdaBoost', 'fit_time'] / 60:.0f} min", f"{comp.loc['AdaBoost', 'predict_time']:.0f} s"],
])
para("body", ("Activation Functions",), f" Following the feedback on Phase III, we retrained the neural network with every activation function available in scikit-learn: identity, logistic, tanh and ReLU (rectified linear unit), which had been the only one used so far. Each was trained on the full training set around the best settings of the earlier rounds (16 or 64 neurons, L2 penalty 0.03 or 0.1) with three random seeds per setting, and compared by the mean over seeds (Table 2). ReLU and tanh tie at {best_act.iloc[0]['mean']:.5f}, with tanh the more stable of the two. Logistic and identity stop training after a few dozen epochs and end close to logistic regression ({comp.loc['Logistic Regression', 'log_loss']:.4f}), as expected since an identity activation makes the network linear. The regularization mattered more than the activation: for ReLU and tanh, the stronger penalty (0.1) beat the weaker one (0.03), which trained longer and overfitted. Smoother variants such as Leaky ReLU or GELU are not available in scikit-learn and mainly help deeper networks than ours, so they are left for future work.")
table("Table 2: Best setting per activation function (validation log loss, mean and std over three seeds), with the random forest for reference.", [["Activation", "Neurons, penalty", "Log loss", "Std", "Epochs"]] + [
    [row.activation, f"{row.hidden_layer_sizes[1:-2] if isinstance(row.hidden_layer_sizes, str) else row.hidden_layer_sizes[0]}, {row.alpha}", f"{row['mean']:.5f}", f"{row['std']:.5f}", f"{row.epochs:.0f}"] for _, row in best_act.iterrows()] + [
    ["Random forest", "100 trees, depth 8", f"{rf_seed.mean():.5f}", f"{rf_seed.std():.5f}", "-"]])
para("body", f"We choose the random forest [2], although it is not the model with the lowest mean loss: the best neural network setting, with any activation, is lower by at most {rf_seed.mean() - mlp_best['mean']:.5f}, a difference far smaller than the variation between the neural network's own seeds. The decision rests on four points. Nature of the data: the features are heavy-tailed returns and ratios whose effect on the label depends on each other, for example short-term reversion that is stronger in volatile periods; trees capture such interactions without assuming a functional form and are insensitive to the scale and outliers of the inputs. Size: with 1.1 million training rows, a forest of 100 trees trains in {comp.loc['Random Forest', 'fit_time']:.0f} seconds on a laptop, which makes regular retraining practical as the market changes. Stability: the forest gives the same result with every seed (standard deviation {rf_seed.std():.5f}), while the neural network varies by {mlp_best['std']:.5f} and depends on when training stops. Feasibility in the application: the forest predicts in milliseconds, needs no GPU, handles three classes natively, and its feature importances let the application explain what drives a prediction. k-nearest neighbors and the kernel SVM are too slow to predict, and AdaBoost too slow to retrain.")

para("body", ("2.2  Objective: From Direction to Action",), f" The application tells the user one of three things for the next hour: Buy (the price is expected to rise by more than the fee), Sell (for a holder, the price is expected to fall by more than the fee; for anyone else it means \"do not buy\") or Wait. We restrict the application to spot trading, buying and selling coins the user owns, since short selling requires borrowed money and exposes users to forced liquidation. The 0.2% threshold of Phase II is exactly the round-trip spot taker fee on Binance (0.1% per trade).")
para("body", f"Phase III removed every hour in which the price moved by less than 0.2%. These flat hours are {100 * share.loc['train', -1]:.1f}% of the training set and {100 * share.loc['test', -1]:.1f}% of the test period (Table 3), and they are exactly the cases in which a user who buys loses the fee. The final model therefore predicts three classes over the next hour: up (return above +0.2%), flat (within 0.2%) and down (below -0.2%). This uses all the data, including the 30% that Phase III discarded, and lets the decision rule weigh the cost of a flat outcome.")
table("Table 3: Class shares of the three-class target.", [["Split", "Up", "Flat", "Down"]] + [
    [s.capitalize(), f"{100 * share.loc[s, 1]:.1f}%", f"{100 * share.loc[s, -1]:.1f}%", f"{100 * share.loc[s, 0]:.1f}%"] for s in ["train", "val", "test"]])

para("body", ("2.3  Loss Functions",), " The design uses a loss at two levels. For training, the forest grows each tree by minimizing the multiclass cross-entropy (the log-loss split criterion of scikit-learn [4]), which rewards splits that produce confident and correct class probabilities rather than only pure leaves. This matches how the model is used: its probabilities, not its predicted class, drive the decision. For selecting settings and judging the model, the loss is the one the user experiences: the average net return per signal after fees.")
para("body", ("Why Log Loss",), " Log loss is a proper scoring rule: its expected value is lowest only when the predicted probabilities equal the true ones, so trees grown with it are pushed toward accurate probabilities, which is what the decision rule consumes. Table 4 lists the alternatives we considered and the role each keeps in the design. Gini impurity, the default, usually grows very similar trees and is slightly cheaper, so it stays as a comparison in the implementation. The Brier score is also proper but punishes confident mistakes less, which makes it a useful second check of calibration. A fee-aware asymmetric loss, in which a wrong Buy (the fee plus the drop) costs more than a missed one (only the gain not made), matches the user's situation well, but scikit-learn's forests accept only built-in split criteria, so this asymmetry enters through the decision margins and the move-size weights instead. Regressing the next-hour return directly with a squared or Huber loss would target the expected return itself, but the return is heavy-tailed and noisy and linear regression was among the weaker Phase III models, so it remains an alternative rather than the main design. Focal loss targets rare, hard classes in gradient-trained models, which does not match balanced classes and a random forest.")
table("Table 4: Loss functions considered.", [
    ["Loss", "Optimizes", "Role in the design"],
    ["Log loss (cross-entropy)", "Accurate class probabilities", "Training criterion"],
    ["Gini impurity", "Purity of the leaves", "Comparison (default)"],
    ["Brier score", "Squared probability error", "Calibration check"],
    ["Fee-aware asymmetric cost", "Wrong Buy costs more than a missed one", "Through margins and weights"],
    ["Squared / Huber on return", "Expected return directly", "Alternative, not chosen"],
    ["Net return per signal", "What the user earns", "Selection metric"],
])
para("body", f"The decision rule converts the three probabilities into an expected net return. In the training set, upward moves average {100 * UP:+.2f}%, flat ones {100 * FLAT:+.3f}% and downward ones {100 * DOWN:+.2f}%. The expected net return of buying now is P(up) x {100 * UP:.2f}% + P(flat) x {100 * FLAT:.3f}% + P(down) x ({100 * DOWN:.2f}%) - 0.2%, and selling is the mirror image. The application signals Buy or Sell only when this value exceeds a margin tuned on validation data, and Wait otherwise. Because the fee is large compared with the average move, a Buy requires P(up) to exceed P(down) by about {100 * gap:.0f} percentage points. Phase III's probabilities were mostly close to 50%, so the application is expected to say Wait most of the time, which is the honest output for a weak signal.")
figure("phase4_decision.png", "Figure 1: From the latest bar to an action. While a user holds a position, the model is re-run every 15 minutes.", width=3.3)

para("body", ("2.4  Custom Weighting",), " Two sample weightings are part of the design and will be kept only if they improve the net return on validation. First, weighting each training bar by the size of its move, since a wrong call on a 2% move costs the user far more than one on a 0.3% move. Second, a recency weight that halves every year or two, since Phase III showed that the AUC changes more between months than between models and older market regimes are less informative. The classes are close to balanced (Table 3), so no class weights are needed.")
para("body", f"Feature weighting in the usual sense does not apply to trees: a split depends only on the order of the values, so multiplying a feature by a weight changes nothing. The tree equivalent is feature selection. In Phase III the permutation importance of {', '.join(weak)} was zero or negative, so the design tests the forest without them. Removing inputs that only add noise also lowers the number of values the application must compute and check.")

para("body", ("2.5  Modifications to the Default Design",), " Hyperparameters: we keep the Phase III settings (100 trees, depth 8, the square root of the number of features at each split), which Phase III bracketed on both sides, and replace the default Gini criterion with the log-loss criterion. Calibration: the probabilities of a random forest are known to be pulled toward the middle [3]; since the decision rule multiplies them by expected returns, they are calibrated on held-out data with a sigmoid or isotonic mapping. Decision layer: the action comes from the expected net return with tuned margins, not from the class with the highest probability. Exit rule: the model is trained for a one-hour horizon, but while a user holds a coin the application re-runs it every 15 minutes and advises selling as soon as the action turns to Sell, or after one hour at the latest. Coin: the pair stays a one-hot input, following the feedback on Phase II, so a single model serves every pair while still learning coin-specific patterns.")

para("body", ("2.6  Evaluation Protocol",), " The data keeps the chronological split of Phase II, with the validation period divided in two (Figure 2). Settings, calibration and margins are fitted on the first half (val-A) and confirmed on the second (val-B), so that the margins are never judged on the data used to choose them. Every setting with randomness is averaged over several seeds. The main results are the average net return per signal and the share of signals that made money, reported at several signal frequencies, with the multiclass log loss as a check on the probabilities. The test period is evaluated once, after every choice is fixed. Phase III already evaluated binary models on this period, so we report it as such; the three-class model and its decision rule will not have been evaluated on it, and the application's own history of resolved predictions provides a truly new test.")
figure("phase4_periods.png", "Figure 2: Data periods and their role in the evaluation.", width=3.3)

para("head", ("3.  Utility Application Design",))
para("body", "The application lets a user choose a pair, fetches fresh market data from Binance, asks the model for an action and shows it with its probabilities. It takes the input a person would naturally give, the name of a pair, and pulls the market data itself. It follows a client/server architecture in which the model runs as a separate service, so it does not need to be on the same machine as the client or the application server.")
para("body", ("3.1  Requirements",), " Functionally, the user selects one of the 10 supported pairs and requests an action; the application retrieves the latest market data without manual entry; it shows the action, the three probabilities and the time at which the advice expires; it shows a candlestick chart of the pair; and it stores every prediction, records what actually happened one hour later, and reports the running results. Non-functionally, the model must be reachable only through an API, features must be computed exactly as in training, missing or stale data must produce an error rather than a prediction, the interface must state that the signal is weak and is not financial advice, the system must start with one command, and a new model must be deployable without changing the application code.")
para("body", ("3.2  Architecture",), " The system has three tiers (Figure 3). The browser talks only to the application API, which coordinates the feature pipeline, the Binance client, the database and the model service. The model service receives a feature vector and returns the three probabilities and the action; it loads a versioned artifact containing the fitted model, the scaler, the feature schema and the decision parameters, so a model can never be paired with the wrong scaler or margins.")
figure("phase4_architecture.png", "Figure 3: Application architecture with client, application and model tiers.", width=3.3)

para("body", ("3.3  Data Flow",), " A prediction request follows these steps: (1) the browser sends the chosen pair; (2) the API checks it against the 10 supported pairs; (3) the feature pipeline fetches 1-minute candles for the pair and for BTC covering the last seven days, which the 7-day return and the daily volatility ratio require; (4) it drops the still-open bar, checks for gaps and staleness, and computes the features with the same functions that built the training set (Figure 4); (5) the model service clips and standardizes them with the stored scaler, runs the model and applies the decision rule; (6) the API stores the prediction and returns it. Fetching 1-minute data rather than Binance's own 15-minute bars matters: the training bars were built from 1-minute data with minutes without trades removed, so only the same aggregation guarantees identical features. Repeated requests for the same pair within the same closed bar return the stored prediction.")
figure("phase4_pipeline.png", "Figure 4: Feature pipeline from request to action.", width=3.0)
table("Table 5: Main API endpoints (api: application server; model: model service).", [
    ["Endpoint", "Purpose"],
    ["api GET /api/pairs", "The 10 supported pairs"],
    ["api GET /api/candles", "Recent candles for the chart"],
    ["api POST /api/predict", "Action for a pair"],
    ["api GET /api/predictions", "History with outcomes"],
    ["api GET /api/stats", "Results of resolved predictions"],
    ["model POST /v1/predict", "Features in; probabilities and action out"],
    ["model GET /health", "Liveness, model and schema version"],
])
para("body", "A response to the browser contains the pair, the time of the bar used, the time at which the advice expires, the three probabilities, the action and the model version, for example {\"pair\": \"ETHUSDT\", \"p_up\": 0.38, \"p_flat\": 0.30, \"p_down\": 0.32, \"action\": \"WAIT\", \"model_version\": \"v1\"}.")
para("body", ("3.4  Data Model and Outcomes",), " Each prediction row stores the pair, the request time, the bar time, the expiry time, the three probabilities, the action, the entry price and the model and schema versions. A scheduled resolver checks predictions whose hour has passed, fetches the closing price and stores the realized return and the outcome. As in training, a realized move within 0.2% is recorded as flat. The statistics report, for the actions the application gave, the share of Buy and Sell signals that made money after fees and their average net return, with the number of resolved predictions next to every figure so that results on few predictions are not over-read.")
para("body", ("3.5  Human Interaction Flow",), " The interface has one main screen and a history tab (Figure 5). In the idle state the user picks a pair and sees its chart; pressing Predict shows a loading state and then an action card with Buy, Sell or Wait, the three probabilities, the time of the data and the time the advice expires, and a fixed note that the model's predictive power is weak and that the result is not financial advice. After a Buy, the card offers to re-check the position every 15 minutes until the action turns to Sell or the hour ends. Errors are explained in plain words with a retry button, and the history tab lists past predictions with their status: pending, correct, incorrect or flat.")
figure("phase4_states.png", "Figure 5: Interface states and the states of a stored prediction.", width=3.3)
para("body", ("3.6  Deployment",), " The system runs under Docker Compose as three containers: a web container (nginx serving the React build and proxying the API), an api container (the API, the feature pipeline and the resolver) and a model container (the inference service with its artifacts mounted read-only) (Figure 6). Only the web container is exposed. Moving the model to another machine means changing one environment variable, MODEL_URL, and deploying a new model means placing a new artifact and changing MODEL_VERSION; every stored prediction records the version that produced it. The model container pins the scikit-learn version used for training, since a saved model only loads reliably with the same version.")
figure("phase4_deployment.png", "Figure 6: Deployment with Docker Compose.", width=3.0)

para("head", ("4.  Risks and Limitations",))
para("body", "The signal is weak: Phase III's best models beat the baseline by only a few points, and after the 0.2% fee the expected net return of most hours is negative, so the application will mostly advise waiting. The market changes over time, so the model's results will drift; the application's running statistics are the early warning, and the fast training of the forest allows regular retraining. Any difference between live and training features would degrade the model silently, which the shared feature code and an equivalence test on recorded data guard against. The application depends on the public Binance API, and only the 10 trained pairs are supported. Sequence models such as LSTM networks and gradient-boosted trees are left as alternatives to compare against the random forest in the implementation phase.")

para("head", ("5.  Next Steps",))
para("body", "The implementation phase will build the three-class model with the weightings and feature selection above, calibrate it, tune the decision margins on val-A, confirm them on val-B and the exit rule against a fixed one-hour hold, evaluate the final model once on the test period, and export the artifact for the model service. In parallel, the three containers are implemented, starting with the feature pipeline and the model service.")

para("head", ("REFERENCES",))
para("ref", "[1] Binance. 2026. Binance Public Data. Retrieved October 3, 2026 from https://data.binance.vision")
para("ref", "[2] Leo Breiman. 2001. Random Forests. Machine Learning 45, 1 (2001), 5-32.")
para("ref", "[3] Alexandru Niculescu-Mizil and Rich Caruana. 2005. Predicting Good Probabilities with Supervised Learning. In Proceedings of the 22nd International Conference on Machine Learning (ICML 2005), 625-632.")
para("ref", "[4] Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 (2011), 2825-2830.")

d.save("report/ML - Project Phase IV.docx")
