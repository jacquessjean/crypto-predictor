"""Builds the Phase II report from the Phase I docx template: python report/make_docx.py"""
import copy
import docx
from docx.shared import Inches, Pt

d = docx.Document("report/ML - Project Phase I.docx")
P = d.paragraphs
TITLE, HEAD, BODY, ABS, REF = P[0], P[10], P[11], P[7], P[40]
FIG = "out/report/"

r = TITLE.runs
r[0].text = "Data Preparation, Cleaning & Feature Generation"
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
            run = t.cell(i, j).paragraphs[0].add_run(str(val))
            run.font.size, run.font.name, run.bold = Pt(9), "Times New Roman", i == 0
    place(t._tbl)
    para("body")


para("abs", "The goal of this phase is to turn raw cryptocurrency market data into a clean dataset that a classification model can learn from. While validating the dataset selected in Phase I, we found that it does not match the official Binance records, so we replaced it with Binance's own public 1-minute data. We use 34.3 million 1-minute candles for 10 large-cap trading pairs between January 2020 and August 2026, aggregate them into 15-minute samples, and engineer 26 features that only use past information, including realized volatility features that can only be computed from 1-minute data. Each sample is labelled by whether the price moves up or down by more than 0.2% over the next hour, and the data is split by time with all scaling fitted on the training period only. The final dataset contains 1.58 million labelled instances. The strongest relationship we find is short-term mean reversion: the probability of an upward move drops from 57% to 47% across the deciles of the previous hour's return.")
para("head", ("KEYWORDS",))
para("abs", "Data cleaning, Feature engineering, Cryptocurrency, Time series, Data leakage")

para("head", ("1.  Dataset Description",))
para("body", ("Validating the Phase I Dataset",), " In Phase I we selected the v2-crypto-ohlcv-data dataset [2] from HuggingFace, which claims to contain Binance candlestick data. Before cleaning it, we compared one month (March 2024) of its 15-minute candles against the official Binance records for the same pairs and timestamps. The two sources did not agree: only about 4% of closing prices matched exactly, the remaining prices differed by around 0.3%, and the volumes were roughly 1,000 times smaller than the official ones. The dataset also contained a large number of zero-volume candles that do not exist in the official data, for example 1,289 of 2,976 candles for BAT, where Binance recorded trades in every one of them. We could not find any other project or paper that uses this dataset, and its description does not match its own files. Since our volume and order-flow features depend on the data being accurate, we decided to replace it.")
para("body", ("Source",), " We now use the official Binance public market data repository [1], which provides monthly files of 1-minute candlesticks for every trading pair. This is the same data returned by the live Binance API, which keeps the real-time component planned for Phase IV possible, and it is updated every month. Each instance has 11 numeric features: the open time, the opening, high, low and closing prices, the base and quote volume, the number of trades, and the taker buy base and quote volume, which record the portion of the volume initiated by buyers. The only nominal feature is the trading pair.")
para("body", ("Selection",), " We use 10 large-cap pairs against USDT that have traded continuously over the whole period: BTC, ETH, BNB, XRP, ADA, DOGE, SOL, LTC, LINK and AVAX. MATIC was excluded because it was renamed to POL in 2024. The data covers January 2020 to August 2026, or from the listing date for SOL and AVAX, which gives 34,339,076 1-minute instances and around 1.5 GB of compressed files. As described in Phase I, the label is constructed from the future price, so no target variable is included in the raw data.")

para("head", ("2.  Feature Analysis",))
para("body", ("Missing Values and Duplicates",), " The raw data has no NaN values and no duplicate timestamps. However, 292,730 minutes (0.85%) have zero volume, meaning that no trades took place and the recorded price is only a copy of the previous one. Most pairs have fewer than 0.5% of such minutes, while DOGE has the most with 5.4%. In addition, around 2,300 minutes are absent from each pair, which most likely correspond to exchange maintenance periods. Overall the official data is much cleaner than the dataset used in Phase I.")
para("body", ("Unique Values",), " The trading pair is the only nominal (text) feature, with 10 unique values. We deliberately do not one-hot encode it, and it is not used as a model input, for three reasons. First, it is a short category label rather than free text, so text embeddings such as word2vec or doc2vec do not apply, and one-hot encoding would be the only option. Second, our goal is a single model that works for any pair: with pair indicators, the model could memorize how each coin behaved in the training period instead of learning patterns that hold across the market, and it could not handle a pair it has not seen, which matters for the live predictions planned in Phase IV. Third, the properties that actually distinguish one pair from another are already captured by numeric features, namely its volatility level (vol_96, rv_60), its liquidity (idle_minutes, rel_trade_size) and how it moves relative to the market (rel_ret_4). The pair is kept in the dataset only to identify each instance.")
para("body", ("Raw Features",), " The raw price columns span very different ranges: between 2020 and 2026, BTC traded between $3,811 and $126,115, ETH between $86 and $4,955, and DOGE between $0.0011 and $0.74. The volume columns are similarly uneven, with a median quote volume per minute of $789,000 for BTC compared to $39,000 for DOGE, and a median of 978 trades per minute for BTC compared to 151 for DOGE. The taker buy columns are always a fraction of the total volume, and on average buyers initiate between 48.8% and 49.8% of the volume depending on the pair. All raw columns are strongly right-skewed and follow the long-term trend of each asset, which makes them unsuitable as direct model inputs.")
para("body", ("Distributions",), " Prices and volumes differ by several orders of magnitude between pairs, so the raw values cannot be compared across pairs. For this reason every feature we use is a ratio or a return. Table 1 summarizes the main features, and Figures 1 to 5 show the distribution of each feature group on the training set, with the most extreme 1% of values hidden for readability. The four time features are not shown, since the sine and cosine encodings of the hour and weekday are uniform by construction.")
table("Table 1: Summary statistics of the main features (training set).", [
    ["Feature", "Mean", "Std", "Median", "Skew"],
    ["returns", "0.0000", "0.0060", "0.0000", "0.90"],
    ["volatility", "0.0070", "0.0076", "0.0051", "10.30"],
    ["ret_4", "0.0001", "0.0117", "0.0000", "1.45"],
    ["ret_672", "0.0226", "0.1837", "0.0035", "7.54"],
    ["rv_60", "0.0012", "0.0011", "0.0009", "7.27"],
    ["taker_buy_ratio", "0.4945", "0.0979", "0.4952", "-0.03"],
    ["relative_volume", "1.0308", "0.7978", "0.8213", "3.82"],
    ["btc_ret_4", "0.0001", "0.0071", "0.0001", "0.13"],
])
figure("dist_candle.png", "Figure 1: Distributions of the candle shape features.")
para("body", ("Candle Shape",), " The return of a bar is centred at zero and symmetric, but has a kurtosis of 97, meaning that rare large moves are far more common than in a normal distribution. The body ratio is bounded between -1 and 1 and has two broad peaks around -0.5 and 0.5, showing that most bars move clearly in one direction, plus a sharp spike at zero from bars that close exactly where they opened. Its jagged shape comes from the minimum price step of each pair, which only allows certain ratios in quiet bars. The range and wick features are always positive and strongly right-skewed: a typical bar moves 0.5% between its high and low, but 1% of bars move more than 3.4%.")
figure("dist_momentum.png", "Figure 2: Distributions of the momentum features.")
para("body", ("Momentum",), " The returns over 1 hour, 4 hours, 1 day and 7 days are all centred near zero and become wider as the window grows, from a standard deviation of 1.2% over one hour to 18% over seven days. The longer windows are skewed to the right (skewness 7.5 for the 7-day return), which reflects the strong bull periods included in the training data, such as 2021.")
figure("dist_volatility.png", "Figure 3: Distributions of the volatility features.")
para("body", ("Volatility",), " All volatility features are positive and right-skewed, since calm periods are common and volatile periods are rare but extreme. The realized volatility over 15 minutes and over one hour have almost identical distributions, with a median of 0.09% per minute. The volatility ratio is centred around 1, as expected for a measure that compares the last hour with the last day, while the normalized volatility is centred around zero by construction.")
figure("dist_activity.png", "Figure 4: Distributions of the order flow and activity features.")
para("body", ("Order Flow and Activity",), " The taker buy ratio is centred at 0.49 and roughly normal, showing that buyers and sellers initiate trades almost equally on average, and its 4-hour version is much narrower. The relative volume and relative trade size are centred near 1, as they compare the current bar with its recent average, but the relative volume has a long right tail with some bars trading more than four times their usual volume. The idle minutes feature is zero for almost every bar, which confirms that the selected pairs trade nearly every minute, and it only becomes non-zero for a few illiquid periods.")
figure("dist_market.png", "Figure 5: Distributions of the market features.")
para("body", ("Market",), " The BTC returns are narrower than the returns of the other pairs, with a standard deviation of 0.7% over one hour compared to 1.2% for the average pair, since BTC is the least volatile of the selected assets. The relative return, the difference between the pair's return and BTC's, is centred at zero and has heavy tails, which come from moves specific to a single pair, such as news about that coin.")
para("body", ("Correlation with the Label",), " None of the raw columns relate directly to the label, since the label is a future price change. Figure 6 and Table 2 show the Spearman correlation of each engineered feature with the label on the training set. The correlations are small, which is expected in liquid markets where any strong and simple signal would be quickly traded away. However, they are consistent: every recent-return feature has a negative correlation, which indicates that prices tend to partially reverse after a strong move, and the same holds for the recent move of BTC. The taker buy ratio is also negative, meaning that periods of heavy buying tend to be followed by small declines. Table 3 shows that these effects are gradual across the deciles and are not caused by a few outliers. The realized volatility ratio shows a weaker effect in the opposite direction, with the highest decile having a 52.9% chance of an upward move. We also checked every feature for label leakage, using a threshold of 0.85: a feature this correlated with the label would most likely contain future information, and it would make a model unnecessary. The largest absolute correlation is 0.068 for ret_4, far below this threshold, so no feature was removed for leakage.")
figure("label_correlation.png", "Figure 6: Spearman correlation of each feature with the label (training set).")
table("Table 2: Features most correlated with the label.", [
    ["Feature", "Spearman", "Feature", "Spearman"],
    ["ret_4", "-0.068", "body_ratio", "-0.045"],
    ["ret_16", "-0.066", "ret_96", "-0.044"],
    ["btc_ret_4", "-0.063", "rel_ret_4", "-0.038"],
    ["btc_ret_16", "-0.062", "taker_buy_ratio", "-0.035"],
    ["returns", "-0.046", "taker_buy_ratio_16", "-0.034"],
])
table("Table 3: Probability of an upward move by feature decile.", [
    ["Feature", "D1", "D3", "D5", "D7", "D10"],
    ["ret_4", ".569", ".536", ".507", ".480", ".466"],
    ["ret_16", ".563", ".537", ".507", ".485", ".463"],
    ["btc_ret_4", ".571", ".532", ".508", ".483", ".480"],
    ["taker_buy_ratio", ".529", ".524", ".510", ".498", ".477"],
    ["rv_ratio", ".499", ".504", ".500", ".506", ".529"],
])

para("head", ("3.  Data Cleaning and Preprocessing",))
para("body", ("Zero-Volume Minutes and Gaps",), " We first replace all zero-volume minutes with missing values, since they do not represent a real traded price and would otherwise create fake zero returns. We then place each pair on a complete 1-minute grid so that the maintenance gaps become explicit missing values. Without this step, lagged features would silently compare prices that are far apart in time.")
para("body", ("Aggregation to 15 Minutes",), " Our label looks one hour ahead, so if we kept one instance per minute, consecutive instances would share 59 of their 60 future minutes. This would multiply the dataset size by 15 without adding much new information, and would make training in Phase III much slower. We therefore aggregate the 1-minute data into 15-minute bars, taking the first open, the highest high, the lowest low and the last close, and summing the volumes and trade counts. A bar with no trades at all is marked as missing.")
para("body", ("Feature Generation",), " All features are computed separately for each pair and only use information available at the end of the bar. The 1-minute data allows us to compute realized volatility, the standard deviation of 1-minute returns within the bar and over the last hour, which is a more accurate measure of risk than the high-low range of a single bar. Rolling windows require at least half of the window to be present, and lagged prices are carried forward for at most one day. The timestamp itself cannot be used directly, so we convert the hour of the day and the day of the week into sine and cosine pairs, which keeps 23:00 close to 00:00 and Sunday close to Monday. We also add features based on BTC, joined on timestamp, because most cryptocurrencies move together with the market leader.")
para("body", ("Target Construction",), " For each bar we compute the return over the next four bars (one hour). The instance is labelled 1 if the return is above 0.2% and 0 if it is below -0.2%. Moves within this range are removed, since 0.2% is roughly the cost of a round-trip trade on Binance and such moves could not be traded profitably. This follows the approach of the FI-2010 benchmark discussed in Phase I, which also separates stationary movements from real ones. If the bar four steps ahead is missing, no label is created.")
para("body", ("Removing Incomplete Rows",), " Rows with any missing feature or target are removed, which drops only 32,771 rows (1.4%), most of them at the start of each pair where the rolling windows are not yet filled. We chose not to impute these values, since an imputed price would be invented market data.")
para("body", ("Time-Based Split",), " The data is split by date using the same cutoffs for all pairs: training before October 2024, validation until July 2025, and testing after that. A random split would leak future information into training, and using different cutoffs per pair would let the model learn about market-wide events in the test period through other pairs.")
para("body", ("Outliers and Scaling",), " Each feature is clipped at its 0.1% and 99.9% quantiles and then standardized to zero mean and unit variance. Scaling is needed because the features have very different ranges: the 1-hour return has a standard deviation of about 0.01, while the relative volume reaches values above 8 and the time features lie between -1 and 1. Without it, distance-based and gradient-based models such as k-nearest neighbours, logistic regression and neural networks would be dominated by the features with the largest values. We clip before scaling because most features are heavily skewed with long tails (Table 1), and a few extreme bars, such as market crashes, would otherwise inflate the standard deviation and squeeze the remaining values close to zero. We use standardization instead of min-max scaling for the same reason, since min-max scaling depends entirely on the most extreme values. The quantiles, means and standard deviations are computed on the training set only and applied unchanged to the validation and test sets. We also wrote a test that removes future data and checks that no past feature value changes, which confirms that the features do not look ahead.")

para("head", ("4.  Final Features",))
para("body", ("Removed Features",), " None of the raw columns are used directly as model inputs. The open, high, low and close prices and the volumes are removed because their scale depends on the pair and the period, so a price of 0.5 for ADA and 60,000 for BTC cannot be compared, and they are replaced by the returns and ratios described below. The close time is redundant with the open time, and the ignore column is always zero. The open time is replaced by the cyclical time features. The trading pair is not encoded, for the reasons given in Section 2, and is kept only to identify each instance.")
para("body", ("Chosen Features",), " The final dataset has 26 features, all of which are scale-free so that a single model can be trained on every pair. Table 4 gives the definition of each feature and why it is relevant to predicting the direction of the price. The features promised in Phase I, the proportion of buyer-initiated volume and the average trade size, are included as taker_buy_ratio and rel_trade_size.")
table("Table 4: Definition and relevance of each final feature (o, h, l, c: open, high, low, close of the 15-minute bar; v: volume).", [
    ["Feature", "Definition", "Relevance"],
    ["returns", "c / o - 1", "Direction and size of the last bar"],
    ["body_ratio", "(c - o) / (h - l)", "How decisively the bar closed up or down"],
    ["volatility", "(h - l) / c", "Price range of the bar"],
    ["upper_wick", "(h - max(o, c)) / c", "Selling pressure at the highs"],
    ["lower_wick", "(min(o, c) - l) / c", "Buying support at the lows"],
    ["ret_4, ret_16, ret_96, ret_672", "Return over the last 1 hour, 4 hours, 1 day, 7 days", "Momentum or reversion at different time scales"],
    ["vol_96", "Std of returns over the last day", "Daily risk level"],
    ["normalized_volatility", "Z-score of volatility vs. the last 20 bars", "Whether the current bar is unusually wide"],
    ["rv_15", "Std of 1-minute log returns within the bar", "Precise short-term risk, only possible with 1m data"],
    ["rv_60", "Std of 1-minute log returns over the last hour", "Precise risk over the prediction horizon"],
    ["rv_ratio", "rv_60 / mean rv_60 over the last day", "Whether this hour is unusually volatile"],
    ["taker_buy_ratio", "Taker buy volume / volume", "Share of buyer-initiated trading (Phase I)"],
    ["taker_buy_ratio_16", "Same over the last 4 hours", "Persistent buying or selling pressure"],
    ["relative_volume", "v / mean v over the last 20 bars", "Unusual activity"],
    ["rel_trade_size", "Average trade size / its 1-day mean", "Large orders vs. many small ones (Phase I)"],
    ["idle_minutes", "Share of minutes in the bar with no trades", "Liquidity of the pair"],
    ["btc_ret_4, btc_ret_16", "BTC return over the last 1 and 4 hours", "Overall market trend"],
    ["rel_ret_4", "ret_4 - btc_ret_4", "Pair-specific move beyond the market"],
    ["hour_sin, hour_cos", "sin and cos of 2\u03c0 \u00b7 hour / 24", "Daily activity cycle (e.g., US and Asian sessions)"],
    ["dow_sin, dow_cos", "sin and cos of 2\u03c0 \u00b7 weekday / 7", "Weekly cycle (lower weekend activity)"],
])
figure("feature_correlation.png", "Figure 7: Spearman correlation between the features (training set).", width=6)
para("body", "Figure 7 shows the correlation between every pair of features. Some of these features are redundant: returns and body_ratio have a correlation of 0.93, the volatility features are correlated with each other between 0.7 and 0.9, and the BTC features have a correlation of 0.71 with the pair's own returns. The time features, ret_672, rel_trade_size, relative_volume and idle_minutes show almost no linear relationship with the label. We decided to keep them for now, since they may still be useful in combination with other features, and to leave feature selection to the pilot study in Phase III, where tree-based models and regularization can handle redundant features and give us feature importance scores.")

para("head", ("5.  Cleaned Dataset",))
para("body", "The 34.3 million 1-minute instances give 2.29 million 15-minute bars, and 2,257,931 remain after removing incomplete rows. Of these, 30% are removed as flat moves, leaving 1,579,191 labelled instances with 26 features, divided as shown in Table 5. The classes are balanced, so accuracy is a suitable metric. The majority class is 50.7% on the training set but only 50.2% on the test set, where downward moves are slightly more common, so a model must beat about 50.2% to show any predictive value. Every feature is standardized, so all features have a mean of zero and a standard deviation of one on the training set.")
table("Table 5: Size of each split after cleaning.", [
    ["Split", "Period", "Instances", "Up", "Down"],
    ["Train", "Jan 2020 - Sep 2024", "1,127,670", "50.7%", "49.3%"],
    ["Validation", "Oct 2024 - Jun 2025", "188,886", "50.6%", "49.4%"],
    ["Test", "Jul 2025 - Aug 2026", "262,635", "49.8%", "50.2%"],
])
para("body", "The cleaning code is available in src/cleanup.py, the feature analysis in src/analysis.py, and the cleaned dataset is saved as cleaned_dataset.csv (422 MB, 1,579,191 rows and 30 columns), containing the timestamp, the trading pair, the 26 standardized features, the label and the split each row belongs to. The same data is also saved as separate train, validation and test Parquet files for faster loading. The statistics used for scaling are saved in scaler.json so that the same transformation can be applied to live data in Phase IV.")

para("head", ("REFERENCES",))
para("ref", "[1] Binance. 2026. Binance Public Data. Retrieved October 3, 2026 from https://data.binance.vision")
para("ref", "[2] zongowo111. 2025. v2-crypto-ohlcv-data. Retrieved September 18, 2026 from https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data")

d.save("report/ML - Project Phase II.docx")
