# StockSense AI

An interactive stock-market analytics app built around **frequent itemset mining** (Apriori and FP-Growth) and extended with
trend, seasonality, cycle, residual, regime, anomaly, clustering, prediction, explainability, what-if and backtesting tools.
Frontend: Streamlit + Plotly. Analytics: Pandas, NumPy, Scikit-learn, Statsmodels, MLxtend.

## IMPORTANT: what your dataset can and cannot support

The app inspects the file before doing anything. For `nse_2000000.csv` the inspection found:

| Fact | Value |
|---|---|
| Rows / columns | 2,000,000 / 34 |
| Distinct trading dates | **1** (`2026-10-05`) |
| Distinct stocks (ISIN / ticker) | 2,678 |
| Exact duplicate rows | 1,997,322 (**99.87%**): the same 2,678 records repeated about 747 times |
| Segment / series / instrument type | all `CM` / `EQ` / `STK` |
| Entirely empty columns | 12 (expiry, strike, option type, open interest, `Rmks`, `Rsvd1-4`, ...) |
| Invalid OHLC rows | 0 (one stock moved more than 30% and is flagged, not dropped) |

So the file is **one trading session of 2,678 stocks**, not a history. Anything that needs a time axis (trend over time, weekday/monthly
seasonality, cycles, multi-day forecasting, walk-forward backtests, statistical regimes, corporate-action detection) **cannot be computed
honestly from it**, and the app does not fake it. Instead:

* Features that work on a cross-section run on your real data: pattern mining, breadth/risk, anomalies, clustering/similarity,
  intraday prediction, explainability, what-if, model comparison.
* Time-series features are **fully implemented and unit-tested** but locked behind minimum-history rules (`stocksense/config.py::MIN_DAYS`).
  Put more daily NSE files (same columns) in a folder, enter the folder path in the sidebar, and they unlock automatically.
  Single-day substitutes are shown where a valid analogue exists (see table).

## The 12 features and how each behaves on your single-day file

| # | Feature | Single day (your file) | 250+ / 500+ days |
|---|---|---|---|
| 1 | Frequent market pattern mining | Full: Apriori, FP-Growth, from-scratch Apriori; support/confidence/lift sliders; rule validation on **held-out stocks** (labelled as cross-sectional, not time-based) | Same, validated on **later dates**; optional "previous-session" items |
| 2 | Price / return prediction | Intraday mode: open-to-close return of the observed day predicted from gap and price level; tested on unseen stocks. **Not a forecast of a future date** | Forward 1/5/10/21-session returns, chronological split with purge gap, real future forecasts with intervals |
| 3 | Actual trend analysis | Up / down / sideways *state of the session* per stock | Rolling-regression trend with t-statistic; segments table |
| 4 | Seasonal trends | Locked (explains requirement) | Weekday and monthly stats + Kruskal-Wallis test, STL |
| 5 | Cyclical variation | Locked | HP filter + periodogram, peaks/troughs, only cycles that repeat >= 3 times |
| 6 | Random / residual variation | OLS of returns on gap, price, liquidity, range; unexplained share and flagged residuals | STL residuals, Ljung-Box tests, flagged days |
| 7 | Market regime detection | Rule-based breadth regime (risk-on / mixed / risk-off) | Gaussian-mixture regimes fitted on the first 70% of days |
| 8 | Anomaly detection | Isolation Forest across stocks with a "main driver" | Plus per-stock historical anomaly markers |
| 9 | Similarity / clustering | K-Means (silhouette-guided), PCA map, nearest neighbours | Plus return-correlation heat-map |
| 10 | What-if simulation | Edit a stock's gap/price; shift all gaps; see estimate + interval | Plus market-shock x beta scenario |
| 11 | Explainable ML | Permutation importance, partial dependence, per-prediction occlusion, Ridge coefficients | Same |
| 12 | Backtesting & model comparison | 5-fold out-of-fold comparison + top-vs-bottom decile ranking test (not a trading backtest) | Walk-forward folds, long-only strategy with costs vs benchmark |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp /path/to/nse_2000000.csv data/nse_2000000.csv        # or type any path in the sidebar
streamlit run app.py
python -m pytest tests -q                               # 64 tests
```

First load parses the 340 MB CSV in chunks (~10 s) and stores the cleaned 2,678 rows as parquet in `.cache/`; later loads take
well under a second. Python 3.10+ is required.

## Pages

1. **Market Overview**: data profile, quality report, filters, breadth, gainers/losers, treemap, risk proxies, stock comparison, downloads.
2. **Pattern Mining Lab**: sidebar controls for algorithm, support, confidence, lift, itemset length, item groups. Tabs: itemsets, rules,
   validation on unseen data, three-way algorithm comparison. Runs can be saved to a local SQLite database.
3. **Trend & Variation Studio**: trend / seasonality / cycles / residuals (or their single-day substitutes).
4. **Prediction Lab**: evaluation vs baselines, explainability, backtest/model comparison, what-if.
5. **Market Intelligence**: regimes, anomalies, clustering and similarity.

## Architecture

```
app.py                      Streamlit entry: sidebar, data loading, page routing, error guard
pages_ui/                   One module per page + common.py (context, cached helpers)
stocksense/
  config.py                 column mapping, thresholds, MIN_DAYS capability rules
  preprocessing.py          chunked load, dedupe, validation, quality report, corporate actions, adjusted close
  features.py               daily & temporal features, ItemBinner (train-fitted bins), time/symbol splits
  mining.py                 Apriori / FP-Growth (MLxtend), from-scratch Apriori, rules, rule evaluation
  timeseries.py             trend, seasonality, STL, HP cycles, residual tests, regimes, risk, betas
  prediction.py             models, splits, conformal intervals, walk-forward, explainability, what-if
  evaluation.py             metrics, conformal width, decile spread, strategy backtest
  market_intel.py           anomalies, clustering, similarity, breadth regime, shock impact
  visualization.py          Plotly builders (no analysis inside)
  storage.py                parquet cache + SQLite result store
tests/                      64 pytest tests; synthetic.py is TEST-ONLY data with known structure
```

## Methods

**Preprocessing.** Read in 250k-row chunks, dropping exact duplicates per chunk (memory stays small). Keep series `EQ` by default. Drop rows with
unparseable dates, non-positive/missing prices, high < low, negative volume; resolve duplicate date/symbol keys (keep last). Flag, but keep,
close outside [low, high], zero volume and |return| > 30%. Returns use `close / prev_close`; NSE adjusts `prev_close` on ex-dates, so splits and
bonuses do not create fake crashes. With 2+ days, `detect_corporate_actions` compares today's `prev_close` with yesterday's `close` to list adjustments.

**Frequent patterns.** Each stock-session is a transaction. Features are discretised into items (e.g. `Return=Strong Up`, `Gap=Gap Down`,
`Range=High`, `ClosePos=Near High`, `Liquidity=Low`). Fixed thresholds are used for returns/gaps; terciles for relative levels are
**fitted on the training slice only** and re-applied to unseen data. Support = P(A and B); confidence = P(B | A); lift = confidence / P(B).
Rules are filtered by confidence and lift, then re-measured on unseen data; a rule "holds" if its test confidence stays above the threshold and
lift stays above 1. Three implementations are cross-checked in tests and in the UI.

**Trend.** Rolling OLS of log price on time; Uptrend/Downtrend require both a minimum slope and |t| >= 2, otherwise Sideways.
**Seasonality.** Mean return with 95% CI per weekday/month plus Kruskal-Wallis p-value. **Cycles.** HP-filter cycle component, periodogram peaks,
reporting only periods that fit at least three times. **Residuals.** STL residual, robust z-score flags, Ljung-Box on residuals and squared
residuals (volatility clustering). **Regimes.** Gaussian mixture on rolling mean/volatility, fitted on the first 70% of days.

**Prediction.** Ridge, Random Forest and Gradient Boosting (deliberately regularised) vs two baselines (zero return, training mean).
Chronological train / calibration / test with a purge gap of `horizon` sessions; training targets winsorised; test data never used for fitting
or calibration (a unit test poisons the test targets to prove it). Intervals use split-conformal prediction calibrated on the calibration slice.

**Evaluation.** MAE, RMSE, directional accuracy, out-of-sample R2 relative to the baseline, % RMSE skill vs baseline, interval coverage,
decile spread and rank correlation, walk-forward folds, long-only backtest with costs (Sharpe, drawdown, hit rate).

## Limitations

* The provided file is a single day, so time-series results require additional data; none are simulated.
* Single-day relationships (including the intraday model) reflect that session's conditions and may not hold on other days.
* Same-session pattern rules describe what a day looks like; they are co-occurrences, not causes or trading signals.
* Returns are mostly noise; the app reports when models fail to beat the baseline. Backtests ignore slippage, taxes, liquidity limits and survivorship bias.
* Cycle and seasonality findings are candidates; with many months/periods some look significant by chance.
* Corporate-action detection relies on consecutive trading sessions being present; gaps in a symbol's history can hide or fake adjustments.
* Walk-forward backtests on large universes are slow on a single CPU; results are cached per setting.
* Not financial advice.
