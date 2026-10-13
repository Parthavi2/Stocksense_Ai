"""Prediction, explainability and model comparison.

Two modes, chosen automatically from how many trading days the data has:

* PANEL mode (several days): predict the forward `horizon`-day log return of each stock from
  lagged/rolling features. Strict chronological train / calibration / test split with a purge gap
  of `horizon` sessions so no target window overlaps a later segment.
* INTRADAY mode (one day): predict the open->close return of the day from information known at
  the open (the overnight gap and the price level). Stocks are split by symbol - this is a
  cross-sectional experiment, NOT a forecast of a future date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .evaluation import conformal_halfwidth, decile_spread, evaluate_models, interval_coverage
from .features import TEMPORAL_FEATURES, add_daily_features, add_temporal_features

BASE_ZERO = "Baseline: no change (0)"
BASE_MEAN = "Baseline: training mean"
INTRADAY_FEATURES = ["gap_pct", "log_prev_close"]


def make_models(seed: int = C.RANDOM_STATE) -> dict:
    """Deliberately regularised models: financial returns are very noisy, so flexible models overfit."""
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=30,
                                               n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                           min_samples_leaf=30, random_state=seed),
    }


# ---------------------------------------------------------------------------- dataset builders
def prepare_panel(df: pd.DataFrame, horizon: int = 1, top_n: int = 150) -> pd.DataFrame:
    """Top-N most liquid symbols -> temporal features + forward target, NaN rows dropped."""
    liq = df.groupby("symbol")["turnover"].median().nlargest(top_n).index
    d = add_temporal_features(df[df["symbol"].isin(liq)], horizon)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=TEMPORAL_FEATURES + ["fwd_ret"])
    d["target_pct"] = d["fwd_ret"] * 100                      # forward log return in percent
    return d.reset_index(drop=True)


def prepare_latest(df: pd.DataFrame, horizon: int = 1, top_n: int = 150) -> pd.DataFrame:
    """Feature rows of the most recent session (target unknown): the inputs for a *future* forecast."""
    liq = df.groupby("symbol")["turnover"].median().nlargest(top_n).index
    d = add_temporal_features(df[df["symbol"].isin(liq)], horizon)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=TEMPORAL_FEATURES)
    return d[d["date"] == df["date"].max()].reset_index(drop=True)


def forecast_latest(labeled: pd.DataFrame, latest: pd.DataFrame, features: list[str], model_name: str,
                    halfwidth: float, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Fit on ALL labelled history, predict the horizon-ahead return for the latest session.
    These are genuine forecasts of unobserved values; interval = split-conformal half-width."""
    y = labeled["target_pct"]
    m = clone(make_models(seed)[model_name]).fit(labeled[features], y.clip(*y.quantile([0.005, 0.995])))
    out = latest[["date", "symbol"]].copy()
    out["forecast_pct"] = m.predict(latest[features])
    out["lower_pct"], out["upper_pct"] = out["forecast_pct"] - halfwidth, out["forecast_pct"] + halfwidth
    return out.sort_values("forecast_pct", ascending=False).reset_index(drop=True)


def prepare_intraday(df: pd.DataFrame) -> pd.DataFrame:
    d = add_daily_features(df).replace([np.inf, -np.inf], np.nan)
    return d.dropna(subset=INTRADAY_FEATURES + ["intraday_pct"]).reset_index(drop=True)


def chronological_masks(dates: pd.Series, horizon: int, test_frac: float = 0.2, cal_frac: float = 0.2):
    """Boolean masks (train, calibration, test) by date, separated by `horizon`-session purge gaps."""
    u = np.sort(dates.unique())
    n = len(u)
    i_test = int(n * (1 - test_frac))
    i_cal = int(i_test * (1 - cal_frac))
    gap = max(horizon, 1)
    if i_cal - gap < 5 or i_test - gap - i_cal < 5 or n - i_test < 5:
        raise ValueError("Not enough trading days for a train/calibration/test split with this horizon.")
    tr = dates.isin(u[:i_cal - gap]); ca = dates.isin(u[i_cal:i_test - gap]); te = dates.isin(u[i_test:])
    return tr.to_numpy(), ca.to_numpy(), te.to_numpy()


def symbol_masks(symbols: pd.Series, fracs=(0.6, 0.2, 0.2), seed: int = C.RANDOM_STATE):
    """Deterministic cross-sectional train/cal/test masks by symbol (one-day data only)."""
    u = np.sort(symbols.unique())
    perm = np.random.default_rng(seed).permutation(u)
    a, b = int(len(u) * fracs[0]), int(len(u) * (fracs[0] + fracs[1]))
    sets = [set(perm[:a]), set(perm[a:b]), set(perm[b:])]
    return tuple(symbols.isin(s).to_numpy() for s in sets)


# -------------------------------------------------------------------------- fit & evaluate
def fit_and_evaluate(X: pd.DataFrame, y: pd.Series, masks, alpha: float = 0.1, seed: int = C.RANDOM_STATE) -> dict:
    """Fit baselines + ML models on the training mask, calibrate intervals on the calibration
    mask, evaluate on the test mask. The test set is never used for fitting or calibration."""
    tr, ca, te = masks
    Xtr, ytr, Xca, yca, Xte, yte = X[tr], y[tr], X[ca], y[ca], X[te], y[te]
    lo, hi = ytr.quantile([0.005, 0.995])                       # winsorise training target only
    ytr_w = ytr.clip(lo, hi)
    fitted, test_pred, cal_pred = {}, {BASE_ZERO: np.zeros(len(yte)), BASE_MEAN: np.full(len(yte), ytr.mean())}, {}
    cal_pred[BASE_ZERO] = np.zeros(len(yca)); cal_pred[BASE_MEAN] = np.full(len(yca), ytr.mean())
    for name, m in make_models(seed).items():
        m = clone(m).fit(Xtr, ytr_w)
        fitted[name] = m
        test_pred[name] = m.predict(Xte); cal_pred[name] = m.predict(Xca)
    metrics = evaluate_models(yte.to_numpy(), test_pred, baseline=BASE_ZERO)
    hw = {k: conformal_halfwidth(yca, cal_pred[k], alpha) for k in test_pred}
    metrics["interval_halfwidth"] = metrics["model"].map(hw)
    metrics["interval_coverage"] = [interval_coverage(yte, test_pred[m] - hw[m], test_pred[m] + hw[m]) for m in metrics["model"]]
    return {"metrics": metrics, "models": fitted, "test_pred": test_pred, "halfwidth": hw, "alpha": alpha,
            "y_test": yte, "test_index": X.index[te], "n_train": int(tr.sum()), "n_cal": int(ca.sum()), "n_test": int(te.sum())}


def cross_sectional_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = C.RANDOM_STATE) -> dict:
    """K-fold out-of-fold predictions (single-day data). Reports per-model metrics and the
    top-vs-bottom decile spread of realised returns when stocks are ranked by prediction."""
    oof = {n: np.zeros(len(y)) for n in [BASE_ZERO, BASE_MEAN, *make_models()]}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        ytr = y.iloc[tr]
        ytr_w = ytr.clip(*ytr.quantile([0.005, 0.995]))
        oof[BASE_MEAN][te] = ytr.mean()
        for name, m in make_models(seed).items():
            oof[name][te] = clone(m).fit(X.iloc[tr], ytr_w).predict(X.iloc[te])
    metrics = evaluate_models(y.to_numpy(), oof, baseline=BASE_ZERO)
    spreads = {k: decile_spread(y, v) for k, v in oof.items() if k not in (BASE_ZERO, BASE_MEAN)}
    metrics["decile_spread"] = metrics["model"].map({k: v["spread"] for k, v in spreads.items()})
    metrics["rank_corr"] = metrics["model"].map({k: v["rank_corr"] for k, v in spreads.items()})
    return {"metrics": metrics, "oof": oof, "spreads": spreads}


def walk_forward(data: pd.DataFrame, features: list[str], horizon: int, n_splits: int = 4, target: str = "target_pct",
                 first_test_frac: float = 0.5, seed: int = C.RANDOM_STATE) -> dict:
    """Expanding-window walk-forward evaluation over several consecutive test blocks.
    Training always ends `horizon` sessions before each test block starts."""
    dates = np.sort(data["date"].unique())
    start = int(len(dates) * first_test_frac)
    edges = np.linspace(start, len(dates), n_splits + 1).astype(int)
    if edges[1] - edges[0] < 5:
        raise ValueError("Not enough history for this many walk-forward folds.")
    pooled, fold_rows = [], []
    for k in range(n_splits):
        test_d = dates[edges[k]:edges[k + 1]]
        train_d = dates[:max(edges[k] - horizon, 0)]
        tr, te = data["date"].isin(train_d), data["date"].isin(test_d)
        if tr.sum() < 200 or te.sum() < 20:
            continue
        ytr = data.loc[tr, target]; ytr_w = ytr.clip(*ytr.quantile([0.005, 0.995]))
        preds = {BASE_ZERO: np.zeros(te.sum()), BASE_MEAN: np.full(te.sum(), ytr.mean())}
        for name, m in make_models(seed).items():
            preds[name] = clone(m).fit(data.loc[tr, features], ytr_w).predict(data.loc[te, features])
        yte = data.loc[te, target].to_numpy()
        met = evaluate_models(yte, preds, BASE_ZERO); met["fold"] = k + 1
        met["test_start"] = pd.Timestamp(test_d[0]); met["test_end"] = pd.Timestamp(test_d[-1])
        fold_rows.append(met)
        block = data.loc[te, ["date", "symbol"]].copy(); block["actual"] = np.expm1(yte / 100); block["fold"] = k + 1
        for n, p in preds.items():
            block[n] = p
        pooled.append(block)
    if not fold_rows:
        raise ValueError("No usable walk-forward folds.")
    return {"folds": pd.concat(fold_rows, ignore_index=True), "predictions": pd.concat(pooled, ignore_index=True)}


# ------------------------------------------------------------------------------ explainability
def importance_table(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance on held-out data: how much RMSE rises when a feature is shuffled."""
    r = permutation_importance(model, X, y, scoring="neg_root_mean_squared_error", n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame({"feature": X.columns, "rmse_increase": r.importances_mean, "std": r.importances_std}
                        ).sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def partial_dependence_1d(model, X: pd.DataFrame, feature: str, n_grid: int = 30) -> pd.DataFrame:
    lo, hi = X[feature].quantile([0.02, 0.98])
    grid = np.linspace(lo, hi, n_grid)
    out = []
    for v in grid:
        Xv = X.copy(); Xv[feature] = v
        out.append(model.predict(Xv).mean())
    return pd.DataFrame({feature: grid, "avg_prediction": out})


def local_explanation(model, X_ref: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Occlusion explanation for one prediction: replace each feature by its training median and
    see how the prediction changes. contribution = pred(row) - pred(row with feature at median)."""
    med = X_ref.median()
    full = float(model.predict(row.to_frame().T.astype(float))[0])
    base = float(model.predict(med.to_frame().T.astype(float))[0])
    rows = []
    for f in X_ref.columns:
        r2 = row.copy(); r2[f] = med[f]
        rows.append((f, row[f], med[f], full - float(model.predict(r2.to_frame().T.astype(float))[0])))
    out = pd.DataFrame(rows, columns=["feature", "value", "typical_value", "contribution"])
    out.attrs.update(prediction=full, baseline_prediction=base)
    return out.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)


def what_if(model, X_ref: pd.DataFrame, base_row: pd.Series, overrides: dict, halfwidth: float) -> dict:
    """Predict for an edited feature row; returns base and scenario predictions with intervals."""
    scen = base_row.copy()
    for k, v in overrides.items():
        scen[k] = v
    pb = float(model.predict(base_row.to_frame().T.astype(float))[0])
    ps = float(model.predict(scen.to_frame().T.astype(float))[0])
    return {"base": pb, "scenario": ps, "change": ps - pb, "interval": (ps - halfwidth, ps + halfwidth)}
