"""Evaluation: forecast metrics, conformal prediction intervals, backtests, rule summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def directional_accuracy(y, p) -> float:
    """Share of non-zero actual moves whose sign was predicted correctly."""
    y, p = np.asarray(y), np.asarray(p)
    m = y != 0
    return float(np.mean(np.sign(y[m]) == np.sign(p[m]))) if m.any() else np.nan


def oos_r2(y, p, baseline) -> float:
    """Out-of-sample R^2 relative to a baseline forecast: 1 - SSE(model)/SSE(baseline).
    > 0 means the model beats the baseline; <= 0 means it does not."""
    y, p, b = map(np.asarray, (y, p, baseline))
    denom = np.sum((y - b) ** 2)
    return float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan


def evaluate_models(y_true, preds: dict, baseline: str) -> pd.DataFrame:
    """One row per model; `skill_vs_baseline_pct` = % RMSE reduction relative to `baseline`."""
    base = np.asarray(preds[baseline])
    base_rmse = rmse(y_true, base)
    rows = []
    for name, p in preds.items():
        r = rmse(y_true, p)
        rows.append({"model": name, "MAE": mae(y_true, p), "RMSE": r,
                     "R2_oos_vs_baseline": oos_r2(y_true, p, base),
                     "directional_accuracy": directional_accuracy(y_true, p),
                     "skill_vs_baseline_pct": (1 - r / base_rmse) * 100 if base_rmse > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def conformal_halfwidth(cal_y, cal_pred, alpha: float = 0.1) -> float:
    """Split-conformal half-width: the (1-alpha) quantile of absolute calibration errors
    (with the finite-sample correction). Interval = prediction +/- half-width."""
    err = np.abs(np.asarray(cal_y) - np.asarray(cal_pred))
    n = len(err)
    if n == 0:
        return np.nan
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(err, q))


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def decile_spread(y, pred, q: float = 0.1) -> dict:
    """Mean realised outcome of the top-q vs bottom-q ranked by prediction. Positive spread
    means the ranking carries signal. (Cross-sectional 'backtest' for one-day data.)"""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred)}).dropna()
    k = max(1, int(len(d) * q))
    d = d.sort_values("p")
    bottom, top = d["y"].iloc[:k].mean(), d["y"].iloc[-k:].mean()
    return {"top": float(top), "bottom": float(bottom), "spread": float(top - bottom), "n_each": k,
            "rank_corr": float(d["y"].corr(d["p"], method="spearman"))}


def performance_stats(returns: pd.Series, periods_per_year: float = 252) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"total_return_pct": np.nan, "ann_return_pct": np.nan, "ann_vol_pct": np.nan,
                "sharpe": np.nan, "max_drawdown_pct": np.nan, "hit_rate_pct": np.nan}
    eq = (1 + r).cumprod()
    ann_vol = r.std() * np.sqrt(periods_per_year)
    ann_ret = eq.iloc[-1] ** (periods_per_year / len(r)) - 1
    return {"total_return_pct": (eq.iloc[-1] - 1) * 100, "ann_return_pct": ann_ret * 100,
            "ann_vol_pct": ann_vol * 100, "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown_pct": (eq / eq.cummax() - 1).min() * 100, "hit_rate_pct": (r > 0).mean() * 100}


def long_only_backtest(pred_df: pd.DataFrame, horizon: int, top_q: float = 0.2, cost_bps: float = 10.0) -> pd.DataFrame:
    """Rebalance every `horizon` sessions: hold the top-q stocks ranked by prediction, equal weight.
    Compared with an equal-weight benchmark of all stocks. Costs are charged on every rebalance.
    pred_df columns: date, symbol, pred, actual (actual = realised forward simple return)."""
    dates = np.sort(pred_df["date"].unique())[::horizon]
    rows = []
    for d in dates:
        day = pred_df[pred_df["date"] == d]
        if len(day) < 5:
            continue
        k = max(1, int(len(day) * top_q))
        picks = day.nlargest(k, "pred")
        rows.append({"date": d, "strategy": picks["actual"].mean() - 2 * cost_bps / 1e4, "benchmark": day["actual"].mean()})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["strategy", "benchmark"])
