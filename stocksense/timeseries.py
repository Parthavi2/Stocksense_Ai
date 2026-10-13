"""Time-series analysis: trend, seasonality, cycles, residual variation, regimes, risk.

All functions need several trading days of history. They raise `InsufficientHistory` rather
than returning misleading results when the data is too short (see config.MIN_DAYS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

from . import config as C


class InsufficientHistory(Exception):
    pass


def require_days(n_days: int, feature: str):
    need = C.MIN_DAYS[feature]
    if n_days < need:
        raise InsufficientHistory(f"'{feature}' needs at least {need} trading days; the data has {n_days}.")


# ----------------------------------------------------------------------------- series builders
def market_series(df: pd.DataFrame, weighting: str = "equal") -> pd.DataFrame:
    """Daily market return (equal- or turnover-weighted across stocks) and a compounded index
    (base 100). Uses exchange-adjusted close/prev_close so splits do not create fake returns."""
    d = df[["date", "close", "prev_close", "turnover"]].copy()
    d["r"] = d["close"] / d["prev_close"] - 1
    d = d[np.isfinite(d["r"])]
    if weighting == "turnover":
        d["w"] = d["turnover"]
        agg = d.groupby("date").apply(lambda x: np.average(x["r"], weights=x["w"]) if x["w"].sum() > 0 else np.nan,
                                      include_groups=False)
    else:
        agg = d.groupby("date")["r"].mean()
    out = pd.DataFrame({"ret": agg}).dropna().sort_index()
    out["level"] = 100 * (1 + out["ret"]).cumprod()
    return out


def stock_series(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    d = df[df["symbol"] == symbol].sort_values("date").set_index("date")
    out = pd.DataFrame({"ret": d["close"] / d["prev_close"] - 1})
    out["level"] = d["close"].iloc[0] * (1 + out["ret"]).cumprod() / (1 + out["ret"].iloc[0])
    return out.dropna()


# ----------------------------------------------------------------------------------- trend
def classify_trend(level: pd.Series, window: int = 20, min_slope_pct: float = 0.05,
                   t_threshold: float = 2.0) -> pd.DataFrame:
    """Rolling OLS of log-price on time. Label:
        Uptrend   : slope >=  min_slope_pct per day AND t-stat >=  t_threshold
        Downtrend : slope <= -min_slope_pct per day AND t-stat <= -t_threshold
        Sideways  : otherwise
    The t-statistic guards against calling noise a trend."""
    y = np.log(level.to_numpy(dtype=float))
    n = len(y)
    if n < window + 2:
        raise InsufficientHistory(f"Need more than {window + 2} observations for a {window}-day trend.")
    x = np.arange(window, dtype=float)
    xm, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    slope = np.full(n, np.nan); tstat = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = y[i - window + 1:i + 1]
        b = ((x - xm) * (w - w.mean())).sum() / sxx
        resid = w - (w.mean() + b * (x - xm))
        se = np.sqrt((resid ** 2).sum() / (window - 2) / sxx)
        slope[i], tstat[i] = b, (b / se if se > 0 else 0.0)
    out = pd.DataFrame({"level": level.to_numpy(), "slope_pct_day": slope * 100, "t_stat": tstat}, index=level.index)
    lab = np.where((out.slope_pct_day >= min_slope_pct) & (out.t_stat >= t_threshold), "Uptrend",
          np.where((out.slope_pct_day <= -min_slope_pct) & (out.t_stat <= -t_threshold), "Downtrend", "Sideways"))
    out["trend"] = np.where(out.slope_pct_day.isna(), None, lab)
    out["sma_fast"] = level.rolling(max(window // 2, 2)).mean()
    out["sma_slow"] = level.rolling(window * 2).mean()
    return out


def trend_segments(trend: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive identical labels into segments with start/end/return."""
    t = trend.dropna(subset=["trend"])
    if t.empty:
        return pd.DataFrame(columns=["trend", "start", "end", "days", "return_pct"])
    grp = (t["trend"] != t["trend"].shift()).cumsum()
    rows = []
    for _, s in t.groupby(grp):
        rows.append((s["trend"].iloc[0], s.index[0], s.index[-1], len(s), (s["level"].iloc[-1] / s["level"].iloc[0] - 1) * 100))
    return pd.DataFrame(rows, columns=["trend", "start", "end", "days", "return_pct"])


# ------------------------------------------------------------------------------- seasonality
def seasonal_stats(ret: pd.Series, by: str = "weekday") -> tuple[pd.DataFrame, float]:
    """Mean return per weekday or month with 95% CI, plus a Kruskal-Wallis p-value for
    'do the groups differ at all?'. With 12 or 5 groups some will look 'significant' by chance."""
    require_days(len(ret), "weekday_seasonality" if by == "weekday" else "monthly_seasonality")
    r = (ret * 100).dropna()
    key = r.index.dayofweek if by == "weekday" else r.index.month
    names = (["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if by == "weekday" else
             ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    g = r.groupby(key)
    t = pd.DataFrame({"mean_pct": g.mean(), "median_pct": g.median(), "std_pct": g.std(),
                      "n": g.size(), "pct_positive": g.apply(lambda s: (s > 0).mean() * 100)})
    t["ci95"] = 1.96 * t["std_pct"] / np.sqrt(t["n"])
    t["label"] = [names[i - (0 if by == "weekday" else 1)] for i in t.index]
    groups = [s.to_numpy() for _, s in g if len(s) > 2]
    p = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan
    return t.reset_index(drop=True), p


def stl_decompose(level: pd.Series, period: int = 21) -> pd.DataFrame:
    """STL on log price: trend + seasonal(period) + residual."""
    if len(level) < 2 * period + 2:
        raise InsufficientHistory(f"STL with period {period} needs at least {2 * period + 2} observations.")
    res = STL(np.log(level), period=period, robust=True).fit()
    return pd.DataFrame({"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid})


# ----------------------------------------------------------------------------------- cycles
def detect_cycles(level: pd.Series, lamb: float | None = None, top_k: int = 3,
                  min_period: int = 20) -> dict:
    """Hodrick-Prescott cycle component + periodogram peaks.
    A period is only reported if at least three full cycles fit in the sample."""
    require_days(len(level), "cycles")
    lp = np.log(level)
    lamb = lamb or 1600 * (252 / 4) ** 4          # Ravn-Uhlig scaling for daily data
    cycle, trend = hpfilter(lp, lamb=lamb)
    x = signal.detrend(cycle.to_numpy())
    freqs, power = signal.periodogram(x, window="hann")
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1 / freqs, np.inf)
    max_period = len(x) / 3
    mask = (periods >= min_period) & (periods <= max_period)
    peaks, _ = signal.find_peaks(np.where(mask, power, 0))
    cand = sorted(peaks, key=lambda i: power[i], reverse=True)[:top_k]
    med = np.median(power[mask]) if mask.any() else np.nan
    table = pd.DataFrame({"period_days": [periods[i] for i in cand],
                          "relative_power": [power[i] / med if med > 0 else np.nan for i in cand],
                          "share_of_cycle_variance_pct": [100 * power[i] / power[mask].sum() for i in cand]})
    pk, _ = signal.find_peaks(cycle.to_numpy(), prominence=cycle.std() * 0.5)
    tr, _ = signal.find_peaks(-cycle.to_numpy(), prominence=cycle.std() * 0.5)
    return {"cycle": pd.Series(cycle, index=level.index), "trend": pd.Series(np.exp(trend), index=level.index),
            "table": table, "peaks": level.index[pk], "troughs": level.index[tr],
            "freqs": freqs[mask], "power": power[mask], "periods": periods[mask]}


# ---------------------------------------------------------------------------------- residuals
def residual_analysis(resid: pd.Series, z_threshold: float = 3.0) -> dict:
    """Robust z-scores of residuals, Ljung-Box tests for leftover structure.
    Large unexplained moves are *flagged*, not forecast."""
    require_days(len(resid), "residuals")
    r = resid.dropna()
    mad = np.median(np.abs(r - r.median())) * 1.4826
    z = (r - r.median()) / (mad if mad > 0 else r.std())
    lb = acorr_ljungbox(r, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    return {"z": z, "flagged": z[np.abs(z) > z_threshold], "ljung_box_p": float(lb),
            "ljung_box_sq_p": float(lb2), "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "normality_p": float(stats.normaltest(r).pvalue) if len(r) >= 20 else np.nan}


# ------------------------------------------------------------------------------------ regimes
def detect_regimes(ret: pd.Series, n_regimes: int = 3, window: int = 20, train_frac: float = 0.7,
                   seed: int = C.RANDOM_STATE) -> dict:
    """Gaussian-mixture regimes on [rolling mean return, rolling volatility].
    The mixture and scaler are fitted on the first `train_frac` of the sample only and then
    applied to all days, so later labels are genuinely out-of-sample."""
    require_days(len(ret), "regimes")
    f = pd.DataFrame({"mean": ret.rolling(window).mean(), "vol": ret.rolling(window).std()}).dropna()
    n_train = int(len(f) * train_frac)
    sc = StandardScaler().fit(f.iloc[:n_train])
    gm = GaussianMixture(n_components=n_regimes, covariance_type="full", n_init=5, random_state=seed)
    gm.fit(sc.transform(f.iloc[:n_train]))
    raw = gm.predict(sc.transform(f))
    # name regimes by their fitted mean return / volatility (not hardcoded to cluster ids)
    centers = pd.DataFrame(sc.inverse_transform(gm.means_), columns=["mean", "vol"])
    order = centers["mean"].rank(method="first").astype(int) - 1
    med_vol = centers["vol"].median()
    names = {}
    for k in range(n_regimes):
        if n_regimes == 2:
            base = "Bullish" if order[k] == 1 else "Bearish"
        elif n_regimes == 3:
            base = ["Bearish", "Neutral", "Bullish"][order[k]]
        else:
            base = f"Regime {order[k] + 1}"
        names[k] = f"{base} ({'high' if centers['vol'][k] > med_vol else 'low'} vol)"
    labels = pd.Series([names[k] for k in raw], index=f.index, name="regime")
    summary = pd.DataFrame({"regime": [names[k] for k in range(n_regimes)],
                            "avg_daily_return_pct": centers["mean"] * 100, "avg_volatility_pct": centers["vol"] * 100,
                            "share_of_days_pct": [(raw == k).mean() * 100 for k in range(n_regimes)]}).sort_values("avg_daily_return_pct")
    trans = pd.crosstab(labels.shift(1), labels, normalize="index").round(3)
    return {"labels": labels, "summary": summary.reset_index(drop=True), "transitions": trans,
            "train_end": f.index[n_train - 1]}


# ---------------------------------------------------------------------------------------- risk
def risk_metrics(ret: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> dict:
    """Standard risk statistics from a daily simple-return series."""
    r = ret.dropna()
    if len(r) < 2:
        raise InsufficientHistory("Risk metrics need at least two daily returns.")
    level = (1 + r).cumprod()
    dd = level / level.cummax() - 1
    ann_vol = r.std() * np.sqrt(trading_days)
    ann_ret = level.iloc[-1] ** (trading_days / len(r)) - 1
    var95 = -np.percentile(r, 5)
    return {"ann_return_pct": ann_ret * 100, "ann_vol_pct": ann_vol * 100,
            "sharpe": (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown_pct": dd.min() * 100, "var95_pct": var95 * 100,
            "cvar95_pct": -r[r <= np.percentile(r, 5)].mean() * 100, "n_days": len(r)}


def estimate_betas(df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """OLS beta of each stock's daily return on the equal-weighted market return."""
    mk = market_series(df)["ret"].rename("mkt")
    d = df[["date", "symbol", "close", "prev_close"]].copy()
    d["r"] = d["close"] / d["prev_close"] - 1
    d = d.join(mk, on="date").dropna(subset=["r", "mkt"])
    def _beta(g):
        if len(g) < min_obs or g["mkt"].var() == 0:
            return pd.Series({"beta": np.nan, "n_obs": len(g)})
        return pd.Series({"beta": g["r"].cov(g["mkt"]) / g["mkt"].var(), "n_obs": len(g)})
    return d.groupby("symbol")[["r", "mkt"]].apply(_beta).dropna().reset_index()
