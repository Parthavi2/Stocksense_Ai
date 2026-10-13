import numpy as np
import pandas as pd
import pytest

from stocksense import timeseries as T
from tests.synthetic import make_panel


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_trend_labels_on_constructed_series():
    n = 150; rng = np.random.default_rng(0)
    up = pd.Series(100 * np.exp(0.004 * np.arange(n) + rng.normal(0, 0.002, n)), index=_idx(n))
    down = pd.Series(100 * np.exp(-0.004 * np.arange(n) + rng.normal(0, 0.002, n)), index=_idx(n))
    flat = pd.Series(100 * np.exp(rng.normal(0, 0.002, n)), index=_idx(n))
    assert T.classify_trend(up, 30)["trend"].dropna().eq("Uptrend").mean() > 0.95
    assert T.classify_trend(down, 30)["trend"].dropna().eq("Downtrend").mean() > 0.95
    assert T.classify_trend(flat, 30)["trend"].dropna().eq("Sideways").mean() > 0.85


def test_trend_segments_cover_labelled_days():
    s = pd.Series(100 * np.exp(0.003 * np.arange(120)), index=_idx(120))
    tr = T.classify_trend(s, 20)
    seg = T.trend_segments(tr)
    assert seg["days"].sum() == tr["trend"].notna().sum()


def test_trend_needs_enough_data():
    with pytest.raises(T.InsufficientHistory):
        T.classify_trend(pd.Series([1.0, 2.0, 3.0], index=_idx(3)), 20)


def test_weekday_effect_is_recovered_and_null_is_not_flagged():
    effect = T.market_series(make_panel(n_symbols=10, n_days=900, weekday_effect=0.01))
    tab, p = T.seasonal_stats(effect["ret"], "weekday")
    assert p < 0.001 and tab.sort_values("mean_pct").iloc[-1]["label"] == "Mon"
    null = T.market_series(make_panel(n_symbols=10, n_days=900, seed=5))
    assert T.seasonal_stats(null["ret"], "weekday")[1] > 0.01


def test_seasonality_refuses_short_history():
    with pytest.raises(T.InsufficientHistory):
        T.seasonal_stats(pd.Series(np.zeros(30), index=_idx(30)), "month")


def test_cycle_detection_finds_planted_period():
    n = 900; t = np.arange(n); rng = np.random.default_rng(3)
    lvl = pd.Series(100 * np.exp(0.0002 * t + 0.08 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.003, n)), index=_idx(n))
    res = T.detect_cycles(lvl, lamb=1e7)
    assert abs(res["table"]["period_days"].iloc[0] - 100) < 12


def test_cycle_detection_needs_history():
    with pytest.raises(T.InsufficientHistory):
        T.detect_cycles(pd.Series(np.linspace(1, 2, 100), index=_idx(100)))


def test_residual_analysis_flags_planted_shock():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 1, 300), index=_idx(300)); r.iloc[150] = 15
    out = T.residual_analysis(r, 4.0)
    assert r.index[150] in out["flagged"].index and len(out["flagged"]) <= 3
    assert out["ljung_box_p"] > 0.01                       # white noise has no autocorrelation


def test_stl_shapes():
    s = T.market_series(make_panel(n_symbols=5, n_days=300))["level"]
    stl = T.stl_decompose(s, 21)
    assert np.allclose(stl["trend"] + stl["seasonal"] + stl["resid"], stl["observed"])


def test_regimes_separate_calm_and_volatile_periods():
    ms = T.market_series(make_panel(n_symbols=20, n_days=700))
    rg = T.detect_regimes(ms["ret"], 2, 20)
    hi_vol = rg["summary"]["avg_volatility_pct"].max()
    assert hi_vol > 1.5 * rg["summary"]["avg_volatility_pct"].min()
    assert rg["train_end"] < ms.index[-1]                  # model fitted on early part only
    assert np.allclose(rg["transitions"].sum(axis=1), 1)


def test_risk_metrics_known_values():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0])
    m = T.risk_metrics(r)
    lvl = (1 + r).cumprod()
    assert m["max_drawdown_pct"] == pytest.approx(((lvl / lvl.cummax()).min() - 1) * 100)
    assert m["ann_vol_pct"] == pytest.approx(r.std() * np.sqrt(252) * 100)
    assert m["var95_pct"] > 0


def test_betas_recover_construction():
    df = make_panel(n_symbols=30, n_days=400)
    b = T.estimate_betas(df)
    assert 0.5 < b["beta"].mean() < 1.5 and b["beta"].between(0, 3).all()
