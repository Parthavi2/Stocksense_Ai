import numpy as np
import pandas as pd
import pytest

from stocksense import market_intel as MI
from stocksense.features import add_daily_features
from tests.synthetic import make_panel


def _snapshot(n=300):
    return add_daily_features(make_panel(n_symbols=n, n_days=1, seed=3))


def test_anomaly_detector_flags_planted_outlier():
    d = _snapshot()
    d.loc[0, ["high", "close"]] = d.loc[0, "prev_close"] * 1.9
    d.loc[0, "turnover"] *= 500
    d = add_daily_features(d.drop(columns=[c for c in d.columns if c.endswith("_pct") or c.startswith("log_")]))
    out = MI.detect_anomalies(d, contamination=0.02)
    assert out.loc[out.symbol == d.loc[0, "symbol"], "is_anomaly"].iloc[0]
    assert abs(out["is_anomaly"].mean() - 0.02) < 0.015
    assert out.loc[out.symbol == d.loc[0, "symbol"], "main_driver"].iloc[0] in MI.ANOMALY_FEATURES


def test_clusters_recover_two_planted_groups_and_choose_k():
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(0, 0.3, (100, len(MI.CLUSTER_FEATURES))), columns=MI.CLUSTER_FEATURES)
    b = pd.DataFrame(rng.normal(6, 0.3, (100, len(MI.CLUSTER_FEATURES))), columns=MI.CLUSTER_FEATURES)
    d = pd.concat([a, b], ignore_index=True); d["symbol"] = [f"S{i}" for i in range(200)]; d["name"] = d["symbol"]
    sc = MI.choose_k(d)
    assert int(sc.loc[sc.silhouette.idxmax(), "k"]) == 2
    cl, prof = MI.cluster_stocks(d, 2)
    purity = max((cl.cluster[:100] == "0").mean(), (cl.cluster[:100] == "1").mean())
    assert purity > 0.98 and prof["n_stocks"].sum() == 200


def test_nearest_neighbour_is_a_near_duplicate():
    d = _snapshot(100)
    twin = d.iloc[[5]].copy(); twin["symbol"] = "TWIN"; twin["name"] = "twin"
    twin["ret_pct"] += 1e-6
    d = pd.concat([d, twin], ignore_index=True)
    assert MI.similar_stocks(d, d.loc[5, "symbol"], n=3).iloc[0]["symbol"] == "TWIN"
    with pytest.raises(KeyError):
        MI.similar_stocks(d, "NOPE")


def test_breadth_regime_rules():
    base = pd.DataFrame({"prev_close": [100.0] * 10, "turnover": [1.0] * 10})
    up = base.assign(close=[102.0] * 8 + [99.0] * 2); dn = base.assign(close=[98.0] * 8 + [101.0] * 2)
    mixed = base.assign(close=[101.0] * 5 + [99.0] * 5)
    assert "Risk-on" in MI.breadth_regime(up)["regime"] and "Risk-off" in MI.breadth_regime(dn)["regime"]
    assert "Mixed" in MI.breadth_regime(mixed)["regime"]


def test_time_anomaly_detection_flags_planted_spike():
    df = make_panel(n_symbols=2, n_days=200)
    i = df.index[(df.symbol == "SYM00")][100]
    df.loc[i, "close"] = df.loc[i, "prev_close"] * 1.4; df.loc[i, "high"] = df.loc[i, "close"]
    out = MI.stock_time_anomalies(df, "SYM00", 5)
    assert out.loc[df.loc[i, "date"], "is_anomaly"]
    with pytest.raises(ValueError):
        MI.stock_time_anomalies(make_panel(n_symbols=1, n_days=10), "SYM00")


def test_shock_impact_is_linear_in_beta():
    b = pd.DataFrame({"symbol": ["A", "B"], "beta": [0.5, 2.0], "n_obs": [100, 100]})
    out = MI.shock_impact(b, -10).set_index("symbol")
    assert out.loc["A", "expected_move_pct"] == -5 and out.loc["B", "expected_move_pct"] == -20
