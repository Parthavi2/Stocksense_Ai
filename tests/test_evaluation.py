import numpy as np
import pandas as pd
import pytest

from stocksense import evaluation as E


def test_basic_metrics():
    y, p = np.array([1.0, -2.0, 3.0]), np.array([0.0, -1.0, 5.0])
    assert E.mae(y, p) == pytest.approx(4 / 3) and E.rmse(y, p) == pytest.approx(np.sqrt(2))
    assert E.directional_accuracy(y, p) == pytest.approx(2 / 3)


def test_oos_r2_relative_to_baseline():
    y = np.array([1.0, 2.0, 3.0])
    assert E.oos_r2(y, y, np.zeros(3)) == pytest.approx(1.0)
    assert E.oos_r2(y, np.zeros(3), np.zeros(3)) == pytest.approx(0.0)
    assert E.oos_r2(y, -y, np.zeros(3)) < 0


def test_evaluate_models_ranks_by_rmse_and_skill():
    y = np.array([1.0, -1.0, 2.0, -2.0])
    t = E.evaluate_models(y, {"base": np.zeros(4), "good": y * 0.9}, "base")
    assert t.iloc[0]["model"] == "good" and t.iloc[0]["skill_vs_baseline_pct"] == pytest.approx(90.0)


def test_conformal_coverage_is_close_to_target():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 1, 2000); te = rng.normal(0, 1, 5000)
    hw = E.conformal_halfwidth(cal, np.zeros_like(cal), 0.1)
    assert abs(E.interval_coverage(te, -hw, hw) - 0.9) < 0.02


def test_decile_spread_detects_signal_and_noise():
    rng = np.random.default_rng(1); x = rng.normal(size=2000)
    sig = E.decile_spread(x + rng.normal(0, 0.5, 2000), x)
    noise = E.decile_spread(rng.normal(size=2000), x)
    assert sig["spread"] > 1 and abs(noise["spread"]) < 0.5 and sig["rank_corr"] > 0.7


def test_performance_stats():
    r = pd.Series([0.1, -0.1, 0.1])
    s = E.performance_stats(r)
    assert s["total_return_pct"] == pytest.approx((1.1 * 0.9 * 1.1 - 1) * 100)
    assert s["max_drawdown_pct"] == pytest.approx(-10.0)


def test_long_only_backtest_picks_top_predictions_and_charges_costs():
    d = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 10), "symbol": list("abcdefghij"),
                      "pred": np.arange(10.0), "actual": np.arange(10.0) / 100})
    bt = E.long_only_backtest(d, 1, top_q=0.2, cost_bps=10)
    assert bt["strategy"].iloc[0] == pytest.approx(0.085 - 0.002) and bt["benchmark"].iloc[0] == pytest.approx(0.045)
