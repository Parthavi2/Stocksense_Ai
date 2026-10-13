import numpy as np
import pandas as pd
import pytest

from stocksense import prediction as P
from stocksense.features import TEMPORAL_FEATURES
from tests.synthetic import make_panel


def test_chronological_masks_are_ordered_disjoint_and_purged():
    dates = pd.Series(np.repeat(pd.bdate_range("2020-01-01", periods=300), 4))
    tr, ca, te = P.chronological_masks(dates, horizon=5)
    assert not (tr & ca).any() and not (ca & te).any() and not (tr & te).any()
    assert dates[tr].max() < dates[ca].min() < dates[ca].max() < dates[te].min()
    gap = len(pd.bdate_range(dates[tr].max(), dates[ca].min())) - 1
    assert gap >= 5                                       # purge gap >= horizon


def test_masks_reject_tiny_history():
    with pytest.raises(ValueError):
        P.chronological_masks(pd.Series(pd.bdate_range("2020-01-01", periods=15)), 5)


def test_symbol_masks_partition_symbols():
    s = pd.Series([f"S{i}" for i in range(100)])
    tr, ca, te = P.symbol_masks(s)
    assert (tr.astype(int) + ca + te == 1).all() and 50 < tr.sum() < 70


def _signal_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(0, 0.5, n))
    return X, y


def test_models_beat_baseline_when_signal_exists_and_intervals_cover():
    X, y = _signal_data()
    masks = (np.arange(4000) < 2400, (np.arange(4000) >= 2400) & (np.arange(4000) < 3200), np.arange(4000) >= 3200)
    res = P.fit_and_evaluate(X, y, masks, alpha=0.1)
    m = res["metrics"].set_index("model")
    assert m.loc["Ridge", "skill_vs_baseline_pct"] > 40 and m.loc["Ridge", "R2_oos_vs_baseline"] > 0.7
    assert abs(m.loc["Ridge", "interval_coverage"] - 0.9) < 0.05


def test_no_skill_claimed_on_pure_noise():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(3000, 3)), columns=["a", "b", "c"]); y = pd.Series(rng.normal(size=3000))
    masks = (np.arange(3000) < 1800, (np.arange(3000) >= 1800) & (np.arange(3000) < 2400), np.arange(3000) >= 2400)
    m = P.fit_and_evaluate(X, y, masks)["metrics"].set_index("model")
    assert m.loc["Ridge", "skill_vs_baseline_pct"] < 2 and m["R2_oos_vs_baseline"].drop("Baseline: no change (0)").max() < 0.02


def test_test_rows_never_used_for_training():
    X, y = _signal_data(600)
    tr = np.arange(600) < 300; ca = (np.arange(600) >= 300) & (np.arange(600) < 450); te = np.arange(600) >= 450
    y2 = y.copy(); y2[te] = 1e6                             # poison test targets: model must be unaffected
    a = P.fit_and_evaluate(X, y, (tr, ca, te)); b = P.fit_and_evaluate(X, y2, (tr, ca, te))
    assert np.allclose(a["test_pred"]["Ridge"], b["test_pred"]["Ridge"])
    assert a["halfwidth"]["Ridge"] == b["halfwidth"]["Ridge"]


def test_panel_pipeline_end_to_end_and_walk_forward_order():
    df = make_panel(n_symbols=15, n_days=400)
    data = P.prepare_panel(df, horizon=5, top_n=15)
    assert data[TEMPORAL_FEATURES + ["target_pct"]].notna().all().all()
    wf = P.walk_forward(data, TEMPORAL_FEATURES, 5, n_splits=3)
    starts = wf["folds"].groupby("fold")["test_start"].first()
    assert starts.is_monotonic_increasing
    train_end_ok = wf["predictions"].groupby("fold")["date"].min().is_monotonic_increasing
    assert train_end_ok and set(wf["folds"]["model"]) >= {"Ridge", "Random Forest", "Gradient Boosting"}


def test_forecast_latest_is_for_the_unobserved_future():
    df = make_panel(n_symbols=10, n_days=300)
    data = P.prepare_panel(df, 5, 10); latest = P.prepare_latest(df, 5, 10)
    assert latest["date"].nunique() == 1 and latest["date"].iloc[0] == df["date"].max()
    assert data["date"].max() < latest["date"].iloc[0]       # labelled history ends before the forecast origin
    fc = P.forecast_latest(data, latest, TEMPORAL_FEATURES, "Ridge", 1.0)
    assert (fc["upper_pct"] - fc["lower_pct"]).round(6).eq(2.0).all()


def test_intraday_pipeline_on_one_day():
    df = make_panel(n_symbols=300, n_days=1)
    d = P.prepare_intraday(df)
    masks = P.symbol_masks(d["symbol"])
    res = P.fit_and_evaluate(d[P.INTRADAY_FEATURES], d["intraday_pct"], masks)
    assert res["n_train"] + res["n_cal"] + res["n_test"] == len(d)
    cv = P.cross_sectional_cv(d[P.INTRADAY_FEATURES], d["intraday_pct"], n_splits=3)
    assert len(cv["metrics"]) == 5


def test_explainability_finds_the_true_driver():
    X, y = _signal_data()
    model = P.make_models()["Ridge"].fit(X[:3000], y[:3000])
    imp = P.importance_table(model, X[3000:], y[3000:], n_repeats=3)
    assert imp.iloc[0]["feature"] == "a" and imp.set_index("feature").loc["c", "rmse_increase"] < 0.05
    ex = P.local_explanation(model, X[:3000], X.iloc[3500])
    assert ex.iloc[0]["feature"] in ("a", "b")
    assert ex["contribution"].sum() == pytest.approx(ex.attrs["prediction"] - ex.attrs["baseline_prediction"], abs=0.3)
    pdp = P.partial_dependence_1d(model, X, "a")
    assert pdp["avg_prediction"].is_monotonic_increasing


def test_what_if_moves_prediction_in_modelled_direction():
    X, y = _signal_data()
    model = P.make_models()["Ridge"].fit(X, y)
    r = P.what_if(model, X, X.iloc[0], {"a": X.iloc[0]["a"] + 1}, 0.5)
    assert r["change"] == pytest.approx(2.0, abs=0.15) and r["interval"][1] - r["interval"][0] == pytest.approx(1.0)
