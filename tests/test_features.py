import numpy as np
import pandas as pd
import pytest

from stocksense import features as F
from tests.synthetic import make_panel


def _one():
    return pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "symbol": ["A"], "open": [102.0], "high": [110.0], "low": [100.0],
                         "close": [108.0], "prev_close": [100.0], "volume": [1000], "turnover": [105000.0], "trades": [10]})


def test_daily_feature_formulas():
    d = F.add_daily_features(_one()).iloc[0]
    assert d.ret_pct == pytest.approx(8.0) and d.gap_pct == pytest.approx(2.0)
    assert d.intraday_pct == pytest.approx((108 / 102 - 1) * 100)
    assert d.range_pct == pytest.approx(10.0) and d.close_pos == pytest.approx(0.8)
    assert d.vwap == pytest.approx(105.0) and d.avg_trade_value == pytest.approx(10500.0)


def test_flat_bar_close_position_is_neutral():
    x = _one(); x[["open", "high", "low", "close"]] = 100.0
    assert F.add_daily_features(x).iloc[0].close_pos == 0.5


def test_forward_target_matches_manual_and_uses_only_future():
    df = make_panel(n_symbols=2, n_days=60)
    d = F.add_temporal_features(df, horizon=3)
    s = d[d.symbol == "SYM00"].reset_index(drop=True)
    lr = np.log(s.close / s.prev_close)
    assert s.loc[10, "fwd_ret"] == pytest.approx(lr[11:14].sum())
    assert s["fwd_ret"].iloc[-3:].isna().all()          # last rows have no future -> NaN, never filled
    assert s.loc[10, "lr_lag1"] == pytest.approx(lr[9])  # features look backwards only


def test_binner_cuts_fit_on_train_only():
    df = make_panel(n_symbols=20, n_days=1)
    train, test = F.holdout_split_by_symbol(df, 0.3)
    b = F.ItemBinner().fit(train)
    cuts = dict(b.cuts_)
    b.transform(test)
    assert b.cuts_ == cuts                                # transform must not change fitted cut-points


def test_items_in_a_group_are_mutually_exclusive_and_exhaustive():
    df = make_panel(n_symbols=40, n_days=1)
    b = F.ItemBinner().fit(df)
    oh = b.transform(df)
    for g in F.ALL_GROUPS:
        cols = [c for c in oh.columns if c.startswith(g + "=")]
        assert (oh[cols].sum(axis=1) == 1).all(), g


def test_lag_items_drop_first_day_and_use_previous_session():
    df = make_panel(n_symbols=3, n_days=5)
    d = F.add_daily_features(df).sort_values(["symbol", "date"])
    b = F.ItemBinner().fit(d)
    oh = b.transform(d, ["Return"], lag=True)
    assert len(oh) == 3 * 4
    cur = b.transform(d, ["Return"])
    i = oh.index[1]                                       # a row with a previous day
    prev_label = [c for c in cur.columns if cur.loc[i - 1, c]][0].split("=")[1]
    assert oh.loc[i, f"Prev Return={prev_label}"]


def test_chronological_split_is_ordered():
    df = make_panel(n_symbols=3, n_days=50)
    tr, te = F.chronological_split(df, 0.7)
    assert tr.date.max() < te.date.min()


def test_holdout_split_disjoint_and_deterministic():
    df = make_panel(n_symbols=30, n_days=1)
    a1, b1 = F.holdout_split_by_symbol(df); a2, b2 = F.holdout_split_by_symbol(df)
    assert set(a1.symbol).isdisjoint(b1.symbol) and set(b1.symbol) == set(b2.symbol)
