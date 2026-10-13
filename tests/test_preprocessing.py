import pandas as pd
import pytest

from stocksense import preprocessing as pp
from stocksense.config import REQUIRED_RAW
from tests.synthetic import make_panel, to_nse_csv


def _row(**kw):
    base = dict(TradDt="2026-01-05", TckrSymb="AAA", SctySrs="EQ", OpnPric=100, HghPric=110, LwPric=95, ClsPric=105,
                PrvsClsgPric=100, TtlTradgVol=1000, TtlTrfVal=100000, TtlNbOfTxsExctd=50)
    base.update(kw)
    return base


def test_duplicates_and_invalid_rows_are_counted(tmp_path):
    rows = [_row(), _row(), _row(),                                  # 3 identical copies
            _row(TckrSymb="BAD1", HghPric=90, LwPric=95),            # high < low
            _row(TckrSymb="BAD2", ClsPric=-1),                       # negative price
            _row(TckrSymb="BAD3", TtlTradgVol=-5),                   # negative volume
            _row(TckrSymb="BE", SctySrs="BE"),                       # other series
            _row(TckrSymb="OK2", ClsPric=120)]
    f = tmp_path / "x.csv"; pd.DataFrame(rows).to_csv(f, index=False)
    df, rep = pp.load_and_clean(f)
    assert rep.rows_raw == 8 and rep.exact_duplicates == 2
    assert rep.high_below_low == 1 and rep.non_positive_prices == 1 and rep.negative_volume == 1
    assert rep.other_series_removed == 1
    assert sorted(df["symbol"]) == ["AAA", "OK2"] and rep.n_dates == 1


def test_missing_required_columns_raise(tmp_path):
    f = tmp_path / "x.csv"; pd.DataFrame(_row(), index=[0]).drop(columns=["ClsPric"]).to_csv(f, index=False)
    with pytest.raises(pp.DataError, match="ClsPric"):
        pp.load_and_clean(f)


def test_missing_path_raises():
    with pytest.raises(pp.DataError):
        pp.load_and_clean("/does/not/exist.csv")


def test_folder_of_files_is_concatenated(tmp_path):
    df = make_panel(n_symbols=3, n_days=10)
    d = sorted(df.date.unique())
    to_nse_csv(df[df.date.isin(d[:5])], tmp_path / "a.csv"); to_nse_csv(df[df.date.isin(d[5:])], tmp_path / "b.csv")
    out, rep = pp.load_and_clean(tmp_path)
    assert rep.n_dates == 10 and len(out) == 30


def test_corporate_action_detected_and_returns_unaffected():
    df = make_panel(n_symbols=3, n_days=400, split_symbol="SYM01")
    ca = pp.detect_corporate_actions(df)
    assert len(ca) == 1 and ca.iloc[0]["symbol"] == "SYM01"
    assert ca.iloc[0]["adjustment_ratio"] == pytest.approx(0.5, abs=1e-6)
    ret = df[df.symbol == "SYM01"].eval("close/prev_close-1")
    assert ret.abs().max() < 0.2                       # no -50% fake crash in exchange-adjusted returns


def test_adjusted_close_has_no_jump_at_split():
    df = make_panel(n_symbols=2, n_days=400, split_symbol="SYM01")
    adj = pp.add_adjusted_close(df)
    s = adj[adj.symbol == "SYM01"]["adj_close"]
    assert s.pct_change().abs().max() < 0.2


def test_single_day_has_no_corporate_actions():
    assert pp.detect_corporate_actions(make_panel(n_symbols=3, n_days=1)).empty
