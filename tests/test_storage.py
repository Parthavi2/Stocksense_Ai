import pandas as pd

from stocksense import storage as S
from tests.synthetic import make_panel, to_nse_csv


def test_parquet_cache_roundtrip(tmp_path):
    f = tmp_path / "d.csv"; to_nse_csv(make_panel(n_symbols=3, n_days=20), f)
    cache = tmp_path / "cache"
    a, rep_a, hit_a = S.load_with_cache(f, cache_dir=cache)
    b, rep_b, hit_b = S.load_with_cache(f, cache_dir=cache)
    assert not hit_a and hit_b
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
    assert rep_a.rows_clean == rep_b.rows_clean == 60


def test_result_store_roundtrip(tmp_path):
    st = S.ResultStore(tmp_path / "r.db")
    t = pd.DataFrame({"rule": ["A -> B"], "lift": [1.5]})
    rid = st.save("association_rules", {"min_support": 0.05}, t)
    runs = st.list_runs("association_rules")
    assert runs.iloc[0]["id"] == rid and "0.05" in runs.iloc[0]["params"]
    pd.testing.assert_frame_equal(st.load(rid), t)
