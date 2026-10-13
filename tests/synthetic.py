"""TEST-ONLY synthetic market generator.

The app never uses this. It exists so unit tests can exercise the multi-day code paths
(trend, seasonality, cycles, regimes, walk-forward, ...) which cannot run on a one-day file.
The generator has *known* structure so tests can check the algorithms recover it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(n_symbols: int = 25, n_days: int = 700, seed: int = 0, cycle_period: int = 120,
               weekday_effect: float = 0.0, split_symbol: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    t = np.arange(n_days)
    # market factor with a slow sine cycle, a calm->volatile regime shift and a weekday effect
    vol = np.where(t < n_days * 0.6, 0.006, 0.015)
    mkt = 0.0003 + 0.002 * np.sin(2 * np.pi * t / cycle_period) + rng.normal(0, vol)
    mkt = mkt + weekday_effect * (dates.dayofweek == 0)
    rows = []
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        beta = rng.uniform(0.5, 1.5)
        r = beta * mkt + rng.normal(0, 0.01, n_days)
        cont = 100 * (1 + i) * np.cumprod(1 + r)
        close = cont.copy()
        if split_symbol == sym:                      # 2:1 split on day 300: raw price halves
            close[300:] = cont[300:] / 2
        prev = np.r_[cont[0] / (1 + r[0]), close[:-1]]
        if split_symbol == sym:
            prev[300] = close[299] / 2               # exchange-adjusted previous close
        gap = rng.normal(0, 0.004, n_days)
        open_ = prev * (1 + gap)
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = rng.integers(50_000, 5_000_000, n_days) * (1 + i / 10)
        rows.append(pd.DataFrame({"date": dates, "symbol": sym, "name": f"Synthetic {i}", "isin": f"INE{i:09d}",
                                  "series": "EQ", "open": open_, "high": hi, "low": lo, "close": close,
                                  "prev_close": prev, "last": close, "settle": close, "volume": volume.astype(int),
                                  "turnover": volume * close, "trades": (volume / 80).astype(int) + 1}))
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def to_nse_csv(df: pd.DataFrame, path):
    """Write the panel in the same raw NSE schema as the user's file (for end-to-end loader tests)."""
    inv = {"date": "TradDt", "symbol": "TckrSymb", "name": "FinInstrmNm", "isin": "ISIN", "series": "SctySrs",
           "open": "OpnPric", "high": "HghPric", "low": "LwPric", "close": "ClsPric", "last": "LastPric",
           "prev_close": "PrvsClsgPric", "settle": "SttlmPric", "volume": "TtlTradgVol",
           "turnover": "TtlTrfVal", "trades": "TtlNbOfTxsExctd"}
    out = df.rename(columns=inv)[list(inv.values())].copy()
    out["TradDt"] = pd.to_datetime(out["TradDt"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
