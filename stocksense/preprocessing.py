"""Loading, validating and cleaning NSE data.

Design notes
------------
* The raw file can be very large and (as in the provided file) may contain the same
  records repeated many times. We read in chunks and drop exact duplicates per chunk so
  memory stays small, then deduplicate once more globally.
* Nothing is invented: invalid rows are dropped and *counted*; suspicious rows are only
  flagged. Everything is returned in a quality report that the UI displays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataError(Exception):
    """Raised for problems the user can fix (missing file, missing columns...)."""


@dataclass
class QualityReport:
    files: list = field(default_factory=list)
    rows_raw: int = 0
    exact_duplicates: int = 0
    key_conflicts_dropped: int = 0
    bad_dates: int = 0
    non_positive_prices: int = 0
    high_below_low: int = 0
    negative_volume: int = 0
    other_series_removed: int = 0
    rows_clean: int = 0
    flagged_close_outside_range: int = 0
    flagged_zero_volume: int = 0
    flagged_extreme_returns: int = 0
    n_symbols: int = 0
    n_dates: int = 0
    date_min: str = ""
    date_max: str = ""
    column_fill_rate: dict = field(default_factory=dict)   # raw column -> share non-null
    unused_empty_columns: list = field(default_factory=list)
    series_found: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def issues_table(self) -> pd.DataFrame:
        rows = [
            ("Raw rows read", self.rows_raw, "info"),
            ("Exact duplicate rows removed", self.exact_duplicates, "removed"),
            ("Conflicting duplicates (same date/symbol/series) removed", self.key_conflicts_dropped, "removed"),
            ("Unparseable dates removed", self.bad_dates, "removed"),
            ("Non-positive / missing prices removed", self.non_positive_prices, "removed"),
            ("High < Low removed", self.high_below_low, "removed"),
            ("Negative / missing volume removed", self.negative_volume, "removed"),
            ("Rows from other series removed", self.other_series_removed, "removed"),
            ("Close outside [Low, High] (kept, flagged)", self.flagged_close_outside_range, "flagged"),
            ("Zero-volume rows (kept, flagged)", self.flagged_zero_volume, "flagged"),
            (f"|Return| > {C.EXTREME_RETURN_PCT:.0f}% (kept, flagged)", self.flagged_extreme_returns, "flagged"),
            ("Clean rows", self.rows_clean, "info"),
        ]
        return pd.DataFrame(rows, columns=["Check", "Count", "Action"])


def list_csv_files(path) -> list[Path]:
    p = Path(path).expanduser()
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise DataError(f"No .csv files found in folder: {p}")
        return files
    if p.is_file():
        return [p]
    raise DataError(f"Data path not found: {p}")


def load_raw(path, chunksize: int = 250_000) -> tuple[pd.DataFrame, QualityReport]:
    """Read one CSV (or a folder of CSVs) in chunks; drop exact duplicates as we go."""
    files = list_csv_files(path)
    rep = QualityReport(files=[f.name for f in files])
    non_null = None
    kept = []
    for f in files:
        try:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                chunk.columns = chunk.columns.astype(str).str.strip()
                missing = [c for c in C.REQUIRED_RAW if c not in chunk.columns]
                if missing:
                    raise DataError(f"{f.name} is missing required columns: {missing}")
                rep.rows_raw += len(chunk)
                nn = chunk.notna().sum()
                non_null = nn if non_null is None else non_null.add(nn, fill_value=0)
                kept.append(chunk.drop_duplicates())
        except pd.errors.ParserError as e:
            raise DataError(f"Could not parse {f.name}: {e}") from e
    if not kept:
        raise DataError("The file(s) contain no data rows.")
    raw = pd.concat(kept, ignore_index=True).drop_duplicates().reset_index(drop=True)
    rep.exact_duplicates = rep.rows_raw - len(raw)
    fill = (non_null / max(rep.rows_raw, 1)).round(4)
    rep.column_fill_rate = {str(k): float(v) for k, v in fill.items()}
    rep.unused_empty_columns = [str(c) for c, v in fill.items() if v == 0]
    return raw, rep


def clean(raw: pd.DataFrame, rep: QualityReport, series=("EQ",)) -> tuple[pd.DataFrame, QualityReport]:
    """Validate, type-convert and filter. Returns the clean frame sorted by symbol/date."""
    cols = [c for c in C.RAW_TO_INTERNAL if c in raw.columns]
    df = raw[cols].rename(columns=C.RAW_TO_INTERNAL).copy()
    rep.series_found = {str(k): int(v) for k, v in df["series"].value_counts().items()}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "prev_close", "last", "settle", "volume", "turnover", "trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip()

    n = len(df)
    if series:
        df = df[df["series"].isin(series)]
    rep.other_series_removed = n - len(df)

    ok = df["date"].notna()
    rep.bad_dates = int((~ok).sum()); df = df[ok]

    price_ok = df[C.PRICE_COLS].notna().all(axis=1) & (df[C.PRICE_COLS] > 0).all(axis=1)
    rep.non_positive_prices = int((~price_ok).sum()); df = df[price_ok]

    hl = df["high"] >= df["low"] - C.OHLC_TOLERANCE
    rep.high_below_low = int((~hl).sum()); df = df[hl]

    vol_ok = df["volume"].notna() & (df["volume"] >= 0) & df["turnover"].notna() & df["trades"].notna()
    rep.negative_volume = int((~vol_ok).sum()); df = df[vol_ok]

    before = len(df)
    df = df.drop_duplicates(["date", "symbol", "series"], keep="last")
    rep.key_conflicts_dropped = before - len(df)

    # Flags only: these rows are kept.
    outside = (df["close"] > df["high"] + C.OHLC_TOLERANCE) | (df["close"] < df["low"] - C.OHLC_TOLERANCE)
    rep.flagged_close_outside_range = int(outside.sum())
    rep.flagged_zero_volume = int((df["volume"] == 0).sum())
    ret = (df["close"] / df["prev_close"] - 1) * 100
    rep.flagged_extreme_returns = int((ret.abs() > C.EXTREME_RETURN_PCT).sum())

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    rep.rows_clean = len(df)
    if df.empty:
        raise DataError("No valid rows remain after cleaning.")
    rep.n_symbols = int(df["symbol"].nunique())
    rep.n_dates = int(df["date"].nunique())
    rep.date_min = str(df["date"].min().date())
    rep.date_max = str(df["date"].max().date())
    return df, rep


def load_and_clean(path, series=("EQ",)) -> tuple[pd.DataFrame, QualityReport]:
    raw, rep = load_raw(path)
    return clean(raw, rep, series)


def detect_corporate_actions(df: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """Find days where today's *previous close* differs from the actual close of the previous
    trading day. NSE adjusts PrvsClsgPric on the ex-date of splits/bonuses/rights, so the
    ratio reveals the adjustment. Requires at least two consecutive trading days."""
    empty = pd.DataFrame(columns=["date", "symbol", "adjustment_ratio", "likely_action"])
    dates = np.sort(df["date"].unique())
    if len(dates) < 2:
        return empty
    pos = pd.Series(np.arange(len(dates)), index=dates)
    d = df[["date", "symbol", "close", "prev_close"]].copy()
    d["_pos"] = d["date"].map(pos)
    d = d.sort_values(["symbol", "date"])
    g = d.groupby("symbol")
    d["_prev_pos"] = g["_pos"].shift(1)
    d["_prev_day_close"] = g["close"].shift(1)
    d = d[d["_pos"] - d["_prev_pos"] == 1].copy()          # only truly consecutive sessions
    d["adjustment_ratio"] = d["prev_close"] / d["_prev_day_close"]
    out = d[(d["adjustment_ratio"] - 1).abs() > tol].copy()
    if out.empty:
        return empty
    out["likely_action"] = np.where(out["adjustment_ratio"] < 0.9, "split/bonus-like (price adjusted down)",
                           np.where(out["adjustment_ratio"] > 1.1, "reverse split/consolidation-like",
                                    "dividend/rights-like (small adjustment)"))
    return out[["date", "symbol", "adjustment_ratio", "likely_action"]].reset_index(drop=True)


def add_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """Corporate-action-safe price level: compound the exchange-adjusted daily returns
    (close / prev_close) per symbol, anchored on each symbol's first observed close."""
    d = df.sort_values(["symbol", "date"]).copy()
    r = d["close"] / d["prev_close"]
    cum = r.groupby(d["symbol"]).cumprod()
    first_close = d.groupby("symbol")["close"].transform("first")
    first_r = r.groupby(d["symbol"]).transform("first")
    d["adj_close"] = first_close * cum / first_r
    return d
