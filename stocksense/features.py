"""Feature engineering.

Three layers:
1. `add_daily_features`     - per-row features from one day's OHLCV (works with a single day).
2. `add_temporal_features`  - lags, rolling volatility, forward targets (needs several days).
3. `ItemBinner`             - turns features into discrete "items" for frequent pattern mining.
   Cut-points are *fitted on a training slice only* and re-applied to later data (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- daily features
def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_pct"] = (d["close"] / d["prev_close"] - 1) * 100            # exchange-adjusted day return
    d["gap_pct"] = (d["open"] / d["prev_close"] - 1) * 100             # overnight gap
    d["intraday_pct"] = (d["close"] / d["open"] - 1) * 100             # open -> close
    d["range_pct"] = (d["high"] - d["low"]) / d["low"] * 100
    rng = (d["high"] - d["low"]).to_numpy()
    pos = np.divide((d["close"] - d["low"]).to_numpy(), rng, out=np.full(len(d), 0.5), where=rng > 0)
    d["close_pos"] = np.clip(pos, 0, 1)                                 # 0 = closed at low, 1 = at high
    vol = d["volume"].replace(0, np.nan)
    d["vwap"] = d["turnover"] / vol
    d["vwap_dev_pct"] = (d["close"] / d["vwap"] - 1) * 100
    d["parkinson_vol_pct"] = np.log(d["high"] / d["low"]) / (2 * np.sqrt(np.log(2))) * 100
    d["avg_trade_value"] = d["turnover"] / d["trades"].replace(0, np.nan)
    d["log_price"] = np.log(d["close"])
    d["log_prev_close"] = np.log(d["prev_close"])
    d["log_turnover"] = np.log10(d["turnover"].clip(lower=1))
    d["log_volume"] = np.log10(d["volume"].clip(lower=1))
    d["log_trades"] = np.log10(d["trades"].clip(lower=1))
    return d


# ------------------------------------------------------------------ temporal (multi-day) features
TEMPORAL_FEATURES = ["lr", "lr_lag1", "lr_lag2", "lr_lag3", "lr_lag4", "mom5", "mom10",
                     "vol5", "vol10", "vol20", "range_pct", "close_pos", "gap_pct",
                     "volume_ratio", "mkt_lr"]


def add_temporal_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Adds lag/rolling features and the forward target `fwd_ret` (log return from today's close
    to the close `horizon` sessions later). Every feature at row t uses data up to t only."""
    d = add_daily_features(df).sort_values(["symbol", "date"]).reset_index(drop=True)
    d["lr"] = np.log(d["close"] / d["prev_close"])
    g = d.groupby("symbol")["lr"]
    for k in range(1, 5):
        d[f"lr_lag{k}"] = g.shift(k)
    d["mom5"] = g.transform(lambda s: s.rolling(5).sum())
    d["mom10"] = g.transform(lambda s: s.rolling(10).sum())
    for w in (5, 10, 20):
        d[f"vol{w}"] = g.transform(lambda s, w=w: s.rolling(w).std())
    lv = d.groupby("symbol")["log_volume"]
    d["volume_ratio"] = d["log_volume"] - lv.transform(lambda s: s.rolling(20).mean())
    d["mkt_lr"] = d.groupby("date")["lr"].transform("mean")
    # target: sum of the next `horizon` daily log returns
    d["fwd_ret"] = g.transform(lambda s: s.rolling(horizon).sum().shift(-horizon))
    return d


# ----------------------------------------------------------------------------- item binning
# (group name) -> (column, kind, edges-or-None, labels)
_FIXED = {
    "Return":   ("ret_pct", [-np.inf, -2, -0.5, 0.5, 2, np.inf], ["Strong Down", "Down", "Flat", "Up", "Strong Up"]),
    "Gap":      ("gap_pct", [-np.inf, -1, 1, np.inf], ["Gap Down", "No Gap", "Gap Up"]),
    "Intraday": ("intraday_pct", [-np.inf, -1, 1, np.inf], ["Fell", "Flat", "Rose"]),
    "ClosePos": ("close_pos", [-np.inf, 0.25, 0.75, np.inf], ["Near Low", "Mid", "Near High"]),
    "VWAP":     ("vwap_dev_pct", [-np.inf, -0.5, 0.5, np.inf], ["Below VWAP", "Near VWAP", "Above VWAP"]),
    "Price":    ("close", [0, 50, 500, 2000, np.inf], ["Under 50", "50-500", "500-2000", "Above 2000"]),
}
_QUANTILE = {   # relative levels -> terciles fitted on training data
    "Range":     ("range_pct", ["Low", "Medium", "High"]),
    "Liquidity": ("turnover", ["Low", "Medium", "High"]),
    "TradeSize": ("avg_trade_value", ["Small", "Medium", "Large"]),
}
ALL_GROUPS = list(_FIXED) + list(_QUANTILE)


class ItemBinner:
    """Converts numeric stock-day features into boolean 'items' (one column per Group=Label)."""

    def __init__(self):
        self.cuts_: dict[str, list[float]] = {}

    def fit(self, df: pd.DataFrame) -> "ItemBinner":
        d = add_daily_features(df) if "ret_pct" not in df else df
        for g, (col, _) in _QUANTILE.items():
            q = d[col].replace([np.inf, -np.inf], np.nan).dropna().quantile([1 / 3, 2 / 3]).to_numpy()
            self.cuts_[g] = [float(q[0]), float(q[1])]
        return self

    def _labels(self, d: pd.DataFrame, g: str) -> pd.Series:
        if g in _FIXED:
            col, edges, labels = _FIXED[g]
            return pd.cut(d[col], bins=edges, labels=labels, right=False).astype("object")
        col, labels = _QUANTILE[g]
        lo, hi = self.cuts_[g]
        x = d[col]
        out = pd.Series(np.where(x < lo, labels[0], np.where(x < hi, labels[1], labels[2])), index=d.index, dtype="object")
        return out.where(x.notna())

    def item_names(self, groups, lag: bool = False) -> list[str]:
        names = []
        for g in groups:
            labels = _FIXED[g][2] if g in _FIXED else _QUANTILE[g][1]
            names += [f"{g}={l}" for l in labels]
            if lag:
                names += [f"Prev {g}={l}" for l in labels]
        return names

    def transform(self, df: pd.DataFrame, groups=ALL_GROUPS, lag: bool = False) -> pd.DataFrame:
        if not self.cuts_:
            raise RuntimeError("ItemBinner must be fitted first.")
        d = add_daily_features(df) if "ret_pct" not in df else df
        d = d.reset_index(drop=True)
        cols = {}
        for g in groups:
            lab = self._labels(d, g)
            labels = _FIXED[g][2] if g in _FIXED else _QUANTILE[g][1]
            for l in labels:
                cols[f"{g}={l}"] = (lab == l).to_numpy()
            if lag:
                prev = lab.groupby(d["symbol"]).shift(1)
                for l in labels:
                    cols[f"Prev {g}={l}"] = (prev == l).to_numpy()
        out = pd.DataFrame(cols, index=d.index).astype(bool)
        if lag:   # first day of each symbol has no 'previous' info: drop those transactions
            has_prev = d.groupby("symbol")["date"].cumcount() > 0
            out = out[has_prev.to_numpy()]
        return out


def holdout_split_by_symbol(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    """Deterministic *cross-sectional* split (symbols are hashed into train/test).
    Used only when there is a single trading day. It is NOT a time-based validation."""
    rng = np.random.default_rng(seed)
    syms = np.sort(df["symbol"].unique())
    test = set(rng.choice(syms, size=int(len(syms) * test_frac), replace=False))
    m = df["symbol"].isin(test)
    return df[~m], df[m]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Time-based split on trading dates: earliest `train_frac` of dates -> train, rest -> test."""
    dates = np.sort(df["date"].unique())
    cut = dates[max(1, min(len(dates) - 1, int(len(dates) * train_frac)))]
    return df[df["date"] < cut], df[df["date"] >= cut]
