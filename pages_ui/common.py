"""Shared UI helpers: context object, cached computations, notices and downloads."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from stocksense import config as C
from stocksense.features import add_daily_features
from stocksense.preprocessing import QualityReport
from stocksense.timeseries import market_series


@dataclass
class Context:
    df: pd.DataFrame
    rep: QualityReport
    key: str

    @property
    def n_days(self) -> int:
        return self.rep.n_dates

    @property
    def multi_day(self) -> bool:
        return self.n_days >= 2

    def enough(self, feature: str) -> bool:
        return self.n_days >= C.MIN_DAYS[feature]


@st.cache_data(show_spinner=False)
def daily_features(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    return add_daily_features(_df).replace([float("inf"), float("-inf")], float("nan"))


@st.cache_data(show_spinner=False)
def cached_market_series(_df: pd.DataFrame, key: str, weighting: str) -> pd.DataFrame:
    return market_series(_df, weighting)


def need_more_days(ctx: Context, feature: str, what: str):
    st.info(f"**{what}** needs at least **{C.MIN_DAYS[feature]} trading days** of history. "
            f"The loaded data has **{ctx.n_days}** ({ctx.rep.date_min} to {ctx.rep.date_max}). "
            "Add more daily files in the same NSE format (put them in one folder and point the sidebar path at it) "
            "and this analysis unlocks automatically. Nothing is simulated or estimated in the meantime.")


def download(df: pd.DataFrame, label: str, name: str, key: str | None = None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key or name)


def kpi(cols, items):
    for c, (label, value, *help_) in zip(cols, items):
        c.metric(label, value, help=help_[0] if help_ else None)
