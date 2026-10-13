"""Page 1 - Market Overview: data profile, quality, breadth, risk, stock comparison."""
import numpy as np
import pandas as pd
import streamlit as st

from stocksense import visualization as V
from stocksense.market_intel import breadth_regime, stock_time_anomalies
from stocksense.preprocessing import add_adjusted_close, detect_corporate_actions
from stocksense.timeseries import risk_metrics

from .common import Context, daily_features, download, kpi


def render(ctx: Context):
    rep, df = ctx.rep, ctx.df
    st.header("Market Overview")
    kpi(st.columns(5), [
        ("Stocks (EQ)", f"{rep.n_symbols:,}"), ("Trading days", f"{rep.n_dates}"),
        ("Date range", rep.date_min if rep.n_dates == 1 else f"{rep.date_min} to {rep.date_max}"),
        ("Raw rows", f"{rep.rows_raw:,}"), ("Clean rows", f"{rep.rows_clean:,}")])

    if rep.exact_duplicates > 0.5 * rep.rows_raw:
        st.warning(f"**{rep.exact_duplicates:,} of {rep.rows_raw:,} raw rows ({rep.exact_duplicates / rep.rows_raw:.1%}) were exact duplicates** "
                   f"and were removed. The file contains only {rep.rows_clean:,} distinct stock-day records.")
    if ctx.n_days == 1:
        st.warning(f"The data covers a **single trading day ({rep.date_min})**. Time-series features (trend, seasonality, cycles, "
                   "forecasting over time, backtesting) are disabled rather than faked. Cross-sectional analyses use all stocks of that day.")

    with st.expander("Data quality report", expanded=False):
        c1, c2 = st.columns(2)
        c1.dataframe(rep.issues_table(), hide_index=True, width="stretch")
        fill = pd.DataFrame({"column": list(rep.column_fill_rate), "non-null share": list(rep.column_fill_rate.values())})
        c2.dataframe(fill, hide_index=True, width="stretch", height=380)
        st.caption(f"Columns that are entirely empty and therefore unused: {', '.join(rep.unused_empty_columns) or 'none'}. "
                   f"Series found: {rep.series_found}. Files: {', '.join(rep.files)}.")
        if ctx.multi_day:
            ca = detect_corporate_actions(df)
            st.write(f"**Corporate actions detected:** {len(ca)} (previous-close adjustments on consecutive sessions)")
            if len(ca):
                st.dataframe(ca, hide_index=True, width="stretch")
        else:
            st.caption("Corporate-action detection compares each day's adjusted previous close with the prior day's close, so it needs 2+ days. "
                       "Returns here use the exchange-adjusted previous close, so splits/bonuses do not distort them.")

    d = daily_features(df, ctx.key)
    last_date = d["date"].max()
    snap = d[d["date"] == last_date].copy()

    # ---- filters
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        pmin, pmax = float(snap["close"].min()), float(snap["close"].max())
        price = f1.slider("Close price (INR)", pmin, pmax, (pmin, pmax), key="ov_price")
        min_to = f2.number_input("Min turnover (INR lakh)", 0.0, float(snap["turnover"].max() / 1e5), 0.0, step=10.0, key="ov_to")
        q = f3.text_input("Symbol contains", "", key="ov_q").upper().strip()
    snap = snap[snap["close"].between(*price) & (snap["turnover"] >= min_to * 1e5)]
    if q:
        snap = snap[snap["symbol"].str.contains(q, regex=False)]
    st.caption(f"Showing {len(snap):,} stocks for {last_date.date()} after filters.")
    if snap.empty:
        st.error("No stocks match the filters.")
        return

    # ---- breadth & distribution
    b = breadth_regime(snap)
    kpi(st.columns(5), [
        ("Advancers / Decliners", f"{b['advancers']:,} / {b['decliners']:,}"), ("A/D ratio", f"{b['ad_ratio']:.2f}"),
        ("Median return", f"{b['median_ret_pct']:.2f}%"), ("Turnover-weighted return", f"{b['turnover_weighted_ret_pct']:.2f}%"),
        ("Dispersion (std)", f"{b['dispersion_pct']:.2f}%", "Cross-sectional standard deviation of returns")])
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(V.breadth_donut(b["advancers"], b["decliners"], b["unchanged"]), width="stretch")
    c2.plotly_chart(V.histogram(snap["ret_pct"], "Distribution of day returns", "return %"), width="stretch")

    c1, c2 = st.columns(2)
    cols = ["symbol", "name", "close", "ret_pct", "turnover", "range_pct"]
    c1.subheader("Top gainers"); c1.dataframe(snap.nlargest(10, "ret_pct")[cols], hide_index=True, width="stretch")
    c2.subheader("Top losers"); c2.dataframe(snap.nsmallest(10, "ret_pct")[cols], hide_index=True, width="stretch")
    st.plotly_chart(V.treemap_turnover(snap), width="stretch")

    # ---- risk
    st.subheader("Risk metrics")
    if ctx.multi_day:
        st.caption("Computed from each stock's full daily return history.")
    else:
        st.caption("With one day, risk is measured from that session: intraday range, Parkinson volatility estimate "
                   "(from high/low) and the size of the overnight gap. These are *single-day* proxies, not annualised volatility.")
        risk = snap[["symbol", "name", "close", "range_pct", "parkinson_vol_pct", "gap_pct", "intraday_pct"]].copy()
        risk["abs_gap_pct"] = risk["gap_pct"].abs()
        st.dataframe(risk.nlargest(15, "range_pct"), hide_index=True, width="stretch")
        download(risk, "Download risk table (CSV)", "stocksense_risk_snapshot.csv")

    # ---- comparison
    st.subheader("Stock comparison")
    default = list(snap.nlargest(4, "turnover")["symbol"])
    picks = st.multiselect("Compare stocks", sorted(snap["symbol"]), default=default, key="ov_pick")
    if picks:
        cmp = snap[snap["symbol"].isin(picks)][["symbol", "name", "open", "high", "low", "close", "prev_close", "ret_pct",
                                                 "gap_pct", "intraday_pct", "range_pct", "close_pos", "turnover", "trades"]]
        st.dataframe(cmp, hide_index=True, width="stretch")
        a, b_ = st.columns(2)
        a.plotly_chart(V.compare_bars(cmp, "ret_pct", "Day return %"), width="stretch")
        b_.plotly_chart(V.compare_bars(cmp, "range_pct", "Intraday range %"), width="stretch")
        one = st.selectbox("Inspect one stock", picks, key="ov_one")
        row = d[(d["symbol"] == one) & (d["date"] == last_date)].iloc[0]
        st.plotly_chart(V.day_candle(row), width="stretch")
        if ctx.multi_day:
            hist = add_adjusted_close(df[df["symbol"].isin(picks)])
            norm = hist.pivot(index="date", columns="symbol", values="adj_close")
            st.plotly_chart(V.px.line((norm / norm.iloc[0] * 100)).update_layout(height=380, template="plotly_white",
                            title="Adjusted price, rebased to 100", margin=dict(t=50)), width="stretch")
            hist["r"] = hist["close"] / hist["prev_close"] - 1
            rm = pd.DataFrame({sym: risk_metrics(g.set_index("date")["r"]) for sym, g in hist.groupby("symbol")}).T
            st.dataframe(rm.round(2), width="stretch")
            try:
                an = stock_time_anomalies(df, one)
                st.caption(f"{int(an['is_anomaly'].sum())} historical anomaly days flagged for {one} (robust z > 3.5).")
            except ValueError as e:
                st.caption(str(e))
        download(cmp, "Download comparison (CSV)", "stocksense_comparison.csv")
    download(snap, "Download filtered snapshot (CSV)", "stocksense_snapshot.csv")
