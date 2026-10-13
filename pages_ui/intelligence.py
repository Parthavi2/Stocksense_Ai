"""Page 5 - Market Intelligence: regimes, anomalies, clustering & similarity."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from stocksense import market_intel as MI
from stocksense import timeseries as T
from stocksense import visualization as V

from .common import Context, cached_market_series, daily_features, download, kpi, need_more_days


@st.cache_data(show_spinner="Detecting anomalies...")
def _anomalies(_df, key, contamination):
    return MI.detect_anomalies(_df, contamination=contamination)


@st.cache_data(show_spinner="Choosing number of clusters...")
def _choose_k(_df, key):
    return MI.choose_k(_df)


@st.cache_data(show_spinner="Clustering...")
def _cluster(_df, key, k):
    return MI.cluster_stocks(_df, k)


def render(ctx: Context):
    st.header("Market Intelligence Dashboard")
    d = daily_features(ctx.df, ctx.key)
    snap = d[d["date"] == d["date"].max()]
    t1, t2, t3 = st.tabs(["Market regime", "Anomaly detection", "Clustering & similarity"])
    with t1:
        _regime(ctx, snap)
    with t2:
        _anomaly(ctx, d, snap)
    with t3:
        _clusters(ctx, d, snap)


def _regime(ctx, snap):
    b = MI.breadth_regime(snap)
    st.subheader(f"Latest session ({snap['date'].max().date()}): {b['regime']}")
    kpi(st.columns(4), [("Advancers / decliners", f"{b['advancers']:,} / {b['decliners']:,}"), ("Median return", f"{b['median_ret_pct']:.2f}%"),
                        ("Up >= 2%", f"{b['pct_up_2']:.1f}%"), ("Down <= -2%", f"{b['pct_down_2']:.1f}%")])
    st.caption("Rule: " + b["rule"] + " This is a transparent description of one day's breadth, not a statistical model.")
    if not ctx.enough("regimes"):
        need_more_days(ctx, "regimes", "Statistical regime detection (Gaussian mixture on rolling return & volatility)")
        return
    ms = cached_market_series(ctx.df, ctx.key, "equal")
    n = st.slider("Number of regimes", 2, 4, 3, key="rg_n")
    w = st.slider("Rolling window (sessions)", 10, 60, 20, key="rg_w")
    try:
        rg = T.detect_regimes(ms["ret"], n, w)
    except Exception as e:
        st.error(f"Regime model failed: {e}"); return
    st.plotly_chart(V.regime_fig(ms["level"], rg["labels"], rg["train_end"]), width="stretch")
    c1, c2 = st.columns(2)
    c1.write("Regime characteristics (fitted on the earliest 70% of days):"); c1.dataframe(rg["summary"].round(3), hide_index=True)
    c2.write("Probability of tomorrow's regime given today's:"); c2.dataframe(rg["transitions"])
    download(rg["labels"].reset_index(), "Download regime labels (CSV)", "stocksense_regimes.csv")


def _anomaly(ctx, d, snap):
    cont = st.slider("Expected share of anomalies", 0.005, 0.10, 0.02, 0.005, key="an_c")
    a = _anomalies(snap, f"{ctx.key}|{snap['date'].max()}", cont)
    flagged = a[a["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    kpi(st.columns(3), [("Stocks analysed", f"{len(a):,}"), ("Flagged", f"{len(flagged):,}"), ("Most common driver", flagged["main_driver"].mode().iat[0] if len(flagged) else "-")])
    xs = st.columns(2)
    x = xs[0].selectbox("X axis", ["ret_pct", "gap_pct", "intraday_pct", "range_pct", "vwap_dev_pct"], key="an_x")
    y = xs[1].selectbox("Y axis", ["range_pct", "ret_pct", "gap_pct", "log_turnover", "log_trades"], key="an_y")
    st.plotly_chart(V.anomaly_scatter(a, x, y), width="stretch")
    cols = ["symbol", "name", "close", "ret_pct", "gap_pct", "range_pct", "log_turnover", "anomaly_score", "main_driver", "driver_robust_z"]
    st.dataframe(flagged[cols].round(3), hide_index=True, width="stretch")
    download(flagged[cols], "Download anomalies (CSV)", "stocksense_anomalies.csv")
    st.caption("An Isolation Forest isolates stocks whose combination of return, gap, range, VWAP deviation and activity is unusual versus all others that day. "
               "'Main driver' is the feature with the largest robust z-score. An anomaly is unusual, not necessarily wrong or tradeable.")
    if ctx.multi_day:
        st.subheader("Historical anomaly days for one stock")
        sym = st.selectbox("Stock", sorted(ctx.df["symbol"].unique()), key="an_sym")
        zt = st.slider("Robust z threshold", 2.5, 6.0, 3.5, 0.5, key="an_z")
        try:
            h = MI.stock_time_anomalies(ctx.df, sym, zt)
            s = T.stock_series(ctx.df, sym)
            fig = px.line(s, y="level").update_layout(height=380, template="plotly_white", title=f"{sym} with anomaly markers", margin=dict(t=50))
            pts = s.reindex(h.index[h["is_anomaly"]]).dropna()
            fig.add_scatter(x=pts.index, y=pts["level"], mode="markers", name="anomaly", marker=dict(color=V.PALETTE["down"], size=9, symbol="x"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(h[h["is_anomaly"]].round(2).reset_index(), hide_index=True, width="stretch")
        except ValueError as e:
            st.info(str(e))


def _clusters(ctx, d, snap):
    sc = _choose_k(snap, f"{ctx.key}|{snap['date'].max()}")
    best = int(sc.loc[sc["silhouette"].idxmax(), "k"])
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(V.silhouette_fig(sc), width="stretch")
    k = c2.slider("Number of clusters", 2, 8, best, key="cl_k", help=f"Silhouette-optimal k = {best}")
    c2.caption(f"Silhouette score at k={k}: {float(sc.loc[sc['k'] == k, 'silhouette'].iloc[0]):.3f} "
               "(closer to 1 = better separated; values below ~0.25 mean weak cluster structure).")
    cl, profile = _cluster(snap, f"{ctx.key}|{snap['date'].max()}", k)
    st.plotly_chart(V.cluster_scatter(cl), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(V.profile_heatmap(profile), width="stretch")
    b.write("Cluster medians:"); b.dataframe(profile.round(2), width="stretch")
    download(cl[["symbol", "name", "cluster", "pc1", "pc2"]], "Download cluster assignments (CSV)", "stocksense_clusters.csv")

    st.subheader("Find similar stocks")
    sym = st.selectbox("Stock", sorted(cl["symbol"]), key="sim_sym")
    n = st.slider("Neighbours", 3, 25, 10, key="sim_n")
    sim = MI.similar_stocks(snap, sym, n=n)
    st.dataframe(sim.round(3), hide_index=True, width="stretch")
    if ctx.multi_day:
        corr = MI.return_correlations(ctx.df, [sym] + list(sim["symbol"][:10]))
        st.plotly_chart(V.corr_heatmap(corr), width="stretch")
    else:
        st.caption("Similarity is measured on today's behaviour features (distance in robust-scaled space). Return-correlation similarity needs several days of history.")
