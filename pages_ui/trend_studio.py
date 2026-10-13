"""Page 3 - Trend & Variation Studio: trend, seasonality, cycles, residual variation."""
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

from stocksense import timeseries as T
from stocksense import visualization as V
from stocksense.market_intel import stock_time_anomalies

from .common import Context, cached_market_series, daily_features, download, need_more_days, kpi


def render(ctx: Context):
    st.header("Trend & Variation Studio")
    if not ctx.multi_day:
        _single_day(ctx)
        return
    _multi_day(ctx)


# ============================================================================ single trading day
def _single_day(ctx: Context):
    st.warning(f"Trend, seasonality and cycles describe how prices move **over time**. The data has one session ({ctx.rep.date_min}), "
               "so those analyses cannot be computed honestly. Below are the closest valid *cross-sectional* views of the same ideas.")
    d = daily_features(ctx.df, ctx.key)
    tab1, tab2, tab3 = st.tabs(["Direction of the session", "Residual variation", "Seasonal & cyclical (locked)"])

    with tab1:
        thr = st.slider("Sideways band: |day return| below (%)", 0.1, 2.0, 0.5, 0.1, key="ts_thr")
        d = d.assign(direction=np.where(d["ret_pct"] >= thr, "Upward", np.where(d["ret_pct"] <= -thr, "Downward", "Sideways")))
        cnt = d["direction"].value_counts().reindex(["Upward", "Sideways", "Downward"]).fillna(0).reset_index()
        cnt.columns = ["direction", "stocks"]
        c1, c2 = st.columns([1, 2])
        c1.dataframe(cnt.assign(share=(cnt["stocks"] / cnt["stocks"].sum()).map("{:.1%}".format)), hide_index=True)
        c2.plotly_chart(px.bar(cnt, x="direction", y="stocks", color="direction", color_discrete_map={
            "Upward": V.PALETTE["up"], "Sideways": V.PALETTE["neutral"], "Downward": V.PALETTE["down"]}).update_layout(
            height=300, template="plotly_white", showlegend=False, margin=dict(t=20)), width="stretch")
        st.caption("Each stock is classified by its close versus the previous close. This is a one-day *state*, not a trend over time.")
        agree = pd.crosstab(np.sign(d["gap_pct"]).map({1: "Gap up", -1: "Gap down", 0: "No gap"}), d["direction"])
        st.write("How the opening gap relates to the session direction (counts):")
        st.dataframe(agree, width="stretch")

    with tab2:
        st.write("**How much of today's return variation is explained, and how much is left unexplained?** "
                 "An OLS model regresses each stock's return on its gap, size/price tier and liquidity tier. The residual is the part those "
                 "factors cannot explain; large residuals are flagged, not forecast.")
        z = d.dropna(subset=["ret_pct", "gap_pct", "log_turnover", "log_price"])
        X = sm.add_constant(z[["gap_pct", "log_turnover", "log_price", "range_pct"]])
        ols = sm.OLS(z["ret_pct"], X).fit(cov_type="HC3")
        kpi(st.columns(3), [("R-squared", f"{ols.rsquared:.3f}"), ("Unexplained share of variance", f"{1 - ols.rsquared:.1%}"),
                            ("Residual std (%)", f"{ols.resid.std():.2f}")])
        thr = st.slider("Flag |robust z| above", 2.0, 6.0, 3.5, 0.5, key="res_z")
        r = ols.resid
        mad = np.median(np.abs(r - r.median())) * 1.4826
        zs = (r - r.median()) / mad
        z = z.assign(residual=r, robust_z=zs)
        fl = z[np.abs(z["robust_z"]) > thr].sort_values("robust_z", key=np.abs, ascending=False)
        st.plotly_chart(px.histogram(z, x="residual", nbins=80, color_discrete_sequence=[V.PALETTE["primary"]]).update_layout(
            height=320, template="plotly_white", title="Residual distribution", margin=dict(t=50)), width="stretch")
        st.write(f"**{len(fl)} stocks** have unexplained moves beyond the threshold:")
        st.dataframe(fl[["symbol", "name", "ret_pct", "gap_pct", "range_pct", "residual", "robust_z"]].head(50), hide_index=True, width="stretch")
        download(fl[["symbol", "name", "ret_pct", "residual", "robust_z"]], "Download flagged residuals (CSV)", "stocksense_residuals.csv")
        st.caption("Residuals are heavy-tailed: some stocks move far more than any observable factor explains. Nothing here predicts them.")
        with st.expander("Regression coefficients"):
            st.dataframe(pd.DataFrame({"coef": ols.params, "robust p-value": ols.pvalues}).round(4))

    with tab3:
        for name, feat in [("Weekday & monthly seasonality", "weekday_seasonality"), ("Cyclical variation / market cycles", "cycles")]:
            st.subheader(name)
            need_more_days(ctx, feat, name)


# ============================================================================== multi-day
def _multi_day(ctx: Context):
    df = ctx.df
    c1, c2, c3 = st.columns(3)
    scope = c1.selectbox("Series", ["Market index (equal-weight)", "Market index (turnover-weighted)", "Single stock"], key="ts_scope")
    if scope == "Single stock":
        sym = c2.selectbox("Stock", sorted(df["symbol"].unique()), key="ts_sym")
        s = T.stock_series(df, sym); label = sym
    else:
        s = cached_market_series(df, ctx.key, "turnover" if "turnover" in scope else "equal"); label = "Market index"
    lo, hi = s.index.min().date(), s.index.max().date()
    rng = c3.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key="ts_range")
    if len(rng) == 2:
        s = s.loc[str(rng[0]):str(rng[1])]
    st.caption(f"{label}: {len(s)} sessions. Levels compound exchange-adjusted daily returns, so splits/bonuses do not appear as crashes.")

    t1, t2, t3, t4 = st.tabs(["Trend", "Seasonality", "Cyclical variation", "Residual variation"])

    with t1:
        w = st.slider("Trend window (sessions)", 10, 120, 40, 5, key="tr_w")
        m = st.slider("Min slope for trend (% per day)", 0.0, 0.5, 0.05, 0.01, key="tr_m")
        try:
            T.require_days(len(s), "trend")
            tr = T.classify_trend(s["level"], w, m)
            marks = None
            if scope == "Single stock":
                try:
                    an = stock_time_anomalies(df, sym)
                    marks = an.index[an["is_anomaly"]]
                except ValueError:
                    pass
            st.plotly_chart(V.trend_chart(tr, label, marks), width="stretch")
            seg = T.trend_segments(tr)
            share = (tr["trend"].value_counts(normalize=True) * 100).round(1)
            kpi(st.columns(3), [("Uptrend days", f"{share.get('Uptrend', 0)}%"), ("Downtrend days", f"{share.get('Downtrend', 0)}%"),
                                ("Sideways days", f"{share.get('Sideways', 0)}%")])
            st.dataframe(seg, hide_index=True, width="stretch")
            download(seg, "Download trend segments (CSV)", "stocksense_trend_segments.csv")
        except (T.InsufficientHistory, ValueError) as e:
            st.info(str(e))

    with t2:
        try:
            wd, p = T.seasonal_stats(s["ret"], "weekday")
            st.plotly_chart(V.seasonal_bar(wd, "Average return by weekday"), width="stretch")
            st.caption(f"Kruskal-Wallis p = {p:.3f}. " + ("Weekday differences are statistically detectable." if p < 0.05 else
                       "No statistically reliable weekday effect - apparent differences are within noise."))
        except T.InsufficientHistory as e:
            st.info(str(e))
        try:
            mo, p = T.seasonal_stats(s["ret"], "month")
            st.plotly_chart(V.seasonal_bar(mo, "Average return by calendar month"), width="stretch")
            st.caption(f"Kruskal-Wallis p = {p:.3f}. With 12 months some bars look large by chance; require a small p AND consistency across years.")
        except T.InsufficientHistory:
            need_more_days(ctx, "monthly_seasonality", "Monthly seasonality")
        try:
            period = st.selectbox("STL seasonal period (sessions)", [5, 21, 63], index=1, key="stl_p")
            stl = T.stl_decompose(s["level"], period)
            st.plotly_chart(V.decomposition_fig(stl), width="stretch")
            st.session_state["_stl_resid"] = stl["resid"]
        except T.InsufficientHistory as e:
            st.info(str(e))

    with t3:
        try:
            lamb = st.select_slider("HP smoothing (lambda)", [1e8, 1e9, 1e10, 2.5e10, 1e11], value=2.5e10, key="hp_l")
            cyc = T.detect_cycles(s["level"], lamb)
            st.plotly_chart(V.cycle_fig(cyc), width="stretch")
            tab = cyc["table"].round(2)
            st.dataframe(tab, hide_index=True, width="stretch")
            download(tab, "Download cycle table (CSV)", "stocksense_cycles.csv")
            st.caption("Periods are *candidates*: only those that fit at least 3 times in the sample are shown, and a spectral peak can arise from noise. "
                       "'Relative power' compares the peak with the median spectral level; values near 1-3 are weak evidence.")
        except T.InsufficientHistory:
            need_more_days(ctx, "cycles", "Cyclical analysis")

    with t4:
        try:
            stl = T.stl_decompose(s["level"], 21)
            ra = T.residual_analysis(stl["resid"], st.slider("Flag |robust z| above", 2.0, 6.0, 3.5, 0.5, key="mr_z"))
            kpi(st.columns(4), [("Ljung-Box p (autocorr.)", f"{ra['ljung_box_p']:.3f}"), ("Ljung-Box p (squared)", f"{ra['ljung_box_sq_p']:.3f}"),
                                ("Excess kurtosis", f"{ra['excess_kurtosis']:.2f}"), ("Flagged days", f"{len(ra['flagged'])}")])
            st.plotly_chart(V.residual_fig(stl["resid"], ra["z"], ra["flagged"]), width="stretch")
            st.caption("Residuals are what trend + seasonality cannot explain. A small Ljung-Box p on *squared* residuals means volatility clusters "
                       "(big moves follow big moves) - useful for risk, but the direction of residual moves stays unpredictable. Flagged days are shown, never forecast.")
            download(ra["flagged"].rename("robust_z").reset_index(), "Download flagged residual days (CSV)", "stocksense_residual_days.csv")
        except T.InsufficientHistory:
            need_more_days(ctx, "residuals", "Residual analysis")
