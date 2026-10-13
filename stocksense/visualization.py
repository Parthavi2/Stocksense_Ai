"""Plotly figure builders. No analysis happens here - functions only draw what they are given."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {"primary": "#2563EB", "up": "#16A34A", "down": "#DC2626", "neutral": "#64748B",
           "accent": "#F59E0B", "purple": "#7C3AED"}
TREND_COLORS = {"Uptrend": "rgba(22,163,74,0.15)", "Downtrend": "rgba(220,38,38,0.15)", "Sideways": "rgba(100,116,139,0.10)"}
SEQ = px.colors.qualitative.Safe


def _style(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), height=height, template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
                      font=dict(family="Inter, Segoe UI, sans-serif"))
    return fig


# ------------------------------------------------------------------------------- overview
def histogram(s: pd.Series, title: str, xlabel: str, nbins: int = 60, clip=(0.5, 99.5)) -> go.Figure:
    lo, hi = np.nanpercentile(s, clip)
    fig = px.histogram(s.clip(lo, hi), nbins=nbins, color_discrete_sequence=[PALETTE["primary"]])
    fig.add_vline(x=float(s.median()), line_dash="dash", line_color=PALETTE["accent"], annotation_text="median")
    fig.update_layout(xaxis_title=xlabel, yaxis_title="stocks", showlegend=False)
    return _style(fig, title, 340)


def bar_h(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, height: int = 400) -> go.Figure:
    fig = px.bar(df.iloc[::-1], x=x, y=y, orientation="h", color=color,
                 color_discrete_sequence=[PALETTE["primary"]], color_continuous_scale="RdYlGn")
    return _style(fig, title, height)


def breadth_donut(adv: int, dec: int, unch: int) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["Advancers", "Decliners", "Unchanged"], values=[adv, dec, unch], hole=0.6,
                           marker_colors=[PALETTE["up"], PALETTE["down"], PALETTE["neutral"]], sort=False))
    return _style(fig, "Market breadth", 320)


def treemap_turnover(df: pd.DataFrame, top: int = 60) -> go.Figure:
    d = df.nlargest(top, "turnover").copy()
    d["turnover_cr"] = d["turnover"] / 1e7
    fig = px.treemap(d, path=["symbol"], values="turnover_cr", color="ret_pct", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, hover_data={"ret_pct": ":.2f", "turnover_cr": ":,.1f"})
    return _style(fig, f"Top {top} by turnover (INR crore) - colour = day return %", 460)


def day_candle(row: pd.Series) -> go.Figure:
    fig = go.Figure(go.Candlestick(x=[row["symbol"]], open=[row["open"]], high=[row["high"]], low=[row["low"]],
                                   close=[row["close"]], increasing_line_color=PALETTE["up"], decreasing_line_color=PALETTE["down"]))
    fig.add_hline(y=row["prev_close"], line_dash="dot", line_color=PALETTE["neutral"], annotation_text="prev close")
    fig.add_hline(y=row["vwap"], line_dash="dash", line_color=PALETTE["purple"], annotation_text="VWAP") if np.isfinite(row.get("vwap", np.nan)) else None
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, f"{row['symbol']} - session OHLC", 360)


def compare_bars(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(df, x="symbol", y=metric, color=metric, color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    return _style(fig, title, 340)


# --------------------------------------------------------------------------------- mining
def itemset_bar(itemsets: pd.DataFrame, top: int = 20) -> go.Figure:
    d = itemsets.head(top).copy()
    d["items"] = d["itemsets"].map(lambda s: " + ".join(sorted(s)))
    fig = px.bar(d.iloc[::-1], x="support", y="items", orientation="h", color="length",
                 color_continuous_scale="Blues")
    return _style(fig, f"Top {len(d)} frequent itemsets by support", max(380, 24 * len(d) + 120))


def rules_scatter(rules: pd.DataFrame) -> go.Figure:
    fig = px.scatter(rules, x="support", y="confidence", color="lift", size="lift", hover_name="rule",
                     color_continuous_scale="Viridis", size_max=14)
    return _style(fig, "Rules: support vs confidence (colour/size = lift)", 420)


def rule_validation_fig(ev: pd.DataFrame) -> go.Figure:
    fig = px.scatter(ev, x="confidence", y="test_confidence", color="holds", hover_name="rule",
                     color_discrete_map={True: PALETTE["up"], False: PALETTE["down"]})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=PALETTE["neutral"]))
    fig.update_layout(xaxis_title="confidence (training)", yaxis_title="confidence (unseen data)")
    return _style(fig, "Rule stability: training vs unseen confidence", 420)


# ------------------------------------------------------------------------- trend / variation
def trend_chart(trend: pd.DataFrame, title: str, anomalies: pd.DatetimeIndex | None = None) -> go.Figure:
    fig = go.Figure()
    t = trend.dropna(subset=["trend"])
    if not t.empty:
        grp = (t["trend"] != t["trend"].shift()).cumsum()
        for _, s in t.groupby(grp):
            fig.add_vrect(x0=s.index[0], x1=s.index[-1], fillcolor=TREND_COLORS[s["trend"].iloc[0]], line_width=0, layer="below")
    fig.add_scatter(x=trend.index, y=trend["level"], name="Price / index", line=dict(color=PALETTE["primary"], width=2))
    fig.add_scatter(x=trend.index, y=trend["sma_fast"], name="Fast SMA", line=dict(color=PALETTE["accent"], width=1))
    fig.add_scatter(x=trend.index, y=trend["sma_slow"], name="Slow SMA", line=dict(color=PALETTE["purple"], width=1))
    if anomalies is not None and len(anomalies):
        pts = trend.reindex(anomalies).dropna(subset=["level"])
        fig.add_scatter(x=pts.index, y=pts["level"], mode="markers", name="Anomaly",
                        marker=dict(color=PALETTE["down"], size=9, symbol="x"))
    return _style(fig, title + "  (green = uptrend, red = downtrend, grey = sideways)", 440)


def seasonal_bar(tab: pd.DataFrame, title: str) -> go.Figure:
    colors = np.where(tab["mean_pct"] >= 0, PALETTE["up"], PALETTE["down"])
    fig = go.Figure(go.Bar(x=tab["label"], y=tab["mean_pct"], marker_color=colors,
                           error_y=dict(type="data", array=tab["ci95"], visible=True)))
    fig.update_layout(yaxis_title="mean daily return % (+/- 95% CI)")
    return _style(fig, title, 340)


def decomposition_fig(stl: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Observed (log level)", "Trend", "Seasonal", "Residual"])
    for i, c in enumerate(["observed", "trend", "seasonal", "resid"], 1):
        fig.add_scatter(x=stl.index, y=stl[c], line=dict(width=1.2, color=PALETTE["primary"]), showlegend=False, row=i, col=1)
    return _style(fig, "STL decomposition", 640)


def cycle_fig(cyc: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18, subplot_titles=["HP-filter cycle component", "Periodogram"])
    c = cyc["cycle"]
    fig.add_scatter(x=c.index, y=c * 100, line=dict(color=PALETTE["primary"]), showlegend=False, row=1, col=1)
    fig.add_scatter(x=cyc["peaks"], y=c.reindex(cyc["peaks"]) * 100, mode="markers", name="Peak", marker=dict(color=PALETTE["up"], size=9), row=1, col=1)
    fig.add_scatter(x=cyc["troughs"], y=c.reindex(cyc["troughs"]) * 100, mode="markers", name="Trough", marker=dict(color=PALETTE["down"], size=9), row=1, col=1)
    fig.add_scatter(x=cyc["periods"], y=cyc["power"], line=dict(color=PALETTE["purple"]), showlegend=False, row=2, col=1)
    fig.update_xaxes(title_text="period (trading days)", row=2, col=1)
    fig.update_yaxes(title_text="% deviation from trend", row=1, col=1)
    return _style(fig, "Possible market cycles", 560)


def residual_fig(resid: pd.Series, z: pd.Series, flagged: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=["Residuals over time", "Distribution"])
    fig.add_scatter(x=resid.index, y=resid, mode="lines", line=dict(color=PALETTE["neutral"], width=1), showlegend=False, row=1, col=1)
    fig.add_scatter(x=flagged.index, y=resid.reindex(flagged.index), mode="markers", name="|z| above threshold",
                    marker=dict(color=PALETTE["down"], size=8), row=1, col=1)
    fig.add_histogram(x=resid, nbinsx=50, marker_color=PALETTE["primary"], showlegend=False, row=1, col=2)
    return _style(fig, "Unexplained (residual) variation", 380)


def regime_fig(level: pd.Series, labels: pd.Series, train_end) -> go.Figure:
    fig = go.Figure(go.Scatter(x=level.index, y=level, line=dict(color="#94A3B8", width=1), name="Index"))
    for i, name in enumerate(sorted(labels.unique())):
        s = labels[labels == name].index
        fig.add_scatter(x=s, y=level.reindex(s), mode="markers", name=name, marker=dict(size=5, color=SEQ[i % len(SEQ)]))
    fig.add_vline(x=train_end, line_dash="dash", line_color=PALETTE["neutral"], annotation_text="model fitted up to here")
    return _style(fig, "Market regimes (Gaussian mixture on rolling return & volatility)", 420)


# ----------------------------------------------------------------------------- prediction
def pred_vs_actual_scatter(y: np.ndarray, p: np.ndarray, lo=None, hi=None, unit: str = "%") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=p, y=y, mode="markers", marker=dict(size=5, color=PALETTE["primary"], opacity=0.5), name="stocks")
    m = float(np.nanmax(np.abs(np.r_[y, p])))
    fig.add_scatter(x=[-m, m], y=[-m, m], mode="lines", line=dict(dash="dash", color=PALETTE["neutral"]), name="perfect")
    fig.update_layout(xaxis_title=f"predicted ({unit})", yaxis_title=f"actual ({unit})")
    return _style(fig, "Predicted vs actual (held-out test set)", 400)


def pred_vs_actual_timeline(dates, actual, pred, lo, hi, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=dates, y=hi, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=dates, y=lo, fill="tonexty", fillcolor="rgba(37,99,235,0.15)", line=dict(width=0), name="prediction interval")
    fig.add_scatter(x=dates, y=actual, name="Actual", line=dict(color=PALETTE["down"]))
    fig.add_scatter(x=dates, y=pred, name="Predicted", line=dict(color=PALETTE["primary"]))
    return _style(fig, title, 400)


def metric_bars(metrics: pd.DataFrame, col: str, title: str) -> go.Figure:
    colors = [PALETTE["neutral"] if m.startswith("Baseline") else PALETTE["primary"] for m in metrics["model"]]
    fig = go.Figure(go.Bar(x=metrics["model"], y=metrics[col], marker_color=colors))
    return _style(fig, title, 320)


def importance_bar(imp: pd.DataFrame) -> go.Figure:
    fig = px.bar(imp.iloc[::-1], x="rmse_increase", y="feature", orientation="h", error_x="std",
                 color_discrete_sequence=[PALETTE["primary"]])
    fig.update_layout(xaxis_title="RMSE increase when shuffled")
    return _style(fig, "Permutation importance (held-out data)", max(300, 40 * len(imp) + 100))


def pdp_fig(pdp: pd.DataFrame) -> go.Figure:
    f = pdp.columns[0]
    fig = px.line(pdp, x=f, y="avg_prediction", markers=True, color_discrete_sequence=[PALETTE["purple"]])
    return _style(fig, f"Partial dependence: {f}", 340)


def contribution_bar(contrib: pd.DataFrame, title: str = "Why this prediction? (occlusion)") -> go.Figure:
    colors = np.where(contrib["contribution"] >= 0, PALETTE["up"], PALETTE["down"])
    fig = go.Figure(go.Bar(x=contrib["contribution"][::-1], y=contrib["feature"][::-1], orientation="h", marker_color=colors[::-1]))
    fig.update_layout(xaxis_title="effect on predicted return (pct-points)")
    return _style(fig, title, max(300, 40 * len(contrib) + 100))


def equity_curve(bt: pd.DataFrame) -> go.Figure:
    eq = (1 + bt).cumprod() * 100
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq["strategy"], name="Model top-quintile (after costs)", line=dict(color=PALETTE["primary"]))
    fig.add_scatter(x=eq.index, y=eq["benchmark"], name="Equal-weight benchmark", line=dict(color=PALETTE["neutral"]))
    return _style(fig, "Walk-forward backtest (out-of-sample, base = 100)", 400)


def decile_bars(spreads: dict) -> go.Figure:
    names = list(spreads)
    fig = go.Figure()
    fig.add_bar(x=names, y=[spreads[n]["top"] for n in names], name="Top decile by prediction", marker_color=PALETTE["up"])
    fig.add_bar(x=names, y=[spreads[n]["bottom"] for n in names], name="Bottom decile", marker_color=PALETTE["down"])
    fig.update_layout(barmode="group", yaxis_title="realised intraday return %")
    return _style(fig, "Out-of-fold ranking test: do predictions order realised returns?", 360)


# --------------------------------------------------------------------------- intelligence
def cluster_scatter(d: pd.DataFrame) -> go.Figure:
    fig = px.scatter(d, x="pc1", y="pc2", color="cluster", hover_name="symbol",
                     hover_data={"ret_pct": ":.2f", "range_pct": ":.2f", "pc1": False, "pc2": False},
                     color_discrete_sequence=SEQ, opacity=0.75)
    ev = d.attrs.get("explained_variance", [np.nan, np.nan])
    fig.update_layout(xaxis_title=f"PC1 ({ev[0]*100:.0f}% var)", yaxis_title=f"PC2 ({ev[1]*100:.0f}% var)")
    return _style(fig, "Stock map (PCA of behaviour features)", 480)


def silhouette_fig(sc: pd.DataFrame) -> go.Figure:
    fig = px.line(sc, x="k", y="silhouette", markers=True, color_discrete_sequence=[PALETTE["primary"]])
    return _style(fig, "Silhouette score by number of clusters", 300)


def profile_heatmap(profile: pd.DataFrame) -> go.Figure:
    p = profile.drop(columns=["n_stocks"])
    z = (p - p.mean()) / p.std(ddof=0).replace(0, 1)
    fig = px.imshow(z.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                    labels=dict(color="z-score"), x=[f"Cluster {c}" for c in z.index])
    return _style(fig, "Cluster profiles (median feature vs overall, z-score)", 380)


def anomaly_scatter(d: pd.DataFrame, x: str = "ret_pct", y: str = "range_pct") -> go.Figure:
    fig = px.scatter(d, x=x, y=y, color="is_anomaly", hover_name="symbol", hover_data=["main_driver", "anomaly_score"],
                     color_discrete_map={True: PALETTE["down"], False: "#CBD5E1"}, opacity=0.8)
    return _style(fig, "Cross-sectional anomalies (Isolation Forest)", 440)


def corr_heatmap(c: pd.DataFrame) -> go.Figure:
    fig = px.imshow(c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return _style(fig, "Return correlation", 440)


def scenario_hist(base: np.ndarray, scen: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=base, name="Baseline predictions", opacity=0.6, marker_color=PALETTE["neutral"])
    fig.add_histogram(x=scen, name="Scenario predictions", opacity=0.6, marker_color=PALETTE["primary"])
    fig.update_layout(barmode="overlay", xaxis_title="predicted intraday return %")
    return _style(fig, "Scenario: distribution of predicted returns across stocks", 360)
