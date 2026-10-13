"""Page 4 - Prediction Lab: forecasting, explainability, backtesting/model comparison, what-if."""
import numpy as np
import pandas as pd
import streamlit as st

from stocksense import config as C
from stocksense import evaluation as E
from stocksense import prediction as P
from stocksense import timeseries as T
from stocksense import visualization as V
from stocksense.features import TEMPORAL_FEATURES
from stocksense.market_intel import shock_impact

from .common import Context, daily_features, download, kpi

ML_MODELS = ["Ridge", "Random Forest", "Gradient Boosting"]


# ------------------------------------------------------------------------------- cached work
@st.cache_data(show_spinner="Preparing intraday dataset...")
def _intraday(_df, key):
    return P.prepare_intraday(daily_features(_df, key))


@st.cache_data(show_spinner="Preparing panel features...")
def _panel(_df, key, horizon, top_n):
    return P.prepare_panel(_df, horizon, top_n)


@st.cache_data(show_spinner="Training models...")
def _fit(_data, key, features, target, alpha, mode, horizon):
    X, y = _data[list(features)], _data[target]
    masks = (P.symbol_masks(_data["symbol"]) if mode == "intraday"
             else P.chronological_masks(_data["date"], horizon))
    res = P.fit_and_evaluate(X, y, masks, alpha)
    res["masks"] = masks
    return res


@st.cache_data(show_spinner="Running cross-validation...")
def _cv(_data, key, features, target):
    return P.cross_sectional_cv(_data[list(features)], _data[target])


@st.cache_data(show_spinner="Running walk-forward backtest...")
def _wf(_data, key, features, horizon, n_splits):
    return P.walk_forward(_data, list(features), horizon, n_splits)


# ------------------------------------------------------------------------------------- page
def render(ctx: Context):
    st.header("Prediction Lab")
    intraday = not ctx.enough("forecasting")
    if intraday:
        horizon = 0
        st.warning(f"**Intraday cross-sectional mode.** The data has {ctx.n_days} trading day(s); forecasting a future date needs "
                   f"at least {C.MIN_DAYS['forecasting']}. Instead the lab predicts each stock's **open-to-close return of the observed day** "
                   "from information available at the open (overnight gap and price level), training on some stocks and testing on **other, unseen stocks**. "
                   "This is a cross-sectional experiment - it is **not** a forecast of any future date, and relationships found on one day (for example how gaps "
                   "behaved on a strongly trending session) may not hold on other days.")
        data = _intraday(ctx.df, ctx.key)
        features, target, unit = P.INTRADAY_FEATURES, "intraday_pct", "%"
        mode = "intraday"
    else:
        c1, c2, c3 = st.columns(3)
        horizon = c1.selectbox("Forecast horizon (sessions)", [1, 5, 10, 21], index=1, key="pl_h")
        top_n = c2.slider("Universe: most liquid N stocks", 30, 300, 100, 10, key="pl_n")
        features, mode, target, unit = TEMPORAL_FEATURES, "panel", "target_pct", "% (log)"
        try:
            data = _panel(ctx.df, ctx.key, horizon, top_n)
        except Exception as e:
            st.error(f"Could not build features: {e}"); return
    alpha = st.sidebar.slider("Prediction interval miscoverage (alpha)", 0.05, 0.30, 0.10, 0.05, key="pl_alpha",
                              help="0.10 -> 90% prediction interval")
    try:
        res = _fit(data, f"{ctx.key}|{horizon}|{len(data)}", tuple(features), target, alpha, mode, horizon)
    except ValueError as e:
        st.error(str(e)); return
    tr, ca, te = res["masks"]
    kpi(st.columns(4), [("Train rows", f"{res['n_train']:,}"), ("Calibration rows", f"{res['n_cal']:,}"),
                        ("Test rows (never seen)", f"{res['n_test']:,}"), ("Features", f"{len(features)}")])

    # A radio instead of st.tabs: only the selected section is computed (tabs run all sections on every rerun).
    view = st.radio("Section", ["Predictions & evaluation", "Explainability", "Backtest & model comparison", "What-if simulation"],
                    horizontal=True, label_visibility="collapsed", key="pl_view")
    if view == "Predictions & evaluation":
        _evaluation(ctx, res, data, features, target, unit, mode, horizon, ca)
    elif view == "Explainability":
        _explain(res, data, features, target, mode)
    elif view == "Backtest & model comparison":
        _backtest(ctx, data, features, target, mode, horizon)
    else:
        _whatif(ctx, res, data, features, mode, horizon)


# ------------------------------------------------------------------------ tab 1: evaluation
def _evaluation(ctx, res, data, features, target, unit, mode, horizon, ca):
    m = res["metrics"].round(4)
    st.subheader("Model comparison on the held-out test set")
    st.dataframe(m, hide_index=True, width="stretch")
    best = m[~m["model"].str.startswith("Baseline")].iloc[0]
    if best["skill_vs_baseline_pct"] > 0:
        st.success(f"Best ML model: **{best['model']}** - RMSE {best['skill_vs_baseline_pct']:.2f}% lower than the 'no change' baseline "
                   f"(out-of-sample R2 vs baseline {best['R2_oos_vs_baseline']:.4f}). Small gains are normal and may not be economically meaningful.")
    else:
        st.warning(f"**No ML model beat the naive baseline on unseen data** (best: {best['model']}, {best['skill_vs_baseline_pct']:.2f}% vs baseline). "
                 "This is a common, honest outcome: returns are mostly noise, and the lab reports it rather than hiding it.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(V.metric_bars(m, "RMSE", "RMSE (lower is better; grey = baselines)"), width="stretch")
    c2.plotly_chart(V.metric_bars(m, "directional_accuracy", "Directional accuracy (0.5 = coin flip)"), width="stretch")

    ml = [n for n in res["test_pred"] if not n.startswith("Baseline")]
    name = st.selectbox("Model to inspect", ml, index=ml.index(best["model"]), key="pl_inspect")
    p, y = res["test_pred"][name], res["y_test"].to_numpy()
    hw = res["halfwidth"][name]
    cov = float(m.loc[m["model"] == name, "interval_coverage"].iloc[0])
    st.caption(f"{(1 - res['alpha']):.0%} prediction interval = prediction +/- {hw:.3f}{unit.split()[0]} (split-conformal, calibrated on separate data). "
               f"Observed coverage on the test set: {cov:.1%}.")
    if mode == "intraday":
        st.plotly_chart(V.pred_vs_actual_scatter(y, p), width="stretch")
        out = data.loc[res["test_index"], ["symbol", "name", "gap_pct", "close"]].assign(
            actual_pct=y, model_estimate_pct=p, lower_pct=p - hw, upper_pct=p + hw)
        st.caption("These are *model estimates for the observed session*, shown next to what actually happened.")
    else:
        t = data.loc[res["test_index"], ["date", "symbol"]].assign(actual=y, pred=p)
        daily = t.groupby("date")[["actual", "pred"]].mean()
        st.plotly_chart(V.pred_vs_actual_timeline(daily.index, daily["actual"], daily["pred"], daily["pred"] - hw, daily["pred"] + hw,
                                                  f"Average {horizon}-session forward return across stocks: predicted vs actual (test period, HISTORICAL)"),
                        width="stretch")
        st.plotly_chart(V.pred_vs_actual_scatter(y, p), width="stretch")
        out = t.assign(lower=p - hw, upper=p + hw)
        # ---- genuine future forecast
        st.subheader("Forecast for the next sessions (not yet observed)")
        st.warning("These values are **forecasts**, not observations. Expect them to be small, noisy and uncertain.")
        latest = P.prepare_latest(ctx.df, horizon, data["symbol"].nunique())
        fc = P.forecast_latest(data, latest, list(features), name, hw)
        fc["forecast_for"] = f"{horizon} sessions after {pd.Timestamp(fc['date'].iloc[0]).date()}"
        st.dataframe(fc.round(3), hide_index=True, width="stretch")
        download(fc, "Download forecasts (CSV)", "stocksense_forecasts.csv")
    download(out.round(4), "Download test predictions (CSV)", "stocksense_test_predictions.csv")


# ------------------------------------------------------------------------ tab 2: explanation
def _explain(res, data, features, target, mode):
    ml = [n for n in res["models"]]
    name = st.selectbox("Model", ml, index=len(ml) - 1, key="ex_model")
    model = res["models"][name]
    X = data.loc[res["test_index"], list(features)]
    y = data.loc[res["test_index"], target]
    st.caption("Explanations are computed on the held-out test set. They describe what the *model* uses, not what causes returns.")
    imp = P.importance_table(model, X, y, n_repeats=5)
    c1, c2 = st.columns(2)
    c1.plotly_chart(V.importance_bar(imp), width="stretch")
    if (imp["rmse_increase"] <= 0).all():
        c1.warning("Shuffling no feature hurts held-out accuracy: the model has found no usable signal.")
    f = c2.selectbox("Partial dependence for feature", list(features), key="ex_pdp")
    c2.plotly_chart(V.pdp_fig(P.partial_dependence_1d(model, X, f)), width="stretch")
    if name == "Ridge":
        coef = model.named_steps["ridge"].coef_
        st.write("Ridge standardised coefficients (effect of +1 std in each feature):")
        st.dataframe(pd.DataFrame({"feature": features, "coefficient": coef}).sort_values("coefficient", key=np.abs, ascending=False),
                     hide_index=True, width="stretch")
    st.subheader("Explain one prediction")
    info = data.loc[res["test_index"], ["symbol"] + (["date"] if mode == "panel" else [])].reset_index()
    label = info["symbol"] + (" | " + info["date"].astype(str) if mode == "panel" else "")
    pick = st.selectbox("Test observation", label.tolist(), key="ex_obs")
    idx = info["index"].iloc[label.tolist().index(pick)]
    row = data.loc[idx, list(features)].astype(float)
    ex = P.local_explanation(model, X, row)
    kpi(st.columns(3), [("Prediction (%)", f"{ex.attrs['prediction']:.3f}"), ("Typical-input prediction (%)", f"{ex.attrs['baseline_prediction']:.3f}"),
                        ("Actual (%)", f"{data.loc[idx, target]:.3f}")])
    st.plotly_chart(V.contribution_bar(ex), width="stretch")
    st.dataframe(ex.round(4), hide_index=True, width="stretch")


# ------------------------------------------------------------------- tab 3: backtest
def _backtest(ctx, data, features, target, mode, horizon):
    if mode == "intraday":
        st.info("With one trading day there is no time axis to walk forward on. This tab runs **5-fold out-of-fold validation across stocks** "
                "and a ranking test: if predictions carry signal, stocks ranked highest should realise higher returns than stocks ranked lowest. "
                "It is a model-comparison test, **not a trading backtest** (no costs, no execution, one session).")
        cv = _cv(data, f"{ctx.key}|{len(data)}", tuple(features), target)
        st.dataframe(cv["metrics"].round(4), hide_index=True, width="stretch")
        st.plotly_chart(V.decile_bars(cv["spreads"]), width="stretch")
        st.caption("Top/bottom = 10% of stocks by predicted return. A positive spread that is also stable is the minimum bar; a spread of ~0 means no ranking skill.")
        download(cv["metrics"], "Download CV results (CSV)", "stocksense_cv_results.csv")
        return
    if not ctx.enough("backtesting"):
        st.info(f"Walk-forward backtesting needs {C.MIN_DAYS['backtesting']} trading days; loaded: {ctx.n_days}."); return
    n_splits = st.slider("Walk-forward folds", 2, 8, 4, key="bt_folds")
    try:
        wf = _wf(data, f"{ctx.key}|{horizon}|{len(data)}", tuple(features), horizon, n_splits)
    except ValueError as e:
        st.error(str(e)); return
    folds = wf["folds"]
    summary = folds.groupby("model")[["MAE", "RMSE", "R2_oos_vs_baseline", "directional_accuracy", "skill_vs_baseline_pct"]].mean().reset_index().sort_values("RMSE")
    st.subheader("Average across walk-forward folds")
    st.dataframe(summary.round(4), hide_index=True, width="stretch")
    st.plotly_chart(V.px.line(folds, x="fold", y="RMSE", color="model", markers=True).update_layout(
        height=340, template="plotly_white", title="RMSE per fold (consistency matters more than one good fold)", margin=dict(t=50)), width="stretch")
    st.dataframe(folds.round(4), hide_index=True, width="stretch")
    choice = st.selectbox("Strategy model", [m for m in summary["model"] if not m.startswith("Baseline")], key="bt_model")
    q = st.slider("Hold top share of stocks", 0.05, 0.5, 0.2, 0.05, key="bt_q")
    cost = st.slider("Round-trip cost per rebalance (bps per side)", 0, 50, 10, key="bt_cost")
    bt = E.long_only_backtest(wf["predictions"].rename(columns={choice: "pred"}), horizon, q, cost)
    if bt.empty:
        st.info("Not enough data for the strategy backtest."); return
    ppy = 252 / horizon
    stats = pd.DataFrame({"Model strategy": E.performance_stats(bt["strategy"], ppy), "Equal-weight benchmark": E.performance_stats(bt["benchmark"], ppy)}).T
    st.plotly_chart(V.equity_curve(bt), width="stretch")
    st.dataframe(stats.round(2), width="stretch")
    st.caption("Out-of-sample only, rebalanced every horizon, simple costs; no slippage, taxes, liquidity limits or survivorship control. "
               "Treat as an evaluation of the signal, not an investable result.")
    download(bt.reset_index(), "Download backtest returns (CSV)", "stocksense_backtest.csv")


# --------------------------------------------------------------------------- tab 4: what-if
def _whatif(ctx, res, data, features, mode, horizon):
    ml = list(res["models"])
    name = st.selectbox("Model", ml, index=len(ml) - 1, key="wi_model")
    model, hw = res["models"][name], res["halfwidth"][name]
    X_all = data[list(features)]
    if mode == "intraday":
        st.write("**Single stock:** change the opening gap or price level and see the model's estimate move. "
                 "**Market scenario:** shift every stock's gap and see the distribution of estimates change.")
        sym = st.selectbox("Stock", sorted(data["symbol"]), key="wi_sym")
        row = data[data["symbol"] == sym].iloc[0]
        base = row[list(features)].astype(float)
        c1, c2 = st.columns(2)
        gap = c1.slider("Opening gap (%)", -10.0, 10.0, float(np.clip(base["gap_pct"], -10, 10)), 0.1, key="wi_gap")
        px_ = c2.number_input("Previous close (INR)", 0.1, 1e6, float(np.exp(base["log_prev_close"])), key="wi_px")
        r = P.what_if(model, X_all, base, {"gap_pct": gap, "log_prev_close": float(np.log(px_))}, hw)
        kpi(st.columns(3), [("Baseline estimate", f"{r['base']:.3f}%"), ("Scenario estimate", f"{r['scenario']:.3f}%", f"{1 - res['alpha']:.0%} interval: {r['interval'][0]:.2f}% to {r['interval'][1]:.2f}%"),
                            ("Change", f"{r['change']:+.3f} pp")])
        st.caption(f"Interval width (+/- {hw:.2f}%) is much larger than any scenario effect - the honest message of most what-if tests.")
        shift = st.slider("Market scenario: shift all gaps by (pp)", -3.0, 3.0, 1.0, 0.1, key="wi_shift")
        base_pred = model.predict(X_all)
        scen = X_all.copy(); scen["gap_pct"] += shift
        scen_pred = model.predict(scen)
        kpi(st.columns(3), [("Mean estimate, baseline", f"{base_pred.mean():.3f}%"), ("Mean estimate, scenario", f"{scen_pred.mean():.3f}%"),
                            ("Share of stocks estimated up", f"{(scen_pred > 0).mean():.1%}", f"baseline {(base_pred > 0).mean():.1%}")])
        st.plotly_chart(V.scenario_hist(base_pred, scen_pred), width="stretch")
        return
    # multi-day
    st.write("**Single stock:** edit the latest feature values. **Market shock:** apply a % move to the equal-weight market and see expected stock moves through each stock's beta.")
    latest = P.prepare_latest(ctx.df, horizon, data["symbol"].nunique())
    sym = st.selectbox("Stock", sorted(latest["symbol"]), key="wi_sym2")
    base = latest[latest["symbol"] == sym].iloc[0][list(features)].astype(float)
    f = st.multiselect("Features to edit", list(features), default=["mom5", "vol20"], key="wi_feats")
    over = {k: st.number_input(k, value=float(base[k]), format="%.5f", key=f"wi_{k}") for k in f}
    r = P.what_if(model, X_all, base, over, hw)
    kpi(st.columns(3), [("Baseline forecast", f"{r['base']:.3f}%"), ("Scenario forecast", f"{r['scenario']:.3f}%", f"interval {r['interval'][0]:.2f}% to {r['interval'][1]:.2f}%"),
                        ("Change", f"{r['change']:+.3f} pp")])
    shock = st.slider("Market shock (%)", -15.0, 15.0, -5.0, 0.5, key="wi_shock")
    betas = T.estimate_betas(ctx.df)
    imp = shock_impact(betas, shock)
    kpi(st.columns(2), [("Average expected move", f"{imp['expected_move_pct'].mean():.2f}%"), ("Most affected", f"{imp.iloc[0]['symbol']} ({imp.iloc[0]['expected_move_pct']:.2f}%)")])
    st.dataframe(imp.round(3), hide_index=True, width="stretch")
    st.caption("Single-factor linear approximation (beta x shock). Real crashes are non-linear and correlations rise in stress; treat as a rough guide.")
    download(imp, "Download shock impact (CSV)", "stocksense_shock_impact.csv")
