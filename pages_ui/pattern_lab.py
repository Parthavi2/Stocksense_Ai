"""Page 2 - Pattern Mining Lab: Apriori / FP-Growth, association rules, rule validation."""
import pandas as pd
import streamlit as st

from stocksense import config as C
from stocksense import mining as M
from stocksense import visualization as V
from stocksense.features import ALL_GROUPS, ItemBinner, chronological_split, holdout_split_by_symbol
from stocksense.storage import ResultStore

from .common import Context, daily_features, download, kpi


@st.cache_data(show_spinner="Building transactions...")
def build_transactions(_df: pd.DataFrame, key: str, groups: tuple, lag: bool, split_mode: str, split_param: float):
    """Fit item cut-points on the TRAIN slice only, then encode train and test with them."""
    d = daily_features(_df, key)
    if split_mode == "chronological":
        train, test = chronological_split(d, split_param)
    else:
        train, test = holdout_split_by_symbol(d, split_param)
    binner = ItemBinner().fit(train)
    if lag:   # lag items need full history per symbol, so encode everything then slice
        full = binner.transform(d.sort_values(["symbol", "date"]), groups, lag=True)
        order = d.sort_values(["symbol", "date"]).reset_index(drop=True)
        order = order.loc[full.index]
        tr_mask = order["date"].isin(train["date"].unique()).to_numpy()
        return full[tr_mask].reset_index(drop=True), full[~tr_mask].reset_index(drop=True), binner.cuts_
    return binner.transform(train, groups).reset_index(drop=True), binner.transform(test, groups).reset_index(drop=True), binner.cuts_


@st.cache_data(show_spinner="Mining frequent itemsets...")
def mine_cached(_onehot: pd.DataFrame, data_key: str, algorithm: str, min_support: float, max_len: int):
    return M.mine_itemsets(_onehot, algorithm, min_support, max_len)


def render(ctx: Context):
    st.header("Pattern Mining Lab")
    st.caption("Each stock-session is a *transaction*; its discretised behaviours (return, gap, range, liquidity, ...) are *items*. "
               "Apriori / FP-Growth find item combinations that occur together often; association rules express them as A -> B.")

    with st.sidebar:
        st.subheader("Mining controls")
        algo = st.selectbox("Algorithm", ["FP-Growth", "Apriori", "Apriori (from scratch)"], key="pm_algo")
        min_sup = st.slider("Min support", 0.01, 0.50, 0.05, 0.01, key="pm_sup", help="Share of transactions containing the itemset")
        min_conf = st.slider("Min confidence", 0.10, 1.00, 0.60, 0.05, key="pm_conf", help="P(consequent | antecedent)")
        min_lift = st.slider("Min lift", 0.5, 5.0, 1.2, 0.1, key="pm_lift", help="Lift > 1: items co-occur more than by chance")
        max_len = st.slider("Max itemset length", 2, 5, 3, key="pm_len")
        groups = st.multiselect("Item groups", ALL_GROUPS, default=ALL_GROUPS, key="pm_groups")
        lag = st.checkbox("Add previous-session items", value=False, disabled=not ctx.multi_day, key="pm_lag",
                          help="Needs 2+ days. Lets rules say 'yesterday's pattern -> today's pattern'.")

    if len(groups) < 2:
        st.warning("Select at least two item groups."); return

    if ctx.n_days >= C.MIN_DAYS["rule_validation_temporal"]:
        frac = st.slider("Share of earliest days used for mining (rest = later-period validation)", 0.5, 0.9, 0.7, 0.05, key="pm_split")
        mode, param = "chronological", frac
        st.info("Validation: rules are mined on the earliest days and re-measured on the **later** days (true out-of-time test).")
    else:
        mode, param = "symbol", 0.3
        st.info("Validation: with fewer than 20 trading days a time split is not meaningful. Rules are mined on a random 70% of stocks and re-measured "
                "on the other 30%. This is a **cross-sectional hold-out, not a later-period test**.")

    try:
        train_oh, test_oh, cuts = build_transactions(ctx.df, ctx.key, tuple(groups), lag, mode, param)
        itemsets, secs = mine_cached(train_oh, f"{ctx.key}|{groups}|{lag}|{mode}|{param}", algo, min_sup, max_len)
        rules = M.generate_rules(itemsets, len(train_oh), min_conf, min_lift)
    except M.MiningError as e:
        st.error(str(e)); return
    except Exception as e:  # keep the app alive on unexpected input
        st.error(f"Mining failed: {e}"); return

    kpi(st.columns(5), [("Train transactions", f"{len(train_oh):,}"), ("Validation transactions", f"{len(test_oh):,}"),
                        ("Frequent itemsets", f"{len(itemsets):,}"), ("Rules", f"{len(rules):,}"), ("Mining time", f"{secs * 1000:.0f} ms")])
    t1, t2, t3, t4 = st.tabs(["Frequent itemsets", "Association rules", "Validation on unseen data", "Algorithm comparison"])

    with t1:
        if itemsets.empty:
            st.warning("No itemsets at this support. Lower the minimum support.")
        else:
            lens = st.multiselect("Itemset length", sorted(itemsets["length"].unique()), default=sorted(itemsets["length"].unique()), key="pm_lens")
            view = itemsets[itemsets["length"].isin(lens)]
            st.plotly_chart(V.itemset_bar(view), width="stretch")
            show = view.assign(itemsets=view["itemsets"].map(M.fmt_items))
            st.dataframe(show, hide_index=True, width="stretch")
            download(show, "Download itemsets (CSV)", "stocksense_itemsets.csv")

    with t2:
        if rules.empty:
            st.warning("No rules meet the thresholds. Lower confidence / lift or support.")
        else:
            q = st.text_input("Filter rules containing item", "", key="pm_rq")
            rv = rules[rules["rule"].str.contains(q, case=False, regex=False)] if q else rules
            st.plotly_chart(V.rules_scatter(rv), width="stretch")
            st.dataframe(M.rules_for_display(rv), hide_index=True, width="stretch")
            download(M.rules_for_display(rv), "Download rules (CSV)", "stocksense_rules.csv")
            if st.button("Save this run to local database", key="pm_save"):
                rid = ResultStore().save("association_rules", {"algorithm": algo, "min_support": min_sup, "min_confidence": min_conf,
                                                               "min_lift": min_lift, "groups": groups, "validation": mode},
                                         M.rules_for_display(rules))
                st.success(f"Saved as run #{rid}")
        st.caption("Reading the rules: items are measured over the **same** session, so most rules describe how a day looks "
                   "(e.g. closing near the high goes with closing above VWAP). They are co-occurrences, not causes, and are not trading signals "
                   "unless previous-session items are enabled and validated on later data.")

    with t3:
        if rules.empty:
            st.info("Generate rules first.")
        else:
            ev = M.evaluate_rules(rules, test_oh, min_conf)
            kpi(st.columns(3), [("Rules still holding", f"{ev['holds'].mean():.0%}", "test confidence >= min confidence and lift > 1"),
                                ("Mean confidence change", f"{ev['conf_change'].mean():+.3f}"),
                                ("Mean |change|", f"{ev['conf_change'].abs().mean():.3f}")])
            st.plotly_chart(V.rule_validation_fig(ev), width="stretch")
            cols = ["rule", "support", "confidence", "lift", "test_support", "test_confidence", "test_lift", "conf_change", "holds"]
            st.dataframe(ev[cols], hide_index=True, width="stretch")
            download(ev[cols], "Download validation (CSV)", "stocksense_rule_validation.csv")

    with t4:
        st.write("Runs all three implementations on the current training transactions and checks they find identical itemsets.")
        if st.button("Run comparison", key="pm_cmp"):
            rows, sets = [], {}
            for a in ["Apriori", "FP-Growth", "Apriori (from scratch)"]:
                res, t = M.mine_itemsets(train_oh, a, min_sup, max_len)
                rows.append({"algorithm": a, "itemsets": len(res), "seconds": round(t, 4)})
                sets[a] = {(frozenset(i), round(s, 9)) for i, s in zip(res["itemsets"], res["support"])}
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            same = sets["Apriori"] == sets["FP-Growth"] == sets["Apriori (from scratch)"]
            (st.success if same else st.error)("All three algorithms return identical itemsets and supports." if same else "Results differ - please report.")
