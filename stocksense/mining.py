"""Frequent itemset mining and association rules.

* `mine_itemsets`       - Apriori or FP-Growth (MLxtend) with adjustable min-support.
* `apriori_scratch`     - a small, readable from-scratch Apriori used to cross-check MLxtend
                          (see tests) and for teaching.
* `generate_rules`      - rules filtered by confidence and lift.
* `evaluate_rules`      - re-measures the *same* rules on unseen transactions (later period or
                          held-out stocks) to see which rules survive.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

MAX_ITEMSETS = 300_000


class MiningError(Exception):
    pass


def _check_input(onehot: pd.DataFrame, min_support: float):
    if onehot is None or onehot.empty:
        raise MiningError("No transactions to mine.")
    if not 0 < min_support <= 1:
        raise MiningError("min_support must be in (0, 1].")


def apriori_scratch(onehot: pd.DataFrame, min_support: float, max_len: int | None = None) -> pd.DataFrame:
    """Level-wise Apriori: join frequent (k-1)-itemsets, prune candidates that have an
    infrequent subset, then count support with boolean column ANDs."""
    _check_input(onehot, min_support)
    X = onehot.to_numpy(dtype=bool)
    names = list(onehot.columns)
    n = len(X)
    sup1 = X.mean(axis=0)
    level = {(i,): float(sup1[i]) for i in range(len(names)) if sup1[i] >= min_support}
    frequent = dict(level)
    k = 2
    while level and (max_len is None or k <= max_len):
        keys = sorted(level)
        prev = set(keys)
        candidates = []
        for a, b in combinations(keys, 2):
            if a[:-1] == b[:-1]:                                   # join step (a < b by sort order)
                c = a + (b[-1],)
                if all(sub in prev for sub in combinations(c, k - 1)):   # prune step
                    candidates.append(c)
        level = {}
        for c in candidates:
            s = float(X[:, list(c)].all(axis=1).sum() / n)
            if s >= min_support:
                level[c] = s
        frequent.update(level)
        k += 1
    return pd.DataFrame({"support": list(frequent.values()),
                         "itemsets": [frozenset(names[i] for i in key) for key in frequent]})


def mine_itemsets(onehot: pd.DataFrame, algorithm: str = "FP-Growth", min_support: float = 0.05,
                  max_len: int | None = 4) -> tuple[pd.DataFrame, float]:
    """Returns (itemsets DataFrame[support, itemsets, length], runtime_seconds)."""
    _check_input(onehot, min_support)
    t0 = time.perf_counter()
    if algorithm == "Apriori":
        res = apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    elif algorithm == "FP-Growth":
        res = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    elif algorithm == "Apriori (from scratch)":
        res = apriori_scratch(onehot, min_support, max_len)
    else:
        raise MiningError(f"Unknown algorithm: {algorithm}")
    elapsed = time.perf_counter() - t0
    if len(res) > MAX_ITEMSETS:
        raise MiningError(f"{len(res):,} itemsets found - raise min support or lower max length.")
    res = res.copy()
    res["length"] = res["itemsets"].map(len)
    return res.sort_values("support", ascending=False).reset_index(drop=True), elapsed


def generate_rules(itemsets: pd.DataFrame, n_transactions: int, min_confidence: float = 0.5,
                   min_lift: float = 1.0, max_rules: int = 5000) -> pd.DataFrame:
    """Association rules A -> B with confidence >= min_confidence and lift >= min_lift."""
    cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    if itemsets.empty or (itemsets["length"] < 2).all():
        return pd.DataFrame(columns=cols + ["rule"])
    rules = association_rules(itemsets[["support", "itemsets"]], num_itemsets=n_transactions,
                              metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift][cols].copy()
    rules["conviction"] = rules["conviction"].replace(np.inf, np.nan)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(max_rules).reset_index(drop=True)
    rules["rule"] = rules.apply(lambda r: f"{fmt_items(r.antecedents)}  ->  {fmt_items(r.consequents)}", axis=1)
    return rules


def fmt_items(items) -> str:
    return " + ".join(sorted(items))


def measure_rule(onehot: pd.DataFrame, antecedent, consequent) -> tuple[float, float, float]:
    """(support, confidence, lift) of one rule on a transaction table."""
    n = len(onehot)
    a = onehot[list(antecedent)].all(axis=1).to_numpy()
    c = onehot[list(consequent)].all(axis=1).to_numpy()
    sa, sc, sac = a.mean(), c.mean(), (a & c).mean()
    conf = sac / sa if sa > 0 else np.nan
    lift = conf / sc if sc > 0 and not np.isnan(conf) else np.nan
    return float(sac), float(conf), float(lift)


def evaluate_rules(rules: pd.DataFrame, onehot_test: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Re-measure each rule on unseen data. A rule 'holds' if test confidence >= min_confidence
    and test lift > 1. Columns: test_support, test_confidence, test_lift, conf_change, holds."""
    if rules.empty:
        return rules.assign(test_support=[], test_confidence=[], test_lift=[], conf_change=[], holds=[])
    m = [measure_rule(onehot_test, r.antecedents, r.consequents) for r in rules.itertuples()]
    out = rules.copy()
    out["test_support"], out["test_confidence"], out["test_lift"] = zip(*m)
    out["conf_change"] = out["test_confidence"] - out["confidence"]
    out["holds"] = (out["test_confidence"] >= min_confidence) & (out["test_lift"] > 1)
    return out


def rules_for_display(rules: pd.DataFrame) -> pd.DataFrame:
    d = rules.copy()
    d["antecedents"] = d["antecedents"].map(fmt_items)
    d["consequents"] = d["consequents"].map(fmt_items)
    return d.drop(columns=[c for c in ["rule"] if c in d.columns])
