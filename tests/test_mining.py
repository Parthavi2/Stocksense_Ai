import numpy as np
import pandas as pd
import pytest

from stocksense import mining as M

# 5 transactions; supports are easy to verify by hand
TX = pd.DataFrame({"A": [1, 1, 1, 0, 1], "B": [1, 1, 0, 1, 1], "C": [1, 0, 1, 0, 0], "D": [0, 1, 0, 0, 1]}).astype(bool)


def _as_set(df):
    return {(frozenset(i), round(s, 9)) for i, s in zip(df["itemsets"], df["support"])}


def test_known_supports():
    res, _ = M.mine_itemsets(TX, "Apriori", 0.4, None)
    d = {frozenset(i): s for i, s in zip(res["itemsets"], res["support"])}
    assert d[frozenset("A")] == pytest.approx(0.8) and d[frozenset("AB")] == pytest.approx(0.6)
    assert frozenset("CD") not in d                       # support 0 -> absent


@pytest.mark.parametrize("sup", [0.2, 0.4, 0.6])
def test_three_implementations_agree(sup):
    a, _ = M.mine_itemsets(TX, "Apriori", sup, None)
    f, _ = M.mine_itemsets(TX, "FP-Growth", sup, None)
    s, _ = M.mine_itemsets(TX, "Apriori (from scratch)", sup, None)
    assert _as_set(a) == _as_set(f) == _as_set(s)


def test_agreement_on_random_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((300, 12)) < np.linspace(0.1, 0.6, 12)).astype(bool)
    X.columns = [f"i{k}" for k in range(12)]
    a, _ = M.mine_itemsets(X, "Apriori", 0.05, 4); s, _ = M.mine_itemsets(X, "Apriori (from scratch)", 0.05, 4)
    f, _ = M.mine_itemsets(X, "FP-Growth", 0.05, 4)
    assert _as_set(a) == _as_set(s) == _as_set(f)


def test_higher_support_never_gives_more_itemsets():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((200, 8)) < 0.4)
    n = [len(M.mine_itemsets(X, "FP-Growth", s, 3)[0]) for s in (0.05, 0.1, 0.2, 0.4)]
    assert n == sorted(n, reverse=True)


def test_max_len_respected():
    res, _ = M.mine_itemsets(TX, "FP-Growth", 0.2, 2)
    assert res["length"].max() <= 2


def test_rule_metrics_match_hand_calculation():
    res, _ = M.mine_itemsets(TX, "FP-Growth", 0.2, None)
    rules = M.generate_rules(res, len(TX), min_confidence=0.0, min_lift=0.0)
    r = rules[(rules.antecedents == frozenset("C")) & (rules.consequents == frozenset("A"))].iloc[0]
    # P(C)=0.4, P(A)=0.8, P(A&C)=0.4 -> conf = 1.0, lift = 1/0.8
    assert r.support == pytest.approx(0.4) and r.confidence == pytest.approx(1.0) and r.lift == pytest.approx(1.25)
    assert M.measure_rule(TX, ["C"], ["A"]) == pytest.approx((0.4, 1.0, 1.25))


def test_confidence_and_lift_thresholds_filter():
    res, _ = M.mine_itemsets(TX, "FP-Growth", 0.2, None)
    lo = M.generate_rules(res, len(TX), 0.0, 0.0); hi = M.generate_rules(res, len(TX), 0.9, 1.1)
    assert len(hi) < len(lo) and (hi.confidence >= 0.9).all() and (hi.lift >= 1.1).all()


def test_evaluate_rules_on_unseen_data():
    res, _ = M.mine_itemsets(TX, "FP-Growth", 0.2, None)
    rules = M.generate_rules(res, len(TX), 0.7, 1.0)
    ev = M.evaluate_rules(rules, TX, 0.7)
    assert np.allclose(ev["test_confidence"], ev["confidence"]) and ev["holds"].all()   # same data -> identical
    flipped = ~TX
    assert M.evaluate_rules(rules, flipped, 0.7)["holds"].mean() < 1


def test_bad_inputs():
    with pytest.raises(M.MiningError):
        M.mine_itemsets(TX.iloc[0:0], "Apriori", 0.1)
    with pytest.raises(M.MiningError):
        M.mine_itemsets(TX, "Apriori", 0)
    with pytest.raises(M.MiningError):
        M.mine_itemsets(TX, "Nope", 0.1)
