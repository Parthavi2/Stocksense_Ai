"""Market intelligence: anomalies, clustering/similarity, breadth regime, shock scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from . import config as C

CLUSTER_FEATURES = ["ret_pct", "gap_pct", "intraday_pct", "range_pct", "close_pos",
                    "vwap_dev_pct", "log_turnover", "log_trades", "log_price"]
ANOMALY_FEATURES = ["ret_pct", "gap_pct", "intraday_pct", "range_pct", "vwap_dev_pct",
                    "log_turnover", "log_trades"]


def _scaled(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Median/IQR scaling, then clipping at +-5 so one extreme stock cannot dominate distances."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    Z = np.clip(RobustScaler().fit_transform(d[features]), -5, 5)
    return d, Z


# ---------------------------------------------------------------------------------- anomalies
def detect_anomalies(df: pd.DataFrame, features: list[str] = ANOMALY_FEATURES, contamination: float = 0.02,
                     seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Isolation Forest on median/IQR-scaled features (not clipped, so true outliers stay visible).
    Each row also gets the feature that deviates most (robust z) as an interpretable 'reason'."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features).copy()
    Z = RobustScaler().fit_transform(d[features])
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(Z)
    d["anomaly_score"] = -iso.score_samples(Z)                   # higher = more unusual
    d["is_anomaly"] = iso.predict(Z) == -1
    top = np.abs(Z).argmax(axis=1)
    d["main_driver"] = [features[i] for i in top]
    d["driver_robust_z"] = Z[np.arange(len(Z)), top]
    return d


# ---------------------------------------------------------------------------------- clustering
def choose_k(df: pd.DataFrame, features: list[str] = CLUSTER_FEATURES, k_range=range(2, 9),
             seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    d, Z = _scaled(df, features)
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=5, random_state=seed).fit(Z)
        rows.append((k, silhouette_score(Z, km.labels_, sample_size=min(3000, len(Z)), random_state=seed), km.inertia_))
    return pd.DataFrame(rows, columns=["k", "silhouette", "inertia"])


def cluster_stocks(df: pd.DataFrame, k: int, features: list[str] = CLUSTER_FEATURES, seed: int = C.RANDOM_STATE):
    """K-Means on robust-scaled features + 2-D PCA map. Returns (frame with cluster/pc1/pc2, profile)."""
    d, Z = _scaled(df, features)
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    pca = PCA(2, random_state=seed).fit(Z)
    P = pca.transform(Z)
    d = d.copy()
    d["cluster"] = km.labels_.astype(str)
    d["pc1"], d["pc2"] = P[:, 0], P[:, 1]
    profile = d.groupby("cluster")[features].median()
    profile["n_stocks"] = d.groupby("cluster").size()
    d.attrs["explained_variance"] = pca.explained_variance_ratio_.tolist()
    return d, profile


def similar_stocks(df: pd.DataFrame, symbol: str, features: list[str] = CLUSTER_FEATURES, n: int = 10) -> pd.DataFrame:
    """Nearest neighbours in robust-scaled feature space (Euclidean)."""
    d, Z = _scaled(df, features)
    d = d.reset_index(drop=True)
    if symbol not in set(d["symbol"]):
        raise KeyError(f"{symbol} has no complete feature data.")
    i = int(d.index[d["symbol"] == symbol][0])
    nn = NearestNeighbors(n_neighbors=min(n + 1, len(d))).fit(Z)
    dist, idx = nn.kneighbors(Z[[i]])
    out = d.iloc[idx[0][1:]][["symbol", "name"] + features].copy() if "name" in d else d.iloc[idx[0][1:]][["symbol"] + features].copy()
    out.insert(1, "distance", dist[0][1:])
    return out.reset_index(drop=True)


def return_correlations(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Correlation of daily returns between symbols (needs several trading days)."""
    d = df[df["symbol"].isin(symbols)].copy()
    d["r"] = d["close"] / d["prev_close"] - 1
    return d.pivot_table(index="date", columns="symbol", values="r").corr()


# ------------------------------------------------------------------------------ breadth regime
def breadth_regime(df: pd.DataFrame) -> dict:
    """Rule-based, transparent snapshot of today's market state from cross-sectional breadth.
    Not a statistical regime model (that needs history) - it is a description of one day."""
    r = (df["close"] / df["prev_close"] - 1) * 100
    adv, dec = int((r > 0).sum()), int((r < 0).sum())
    ad_ratio = adv / dec if dec else np.inf
    w = df["turnover"] / df["turnover"].sum()
    out = {"advancers": adv, "decliners": dec, "unchanged": int((r == 0).sum()),
           "ad_ratio": float(ad_ratio), "median_ret_pct": float(r.median()),
           "turnover_weighted_ret_pct": float((w * r).sum()), "dispersion_pct": float(r.std()),
           "pct_up_2": float((r >= 2).mean() * 100), "pct_down_2": float((r <= -2).mean() * 100)}
    if ad_ratio >= 1.5 and out["median_ret_pct"] > 0:
        out["regime"] = "Risk-on (broad advance)"
    elif ad_ratio <= 1 / 1.5 and out["median_ret_pct"] < 0:
        out["regime"] = "Risk-off (broad decline)"
    else:
        out["regime"] = "Mixed / rotational"
    out["rule"] = "Risk-on if advancers/decliners >= 1.5 and median return > 0; risk-off if <= 0.67 and median < 0; else mixed."
    return out


# ------------------------------------------------------------------------- market-shock what-if
def shock_impact(betas: pd.DataFrame, shock_pct: float, weights: pd.Series | None = None) -> pd.DataFrame:
    """Expected stock move = beta x market shock (linear single-factor model). Needs betas from
    several days of history. Ignores non-linear and tail behaviour - an approximation."""
    out = betas.copy()
    out["expected_move_pct"] = out["beta"] * shock_pct
    return out.sort_values("expected_move_pct")


# ------------------------------------------------------------------ historical (time) anomalies
def stock_time_anomalies(df: pd.DataFrame, symbol: str, z_threshold: float = 3.5) -> pd.DataFrame:
    """Flags days where a stock's return, range or volume was extreme *relative to its own history*
    (robust z-score = (x - median) / (1.4826 * MAD)). Requires several days of data."""
    from .features import add_daily_features
    d = add_daily_features(df[df["symbol"] == symbol].sort_values("date")).set_index("date")
    if len(d) < 30:
        raise ValueError("Need at least 30 trading days of this stock to flag historical anomalies.")
    out = pd.DataFrame(index=d.index)
    for name, col in [("return", "ret_pct"), ("range", "range_pct"), ("volume", "log_volume")]:
        x = d[col]
        mad = (x - x.median()).abs().median() * 1.4826
        out[f"z_{name}"] = (x - x.median()) / (mad if mad > 0 else x.std())
    out["max_abs_z"] = out.abs().max(axis=1)
    out["driver"] = out[["z_return", "z_range", "z_volume"]].abs().idxmax(axis=1).str.replace("z_", "")
    out["is_anomaly"] = out["max_abs_z"] > z_threshold
    return out
