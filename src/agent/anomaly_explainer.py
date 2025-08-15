# src/agent/anomaly_explainer.py
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import zscore  # <-- class requirement

FEATURES: List[str] = ["cpu_usage", "memory_usage", "network_latency", "disk_io"]

# ---- Class-style z-score helpers ----
def compute_column_zscores(df: pd.DataFrame, features: List[str] = FEATURES) -> pd.DataFrame:
    """
    Per-column z-scores over the whole dataset (mean/std). Matches: numeric_data.apply(zscore)
    """
    numeric_data = df[features]
    z = numeric_data.apply(zscore)  # one column at a time
    return z

def find_anomalous_columns_for_row(row_index: int, z_scores: pd.DataFrame, threshold: float = 3.0) -> list:
    """
    Return list of feature names whose |z| > threshold for a given row index.
    """
    zs = z_scores.loc[row_index]
    return [col for col, val in zs.items() if pd.notna(val) and abs(val) > threshold]

# Optional alias to match your class function name exactly
def findAnomalousColumns(row_index: int, z_scores: pd.DataFrame, threshold: float = 3.0) -> list:
    return find_anomalous_columns_for_row(row_index, z_scores, threshold)

# ---- (Previously added robust helpers remain available if you want them later) ----
def robust_stats(df: pd.DataFrame, features: List[str] = FEATURES) -> Tuple[pd.Series, pd.Series]:
    med = df[features].median()
    mad = (df[features] - med).abs().median()
    robust_std = mad * 1.4826
    fallback_std = df[features].std().replace(0, 1e-9).fillna(1e-9)
    robust_std = robust_std.replace(0, np.nan).fillna(fallback_std).replace(0, 1e-9)
    return med, robust_std

def contributions(df: pd.DataFrame, center: pd.Series, scale: pd.Series, features: List[str] = FEATURES) -> pd.DataFrame:
    z = ((df[features] - center) / scale).abs()
    denom = z.sum(axis=1).replace(0, 1e-9)
    weights = z.div(denom, axis=0)
    out = z.add_prefix("z_").copy()
    for f in features:
        out[f"w_{f}"] = weights[f]
    out["top_feature"] = z.idxmax(axis=1)
    out["top_z"] = z.max(axis=1)
    return out
