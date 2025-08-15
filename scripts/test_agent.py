# scripts/test_agent.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import ast
import pandas as pd
import numpy as np

# Step 3 pieces
from agent.issue_detection import IssueDetector, IssueDetectorConfig, FEATURES
from sklearn.preprocessing import StandardScaler

# Step 4 z-score explainer (class style, train-only stats)
from scipy.stats import zscore

# Step 6 recommender
from agent.recommender import recommend_solution

def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["network_latency"] = df["network_latency"].clip(lower=0)
    df["disk_io"] = df["disk_io"].clip(lower=0)
    return df

def derive_anomalous_columns_zscore(df: pd.DataFrame, train_df: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
    """Return a Series of lists, each list containing feature names whose |z| > threshold."""
    means = train_df[FEATURES].mean()
    stds = train_df[FEATURES].std().replace(0, 1e-9)
    z = (df[FEATURES] - means) / stds
    out = []
    for idx in df.index:
        zs = z.loc[idx]
        cols = [col for col, val in zs.items() if pd.notna(val) and abs(val) > threshold]
        out.append(cols)
    return pd.Series(out, index=df.index)

def derive_root_cause_from_cols(cols) -> str:
    """Single-feature → that feature; empty or multi → 'multivariate'."""
    if isinstance(cols, str):
        try:
            cols = ast.literal_eval(cols)
        except Exception:
            cols = []
    return cols[0] if isinstance(cols, list) and len(cols) == 1 else "multivariate"

def main() -> None:
    data_path = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = simple_clean(df)

    # Hold-out a small "new data" window to simulate real testing
    N_TEST = 20
    split_idx = len(df) - N_TEST
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    # === Step 3: Issue detection (fit on history, score new data) ===
    scaler = StandardScaler().fit(train_df[FEATURES])
    det = IssueDetector(IssueDetectorConfig(contamination=0.06, random_state=42)).fit(train_df)

    test_df = test_df.copy()
    test_df["anomaly_score"] = det.score(test_df)
    test_df["anomaly"] = det.predict(test_df)  # 0/1

    # === Step 4: Column-level drivers via z-scores (relative to train stats) ===
    test_df["anomalous_columns"] = derive_anomalous_columns_zscore(test_df, train_df, threshold=2.5)

    # === Step 5: (optional here) We could load your trained tree and predict a cause.
    # For the minimal test, derive cause from anomalous_columns (class style):
    test_df["root_cause"] = test_df["anomalous_columns"].apply(derive_root_cause_from_cols)

    # === Step 6: Recommend a fix ===
    test_df["recommendation"] = test_df["root_cause"].apply(recommend_solution)

    # Show anomalies in the test window
    view_cols = ["timestamp"] + FEATURES + ["anomaly_score", "anomaly", "anomalous_columns", "root_cause", "recommendation"]
    anoms = test_df[test_df["anomaly"] == 1][view_cols]

    print(f"Test window size: {len(test_df)} rows; anomalies found: {len(anoms)}")
    if len(anoms):
        print("\nAnomalies in test window:")
        print(anoms.to_string(index=False))
    else:
        print("\nNo anomalies in the test window.")

    # Save for inspection
    out = PROJECT_ROOT / "data" / "processed" / "test_agent_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    test_df[view_cols].to_csv(out, index=False)
    print(f"\nSaved Step 7 results to: {out}")

if __name__ == "__main__":
    main()
