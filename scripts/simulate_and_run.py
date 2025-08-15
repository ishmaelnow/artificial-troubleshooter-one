# scripts/simulate_and_run.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import ast
import numpy as np
import pandas as pd

from agent.utils import load_dataset
from agent.issue_detection import IssueDetector, IssueDetectorConfig, FEATURES
from agent.recommender import recommend_solution

# --- helpers -----------------------------------------------------------------
def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["network_latency"] = df["network_latency"].clip(lower=0)
    df["disk_io"] = df["disk_io"].clip(lower=0)
    return df

def zscore_wrt_train(full_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    means = train_df[FEATURES].mean()
    stds  = train_df[FEATURES].std().replace(0, 1e-9)
    return (full_df[FEATURES] - means) / stds

def derive_root_cause_from_cols(cols) -> str:
    # single driver -> that feature; none/multiple -> multivariate
    if isinstance(cols, str):
        try: cols = ast.literal_eval(cols)
        except Exception: cols = []
    return cols[0] if isinstance(cols, list) and len(cols) == 1 else "multivariate"

# --- simulation functions -----------------------------------------------------
def simulate_network_spike(df: pd.DataFrame, idx: int, value: float = 1000.0):
    df.loc[idx, "network_latency"] = value

def simulate_cpu_spike(df: pd.DataFrame, idx: int, value: float = 99.0):
    df.loc[idx, "cpu_usage"] = value

def simulate_memory_drop(df: pd.DataFrame, idx: int, value: float = 5.0):
    df.loc[idx, "memory_usage"] = value

def simulate_disk_burst(df: pd.DataFrame, idx: int, value: float = 120.0):
    df.loc[idx, "disk_io"] = value

# --- main --------------------------------------------------------------------
def main() -> None:
    # 0) Load & clean
    raw = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"
    df = load_dataset(raw).sort_values("timestamp").reset_index(drop=True)
    df = simple_clean(df)

    # 1) Split: train (first 80%), test window (last 20 rows)
    N_TEST = 20
    split_idx = len(df) - N_TEST
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    # 2) SIMULATE issues in the test window (so training remains clean)
    # Example injections (adjust as you like):
    simulate_network_spike(test_df, test_df.index[-1], value=1000.0)  # huge latency at last row
    simulate_cpu_spike(test_df,     test_df.index[-5], value=98.0)    # near-saturation CPU
    simulate_memory_drop(test_df,   test_df.index[-10], value=6.0)    # unusually low memory
    simulate_disk_burst(test_df,    test_df.index[-15], value=125.0)  # high disk I/O

    # 3) Recombine modified test window
    df_sim = pd.concat([train_df, test_df], axis=0).sort_index()

    # 4) Step 3 — detect anomalies (fit on TRAIN, score TEST)
    det = IssueDetector(IssueDetectorConfig(contamination=0.06, random_state=42)).fit(train_df)

    df_sim["anomaly_score"] = det.score(df_sim)
    df_sim["anomaly"] = det.predict(df_sim)  # 0/1

    # 5) Step 4 — per-column drivers via z-scores (w.r.t. TRAIN stats)
    Z = zscore_wrt_train(df_sim, train_df)
    THRESH = 2.5
    df_sim["anomalous_columns"] = [
        [col for col, val in Z.loc[i].items() if pd.notna(val) and abs(val) > THRESH]
        for i in df_sim.index
    ]

    # 6) Step 6 — recommend a solution from the derived cause
    df_sim["root_cause"] = df_sim["anomalous_columns"].apply(derive_root_cause_from_cols)
    df_sim["recommendation"] = df_sim["root_cause"].apply(recommend_solution)

    # 7) Show anomalies only within the test window (what we just perturbed)
    view_cols = ["timestamp"] + FEATURES + ["anomaly_score","anomaly","anomalous_columns","root_cause","recommendation"]
    test_view = df_sim.loc[test_df.index, view_cols]
    anoms = test_view[test_view["anomaly"] == 1]

    print(f"Test window size: {len(test_view)} rows; anomalies found: {len(anoms)}")
    if len(anoms):
        print("\nAnomalies detected in simulated window:")
        print(anoms.to_string(index=False))
    else:
        print("\nNo anomalies detected in the simulated window.")

    # 8) Save results
    out = PROJECT_ROOT / "data" / "processed" / "simulated_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    test_view.to_csv(out, index=False)
    print(f"\nSaved simulated results to: {out}")

if __name__ == "__main__":
    main()
