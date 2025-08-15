# scripts/explain_anomalies.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from agent.utils import load_dataset
from agent.anomaly_explainer import robust_stats, contributions, FEATURES

def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp obviously impossible values to reasonable bounds (same as detection)."""
    df = df.copy()
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["network_latency"] = df["network_latency"].clip(lower=0)
    df["disk_io"] = df["disk_io"].clip(lower=0)
    return df

def main() -> None:
    # Prefer the scored file from Step 3; fallback to raw if needed
    scored_path = PROJECT_ROOT / "data" / "processed" / "anomaly_scores.csv"
    if scored_path.exists():
        df = pd.read_csv(scored_path, parse_dates=["timestamp"])
    else:
        # Re-load raw (will not have anomaly columns if Step 3 wasn't run)
        raw_path = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"
        df = load_dataset(raw_path)

    df = simple_clean(df)

    # Train reference = first 80% (same split as detection script)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    # Compute robust center/scale on train only
    center, scale = robust_stats(train_df, FEATURES)

    # Focus on rows flagged anomalous (if present); otherwise use entire df
    if "anomaly" in df.columns:
        target = df[df["anomaly"] == 1].copy()
    else:
        target = df.copy()

    # Compute per-row contributions
    contrib = contributions(target, center, scale, FEATURES)

    # Merge key fields to view
    view_cols = ["timestamp"] + FEATURES
    result = pd.concat([target[view_cols].reset_index(drop=True), contrib.reset_index(drop=True)], axis=1)

    # Show top 10 most “explained” anomalies (largest top_z)
    top = result.sort_values("top_z", ascending=False).head(10)
    cols_to_show = ["timestamp"] + FEATURES + ["top_feature", "top_z"]
    print("\nTop 10 anomalous rows with dominant feature:")
    print(top[cols_to_show].to_string(index=False))

    # Save full contributions
    out_path = PROJECT_ROOT / "data" / "processed" / "anomaly_explanations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved feature-level explanations to: {out_path}")

if __name__ == "__main__":
    main()
