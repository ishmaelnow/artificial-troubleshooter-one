# scripts/run_issue_detection.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from agent.utils import load_dataset
from agent.issue_detection import IssueDetector, IssueDetectorConfig

def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp obviously impossible values to reasonable bounds."""
    df = df.copy()
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["network_latency"] = df["network_latency"].clip(lower=0)
    df["disk_io"] = df["disk_io"].clip(lower=0)
    return df

def main() -> None:
    data_path = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"
    df = load_dataset(data_path)
    df = simple_clean(df)

    # Train on the first 80% (assumed mostly normal); score the whole dataset
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    full_df  = df

    detector = IssueDetector(IssueDetectorConfig(contamination=0.06))
    detector.fit(train_df)

    full_df = full_df.copy()
    full_df["anomaly_score"] = detector.score(full_df)
    full_df["anomaly"] = detector.predict(full_df)

    print(f"Anomalies detected: {int(full_df['anomaly'].sum())} / {len(full_df)}")

    # Show top 10 most anomalous rows
    cols = ["timestamp", "cpu_usage", "memory_usage", "network_latency", "disk_io", "anomaly_score", "anomaly"]
    top = full_df.sort_values("anomaly_score", ascending=False).head(10)[cols]
    print("\nTop 10 anomalies:")
    print(top.to_string(index=False))

    # Save results
    out_path = PROJECT_ROOT / "data" / "processed" / "anomaly_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved scored dataset to: {out_path}")

if __name__ == "__main__":
    main()
