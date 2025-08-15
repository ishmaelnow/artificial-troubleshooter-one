from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from scipy.stats import zscore

FEATURES = ["cpu_usage", "memory_usage", "network_latency", "disk_io"]
THRESHOLD = 2.5  # a bit gentler than 3.0

def main() -> None:
    scored_path = PROJECT_ROOT / "data" / "processed" / "anomaly_scores.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}. Run Step 3 first.")

    df = pd.read_csv(scored_path, parse_dates=["timestamp"])

    # Determine anomaly label encoding
    uniq = set(df["anomaly"].unique())
    anomaly_flag = -1 if uniq.issubset({-1, 1}) else 1

    # Use the same train split as detection (first 80%) for reference stats
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    # Compute column-wise z-scores for the FULL data, but w.r.t TRAIN stats
    means = train_df[FEATURES].mean()
    stds = train_df[FEATURES].std().replace(0, 1e-9)
    z_scores = (df[FEATURES] - means) / stds  # same spirit as numeric_data.apply(zscore)

    # Class-style function: anomalous columns per row if |z| > THRESHOLD
    def find_anomalous_columns_for_row(idx: int) -> list:
        zs = z_scores.loc[idx]
        return [col for col, val in zs.items() if pd.notna(val) and abs(val) > THRESHOLD]

    df["anomalous_columns"] = [
        find_anomalous_columns_for_row(i) if df.at[i, "anomaly"] == anomaly_flag else []
        for i in df.index
    ]

    # Show a small sample of anomalous rows and their columns
    view = df[df["anomaly"] == anomaly_flag][["timestamp", "anomaly", "anomaly_score", "anomalous_columns"]]
    print(view.head(20).to_string(index=False))

    # Quick stats to understand blanks vs non-blanks
    non_empty = (view["anomalous_columns"].str.len() > 0).sum()
    print(f"\nAnomalous rows with >=1 flagged column: {non_empty} / {len(view)}")

    out_path = PROJECT_ROOT / "data" / "processed" / "anomaly_explanations_zscore.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved class-style z-score explanations to: {out_path}")

if __name__ == "__main__":
    main()
