# scripts/recommend_solutions.py
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import ast
import pandas as pd
from agent.recommender import recommend_solution

FEATURES = ["cpu_usage", "memory_usage", "network_latency", "disk_io"]

def derive_root_cause_from_anomalous_columns(val) -> str:
    # val is a list or a stringified list like "['network_latency']"
    cols = []
    if isinstance(val, list):
        cols = val
    elif isinstance(val, str):
        try:
            cols = ast.literal_eval(val)
        except Exception:
            cols = []
    return cols[0] if isinstance(cols, list) and len(cols) == 1 else "multivariate"

def main() -> None:
    # Prefer the class-style z-score explanations from Step 4
    path = PROJECT_ROOT / "data" / "processed" / "anomaly_explanations_zscore.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/explain_anomalies_zscore.py first.")

    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Focus on rows flagged anomalous (1). Adjust here if you used -1.
    anomalies = df[df["anomaly"] == 1].copy()

    # Derive a single root_cause label per row from anomalous_columns (class method)
    anomalies["root_cause"] = anomalies["anomalous_columns"].apply(derive_root_cause_from_anomalous_columns)

    # Map to recommendations
    anomalies["recommendation"] = anomalies["root_cause"].apply(recommend_solution)

    # Show a quick preview
    cols = ["timestamp"] + FEATURES + ["root_cause", "recommendation"]
    print(anomalies.head(10)[cols].to_string(index=False))

    # Save
    out = PROJECT_ROOT / "data" / "processed" / "recommendations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    anomalies[cols].to_csv(out, index=False)
    print(f"\nSaved recommendations to: {out}")

if __name__ == "__main__":
    main()
