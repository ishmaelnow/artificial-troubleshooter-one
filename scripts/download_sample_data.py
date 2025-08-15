import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    "timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="h"),
    "cpu_usage": np.random.normal(50, 10, n_samples),        # %
    "memory_usage": np.random.normal(60, 15, n_samples),      # %
    "network_latency": np.random.normal(100, 20, n_samples),  # ms
    "disk_io": np.random.normal(75, 10, n_samples),           # MB/s
    "error_rate": np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% error
})

raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "system_metrics.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(raw_path, index=False)

print(df.head())
print(df.info())
print(f"\nSaved dataset to: {raw_path}")
