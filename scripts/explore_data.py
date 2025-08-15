# scripts/explore_data.py
from pathlib import Path
import sys

# Make "src" importable when running this file directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from agent.utils import load_dataset


def main() -> None:
    # Locate the CSV in data/raw
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "system_metrics.csv"

    # Load (timestamp parsed in utils.load_dataset)
    df = load_dataset(data_path)

    # Basic shape & columns
    print("Rows, Cols:", df.shape)
    print("Columns:", list(df.columns))

    # Time coverage
    print("Start:", df["timestamp"].min())
    print("End:  ", df["timestamp"].max())
    print("Span: ", df["timestamp"].max() - df["timestamp"].min())

    # Quick stats
    print("\nDescribe:")
    print(df.describe())

    # Error-rate prevalence
    print("\nError flag value counts:")
    print(df["error_rate"].value_counts(dropna=False))

    # Simple sanity checks (counts of out-of-range values)
    print("\nOut-of-range counts:")
    print({
        "cpu<0": int((df["cpu_usage"] < 0).sum()),
        "cpu>100": int((df["cpu_usage"] > 100).sum()),
        "mem<0": int((df["memory_usage"] < 0).sum()),
        "mem>100": int((df["memory_usage"] > 100).sum()),
        "latency<0": int((df["network_latency"] < 0).sum()),
        "disk_io<0": int((df["disk_io"] < 0).sum()),
    })


if __name__ == "__main__":
    main()
