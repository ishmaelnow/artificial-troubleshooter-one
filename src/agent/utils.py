import pandas as pd
from pathlib import Path

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the dataset with parsed timestamps."""
    return pd.read_csv(path, parse_dates=["timestamp"])
