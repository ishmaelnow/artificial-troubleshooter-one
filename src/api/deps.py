from functools import lru_cache
from pathlib import Path
from src.agent.service import State, build_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "system_metrics.csv"

@lru_cache(maxsize=1)
def get_state() -> State:
    return build_state(
        data_path=DATA_PATH,
        contamination=0.06,
        train_split=0.8,
        thresh_z=2.5,
        random_state=42,
    )
