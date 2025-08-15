# src/agent/root_cause.py
from dataclasses import dataclass
from typing import List, Dict
import ast
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

FEATURES: List[str] = ["cpu_usage", "memory_usage", "network_latency", "disk_io"]

def derive_root_cause_label_from_list_str(s: str) -> str:
    """
    Parse the stringified list from 'anomalous_columns' and return a single label.
    - If one column: that column name (e.g., 'network_latency')
    - If none or multiple: 'multivariate'
    """
    cols = []
    if isinstance(s, list):
        cols = s
    elif isinstance(s, str):
        try:
            cols = ast.literal_eval(s)
        except Exception:
            cols = []
    return str(cols[0]) if isinstance(cols, list) and len(cols) == 1 else "multivariate"

def attach_root_cause_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'root_cause' column derived from 'anomalous_columns'.
    Normal rows (anomaly==0) remain labeled 'normal' to keep them distinct if needed.
    """
    out = df.copy()
    out["root_cause"] = out.apply(
        lambda r: derive_root_cause_label_from_list_str(r.get("anomalous_columns", "")) if r.get("anomaly", 0) in (1, -1)
        else "normal",
        axis=1,
    )
    return out

@dataclass
class RootCauseConfig:
    max_depth: int | None = 4
    min_samples_leaf: int = 5
    random_state: int = 42

class RootCauseAnalyzer:
    def __init__(self, config: RootCauseConfig = RootCauseConfig()):
        self.config = config
        self.model = DecisionTreeClassifier(
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[FEATURES], y)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X[FEATURES])

    def feature_importances(self) -> Dict[str, float]:
        return {f: float(w) for f, w in zip(FEATURES, self.model.feature_importances_)}
