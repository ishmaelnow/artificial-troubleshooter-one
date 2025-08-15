# src/agent/issue_detection.py
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Features used for unsupervised anomaly detection
FEATURES: List[str] = ["cpu_usage", "memory_usage", "network_latency", "disk_io"]

@dataclass
class IssueDetectorConfig:
    contamination: float = 0.06     # expected anomaly fraction (tweak as needed)
    random_state: int = 42

class IssueDetector:
    def __init__(self, config: IssueDetectorConfig = IssueDetectorConfig()):
        self.config = config
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self.fitted = False

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[FEATURES].values

    def fit(self, df: pd.DataFrame):
        X = self._X(df)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before score().")
        Xs = self.scaler.transform(self._X(df))
        # decision_function: higher = more normal. Flip so larger = more anomalous.
        return -self.model.decision_function(Xs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        Xs = self.scaler.transform(self._X(df))
        # sklearn returns {1: normal, -1: anomaly}. Convert to {0,1}.
        return (self.model.predict(Xs) == -1).astype(int)
