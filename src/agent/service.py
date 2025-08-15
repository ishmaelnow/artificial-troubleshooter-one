# src/agent/service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from .issue_detection import IssueDetector, IssueDetectorConfig, FEATURES
from .root_cause import RootCauseAnalyzer, RootCauseConfig, attach_root_cause_labels
from .recommender import recommend_solution
from .utils import load_dataset


@dataclass
class State:
    train_df: pd.DataFrame
    means: pd.Series
    stds: pd.Series
    detector: IssueDetector
    rca: Optional[RootCauseAnalyzer]
    model_used: bool
    thresh_z: float


def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cpu_usage"] = d["cpu_usage"].clip(0, 100)
    d["memory_usage"] = d["memory_usage"].clip(0, 100)
    d["network_latency"] = d["network_latency"].clip(lower=0)
    d["disk_io"] = d["disk_io"].clip(lower=0)
    return d


def build_state(
    data_path: Path,
    contamination: float = 0.06,
    train_split: float = 0.8,
    thresh_z: float = 2.5,
    random_state: int = 42,
) -> State:
    df = load_dataset(data_path).sort_values("timestamp").reset_index(drop=True)
    df = simple_clean(df)

    split = int(len(df) * train_split)
    train_df = df.iloc[:split].copy()

    # Fit anomaly detector on train window
    det = IssueDetector(
        IssueDetectorConfig(contamination=contamination, random_state=random_state)
    ).fit(train_df)

    # Train stats for z-scores
    means = train_df[FEATURES].mean()
    stds = train_df[FEATURES].std().replace(0, 1e-9)

    # Prepare RCA training data (derive labels from z-scores)
    train_df = train_df.copy()
    train_df["anomaly"] = det.predict(train_df)
    Z = (train_df[FEATURES] - means) / stds
    train_df["anomalous_columns"] = [
        [c for c, v in Z.loc[i].abs().items() if pd.notna(v) and abs(v) > thresh_z]
        for i in train_df.index
    ]
    train_anoms = train_df.query("anomaly == 1")

    rca: Optional[RootCauseAnalyzer] = None
    model_used = False
    if len(train_anoms) >= 10:
        labeled = attach_root_cause_labels(train_anoms)  # adds 'root_cause'
        rca = RootCauseAnalyzer(
            RootCauseConfig(max_depth=4, min_samples_leaf=5, random_state=random_state)
        )
        rca.fit(labeled[FEATURES], labeled["root_cause"])
        model_used = True

    return State(
        train_df=train_df,
        means=means,
        stds=stds,
        detector=det,
        rca=rca,
        model_used=model_used,
        thresh_z=thresh_z,
    )


def _z_anomalous_cols(
    one_df: pd.DataFrame, means: pd.Series, stds: pd.Series, thresh: float
) -> List[str]:
    z = (one_df[FEATURES] - means) / stds.replace(0, 1e-9)
    zs = z.iloc[0].abs()
    return [c for c, v in zs.items() if pd.notna(v) and v > thresh]


def analyze_one(entry: Dict[str, Any], state: State) -> Dict[str, Any]:
    one = pd.DataFrame([{**entry}])

    # Robust timestamp handling: default if missing OR null/invalid
    if ("timestamp" not in one.columns) or pd.isna(one.iloc[0].get("timestamp")):
        one.loc[0, "timestamp"] = pd.Timestamp.utcnow()
    else:
        ts_parsed = pd.to_datetime(one.iloc[0]["timestamp"], errors="coerce")
        if pd.isna(ts_parsed):
            ts_parsed = pd.Timestamp.utcnow()
        one.loc[0, "timestamp"] = ts_parsed

    one = simple_clean(one)

    # Detection
    score = float(state.detector.score(one)[0])
    anomaly = int(state.detector.predict(one)[0])  # 0/1

    # Column drivers (z-scores w.r.t. train stats)
    cols = _z_anomalous_cols(one, state.means, state.stds, state.thresh_z)

    # Root cause: prefer trained RCA if anomalous; otherwise derive from columns
    if anomaly == 1 and state.model_used and state.rca is not None:
        cause = str(state.rca.predict(one[FEATURES])[0])
    else:
        tmp = one.copy()
        tmp["anomalous_columns"] = [cols]
        cause = str(attach_root_cause_labels(tmp)["root_cause"].iloc[0])

    # Safe datetime for response
    ts_out = pd.to_datetime(one["timestamp"].iloc[0])
    if hasattr(ts_out, "to_pydatetime"):
        ts_out = ts_out.to_pydatetime()

    rec = recommend_solution(cause)

    return {
        "timestamp": ts_out,
        "features": {k: float(one[k].iloc[0]) for k in FEATURES},
        "anomaly": anomaly,
        "anomaly_score": score,
        "anomalous_columns": cols,
        "root_cause": cause,
        "recommendation": rec,
    }


def analyze_batch(entries: List[Dict[str, Any]], state: State) -> List[Dict[str, Any]]:
    return [analyze_one(e, state) for e in entries]
