from __future__ import annotations
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field

class LogEntry(BaseModel):
    timestamp: Optional[datetime] = Field(default=None)
    cpu_usage: float
    memory_usage: float
    network_latency: float
    disk_io: float

class AnalyzeResult(BaseModel):
    timestamp: Optional[datetime]
    features: Dict[str, float]
    anomaly: int
    anomaly_score: float
    anomalous_columns: List[str]
    root_cause: str
    recommendation: str
