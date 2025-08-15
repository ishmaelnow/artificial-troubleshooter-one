# src/api/routers/analyze.py
from typing import List
from fastapi import APIRouter, Depends
from src.api.schemas import LogEntry, AnalyzeResult
from src.api.deps import get_state
from src.agent.service import analyze_one, analyze_batch, State

router = APIRouter(tags=["analyze"])

def _to_dict(model):
    # Works on Pydantic v2 (model_dump) and v1 (dict)
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

@router.post("/analyze", response_model=AnalyzeResult)
def analyze(entry: LogEntry, state: State = Depends(get_state)):
    return analyze_one(_to_dict(entry), state)

@router.post("/batch", response_model=List[AnalyzeResult])
def batch(entries: List[LogEntry], state: State = Depends(get_state)):
    payload = [_to_dict(e) for e in entries]
    return analyze_batch(payload, state)
