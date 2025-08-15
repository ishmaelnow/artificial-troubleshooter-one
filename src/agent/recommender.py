# src/agent/recommender.py
from typing import Dict

# Normalize various labels to a small set (works with class examples + our feature names)
ALIAS_MAP: Dict[str, str] = {
    "cpu": "high_cpu_usage",
    "cpu_usage": "high_cpu_usage",
    "high_cpu": "high_cpu_usage",
    "network_latency": "network_error",
    "network_issue": "network_error",
    "net": "network_error",
    "db": "database_issue",
    "database": "database_issue",
    "memory": "memory_pressure",
    "memory_usage": "memory_pressure",
    "disk": "disk_bottleneck",
    "disk_io": "disk_bottleneck",
}

SOLUTION_MAP: Dict[str, str] = {
    # Class example keys
    "network_error": "Restart the network service; verify DNS and gateway; run ping/traceroute.",
    "database_issue": "Check DB connectivity and credentials, inspect pool saturation, restart the service if needed.",
    "high_cpu_usage": "Identify hot processes (top/htop), optimize or throttle, scale resources or replicas.",
    # Extras for our feature-style root causes
    "memory_pressure": "Check for leaks, reduce cache size, restart offending service, consider raising memory limits.",
    "disk_bottleneck": "Check I/O queue depth, free space, rotate/compress logs, consider faster storage.",
    "multivariate": "Multiple signals involved—capture diagnostics (logs, top, iostat, netstat) and escalate to on-call.",
    "normal": "No issue detected.",
}

def normalize_root_cause(root_cause: str) -> str:
    key = (root_cause or "").strip().lower()
    return ALIAS_MAP.get(key, key)

def recommend_solution(root_cause: str) -> str:
    key = normalize_root_cause(root_cause)
    return SOLUTION_MAP.get(key, "No recommendation available.")
