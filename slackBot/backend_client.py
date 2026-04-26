"""
Backend client — calls the telemetry backend to re-run a diagnosis.

MVP stub: always returns an empty dict (simulates "all clear").

To wire up a real backend, replace the stub body with:
    import requests, os
    resp = requests.get(f"{os.environ['BACKEND_URL']}/diagnose/{username}", timeout=10)
    resp.raise_for_status()
    return resp.json()   # expected: {"avg_memory_utilization": 0.42, ...}
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_rerun_counts: dict[str, int] = {}


def run_diagnosis(username: str) -> dict:
    """
    Ask the telemetry backend for the latest predictions for the given user.

    Returns a dict of metric_key -> confidence float.
    Returns {} if the backend is unreachable or if no risks are found.

    MVP: stubbed to always return {} (all clear).
    """
    backend_url = os.environ.get("BACKEND_URL", "").rstrip("/")

    # ── MVP stub ─────────────────────────────────────────────────────────
    if not backend_url or backend_url == "http://localhost:8000":
        _rerun_counts[username] = _rerun_counts.get(username, 0) + 1
        if _rerun_counts[username] == 1:
            return {"avg_memory_utilization": 80, "uptime_days": 44, "max_cpu_usage": 96}
        return {}
    # ─────────────────────────────────────────────────────────────────────

    # ── Real backend call (uncomment when ready) ──────────────────────────
    # try:
    #     resp = requests.get(
    #         f"{backend_url}/diagnose/{username}",
    #         timeout=10,
    #     )
    #     resp.raise_for_status()
    #     return resp.json()
    # except Exception as exc:
    #     print(f"[backend_client] Failed to reach backend for {username}: {exc}")
    #     return {}
    # ─────────────────────────────────────────────────────────────────────

    return {}
