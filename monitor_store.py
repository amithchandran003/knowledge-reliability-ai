"""Persistence helpers for monitoring run state."""

import json
from datetime import datetime
from pathlib import Path


MONITOR_DIR = Path("data/monitoring")
LATEST_RUN_FILE = MONITOR_DIR / "latest_run.json"
HISTORY_FILE = MONITOR_DIR / "run_history.json"


def _ensure_directory() -> None:
    """Create the monitoring data directory if needed."""
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)


def save_monitoring_run(results: list[dict]) -> dict:
    """Persist the latest monitoring cycle and append it to history."""

    _ensure_directory()

    run = {
        "run_id": datetime.now().astimezone().strftime(
            "%Y%m%dT%H%M%S%f"
        ),
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_count": len(results),
        "results": results,
    }

    with LATEST_RUN_FILE.open("w", encoding="utf-8") as file:
        json.dump(run, file, indent=2, default=str)

    history = []

    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open("r", encoding="utf-8") as file:
                history = json.load(file)

            if not isinstance(history, list):
                history = []

        except (json.JSONDecodeError, OSError):
            history = []

    history.append(run)

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, default=str)

    return run


def load_latest_monitoring_run() -> dict | None:
    """Load the most recent monitoring cycle."""

    if not LATEST_RUN_FILE.exists():
        return None

    try:
        with LATEST_RUN_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def load_monitoring_history() -> list[dict]:
    """Load all persisted monitoring cycles."""

    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []
