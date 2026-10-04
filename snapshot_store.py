"""File-based, per-source snapshot storage for V2."""

import json
import os
from datetime import datetime, timezone


SNAPSHOT_ROOT = os.path.join(os.path.dirname(__file__), "data", "snapshots_v2")


def build_snapshot(normalized_source: dict) -> dict:
    """Converts normalized content into one immutable source-version snapshot."""
    source_version_id = f"{normalized_source['source_id']}@{normalized_source['content_hash']}"
    return {
        "schema_version": "2.0",
        **normalized_source,
        "source_version_id": source_version_id,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _source_directory(source_id: str, root: str = SNAPSHOT_ROOT) -> str:
    return os.path.join(root, source_id)


def save_snapshot(snapshot: dict, root: str = SNAPSHOT_ROOT) -> str:
    """Stores one version once, under its own source-specific directory."""
    directory = _source_directory(snapshot["source_id"], root)
    os.makedirs(directory, exist_ok=True)
    filename = f"{snapshot['content_hash']}.json"
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(snapshot, file, indent=2)
    return path


def get_latest_snapshot(source_id: str, root: str = SNAPSHOT_ROOT) -> dict | None:
    """Returns the most recently retrieved version for exactly one source."""
    directory = _source_directory(source_id, root)
    if not os.path.exists(directory):
        return None
    snapshots = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), encoding="utf-8") as file:
                snapshots.append(json.load(file))
    return max(snapshots, key=lambda snapshot: snapshot["retrieval_timestamp"], default=None)


def record_source_status(source_id: str, status: dict, root: str = SNAPSHOT_ROOT) -> str:
    """Records the latest safe fetch status without treating failures as versions."""
    status_directory = os.path.join(root, "_status")
    os.makedirs(status_directory, exist_ok=True)
    path = os.path.join(status_directory, f"{source_id}.json")
    payload = {"source_id": source_id, "recorded_at": datetime.now(timezone.utc).isoformat(), **status}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path
