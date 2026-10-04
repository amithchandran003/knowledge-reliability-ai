# change_analyzer.py
"""
Meaningful Change Analysis Module.

Performs deterministic sentence-level text diffing between snapshots to identify
precisely what content was added, removed, or retained without using an LLM.
"""

import difflib
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone


def split_into_units(text: str) -> list[str]:
    """
    Splits text into meaningful units (sentences or distinct lines).
    Strips whitespace and discards empty segments.
    """
    # Split on sentence terminators (.!?) followed by space or newline boundaries
    raw_units = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [u.strip() for u in raw_units if u.strip()]


def analyze_content_diff(old_text: str, new_text: str) -> dict:
    """
    Deterministically compares old_text and new_text at the sentence/unit level.
    Returns:
      - has_changed: bool
      - added: list of strings added in new_text
      - removed: list of strings removed from old_text
      - unchanged: list of strings present in both
    """
    old_units = split_into_units(old_text)
    new_units = split_into_units(new_text)

    matcher = difflib.SequenceMatcher(None, old_units, new_units)
    added = []
    removed = []
    unchanged = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged.extend(old_units[i1:i2])
        elif tag == "replace":
            removed.extend(old_units[i1:i2])
            added.extend(new_units[j1:j2])
        elif tag == "delete":
            removed.extend(old_units[i1:i2])
        elif tag == "insert":
            added.extend(new_units[j1:j2])

    has_changed = bool(added or removed)

    return {
        "has_changed": has_changed,
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
        "summary": {
            "additions_count": len(added),
            "removals_count": len(removed),
            "unchanged_count": len(unchanged),
        },
    }


def format_change_report(diff_result: dict, old_meta: dict = None, new_meta: dict = None) -> str:
    """
    Generates a clear, human-readable change report from the diff result.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CHANGE ANALYSIS REPORT")
    lines.append("=" * 60)

    if old_meta and new_meta:
        lines.append(f"Source URL:          {new_meta.get('source_url', 'N/A')}")
        lines.append(f"Previous Timestamp:  {old_meta.get('retrieval_timestamp', 'N/A')}")
        lines.append(f"Current Timestamp:   {new_meta.get('retrieval_timestamp', 'N/A')}")
        lines.append(f"Previous Hash:       {old_meta.get('content_hash', 'N/A')[:16]}...")
        lines.append(f"Current Hash:        {new_meta.get('content_hash', 'N/A')[:16]}...")
        lines.append("-" * 60)

    if not diff_result["has_changed"]:
        lines.append("Status: No meaningful source change detected.")
        lines.append("All content remains identical.")
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append("Status: Source changed.")
    summary = diff_result["summary"]
    lines.append(
        f"Summary: {summary['additions_count']} addition(s), "
        f"{summary['removals_count']} removal(s), "
        f"{summary['unchanged_count']} unchanged unit(s).\n"
    )

    if diff_result["removed"]:
        lines.append("[-] REMOVED CONTENT:")
        for item in diff_result["removed"]:
            lines.append(f"    • {item}")
        lines.append("")

    if diff_result["added"]:
        lines.append("[+] ADDED CONTENT:")
        for item in diff_result["added"]:
            lines.append(f"    • {item}")
        lines.append("")

    if diff_result["unchanged"]:
        lines.append("[=] UNCHANGED CONTENT:")
        for item in diff_result["unchanged"]:
            lines.append(f"    • {item}")

    lines.append("=" * 60)
    return "\n".join(lines)


def run_controlled_demo():
    """
    Controlled test fixture to demonstrate change detection with real snapshots,
    real timestamps, and genuine SHA-256 hashes.
    """
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test_snapshots")
    os.makedirs(test_dir, exist_ok=True)

    # Controlled source version 1
    content_v1 = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 30 days.\n"
        "Contact support for billing assistance."
    )
    time_v1 = datetime.now(timezone.utc).isoformat()
    hash_v1 = hashlib.sha256(content_v1.encode("utf-8")).hexdigest()

    snapshot_v1 = {
        "source_url": "https://example.internal/terms-and-conditions",
        "retrieval_timestamp": time_v1,
        "content_hash": hash_v1,
        "content": content_v1,
    }

    v1_file = os.path.join(test_dir, "snapshot_v1.json")
    with open(v1_file, "w", encoding="utf-8") as f:
        json.dump(snapshot_v1, f, indent=2)

    # Slight pause to produce genuine distinct timestamps
    time.sleep(1)

    # Controlled source version 2: Refund policy reduced from 30 days to 15 days
    content_v2 = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 15 days.\n"
        "Contact support for billing assistance."
    )
    time_v2 = datetime.now(timezone.utc).isoformat()
    hash_v2 = hashlib.sha256(content_v2.encode("utf-8")).hexdigest()

    snapshot_v2 = {
        "source_url": "https://example.internal/terms-and-conditions",
        "retrieval_timestamp": time_v2,
        "content_hash": hash_v2,
        "content": content_v2,
    }

    v2_file = os.path.join(test_dir, "snapshot_v2.json")
    with open(v2_file, "w", encoding="utf-8") as f:
        json.dump(snapshot_v2, f, indent=2)

    # Run deterministic diff analysis
    diff_result = analyze_content_diff(snapshot_v1["content"], snapshot_v2["content"])
    report = format_change_report(diff_result, old_meta=snapshot_v1, new_meta=snapshot_v2)

    print(report)


if __name__ == "__main__":
    run_controlled_demo()
