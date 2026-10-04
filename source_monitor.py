# source_monitor.py
"""
Minimal Web Source Monitoring Prototype.

Fetches a public web page, extracts clean text, computes a content hash (SHA-256),
stores snapshots locally in data/snapshots/, and detects whether the source has changed.
Integrates with change_analyzer.py to identify specific text additions/removals.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlparse
import requests
from answer_memory import build_source_identity, build_source_version_id
from change_analyzer import analyze_content_diff, format_change_report

# Default test source: reliable, public, and does not block automated requests
DEFAULT_URL = "https://example.com/"
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data", "snapshots")


class HTMLTextExtractor(HTMLParser):
    """Simple parser using Python's standard library to extract visible text from HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.ignore_tag = False
        self.tags_to_ignore = {
            "script",
            "style",
            "head",
            "meta",
            "link",
            "noscript",
            "nav",
            "header",
            "footer",
        }

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.tags_to_ignore:
            self.ignore_tag = True

    def handle_endtag(self, tag):
        if tag.lower() in self.tags_to_ignore:
            self.ignore_tag = False

    def handle_data(self, data):
        if not self.ignore_tag:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


def extract_clean_text(html_content: str) -> str:
    """Extracts readable text from raw HTML content."""
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def compute_content_hash(text: str) -> str:
    """Generates a SHA-256 checksum for the provided text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_source_snapshot(url: str = DEFAULT_URL) -> dict:
    """
    Fetches the web page, extracts clean text, and generates snapshot metadata.
    Returns a dictionary with source_url, retrieval_timestamp, content, and content_hash.
    """
    headers = {"User-Agent": "ReliabilitySourceMonitor/1.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    clean_text = extract_clean_text(response.text)
    content_hash = compute_content_hash(clean_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    source_identifier = urlparse(url).netloc or url
    source_identity = build_source_identity(source_identifier, url)

    return {
        "schema_version": "1.0",
        "source_id": source_identity["source_id"],
        "source_identifier": source_identity["source_identifier"],
        "source_url": url,
        "retrieval_timestamp": timestamp,
        "content_hash": content_hash,
        "source_version_id": build_source_version_id(
            source_identity["source_id"],
            content_hash,
        ),
        "content": clean_text,
    }


def save_snapshot(snapshot: dict, directory: str = SNAPSHOT_DIR) -> str:
    """Saves a snapshot dictionary to a timestamped JSON file in data/snapshots/."""
    os.makedirs(directory, exist_ok=True)

    safe_time = snapshot["retrieval_timestamp"].replace(":", "-")
    file_path = os.path.join(directory, f"snapshot_{safe_time}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    return file_path


def get_latest_snapshot(directory: str = SNAPSHOT_DIR) -> dict | None:
    """Loads and returns the most recent snapshot file from the snapshot directory."""
    if not os.path.exists(directory):
        return None

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("snapshot_") and f.endswith(".json")
    ]

    if not files:
        return None

    files.sort()
    latest_file = files[-1]

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def monitor_source(
    url: str = DEFAULT_URL,
    directory: str = SNAPSHOT_DIR,
) -> dict:
    """
    Fetches the current source, compares it with the previous snapshot,
    reports status, analyzes changes if any, and saves the new snapshot.
    """
    print(f"Fetching: {url}")

    current_snapshot = fetch_source_snapshot(url)
    previous_snapshot = get_latest_snapshot(directory)

    if previous_snapshot is None:
        change_status = (
            "Initial snapshot created "
            "(no previous version to compare)"
        )
        has_changed = False
        prev_hash = "N/A"
        prev_time = "N/A"
        diff_report = None

    else:
        prev_hash = previous_snapshot["content_hash"]
        prev_time = previous_snapshot["retrieval_timestamp"]
        curr_hash = current_snapshot["content_hash"]

        if curr_hash == prev_hash:
            change_status = "No meaningful source change detected"
            has_changed = False
            diff_report = None

        else:
            change_status = "Source changed"
            has_changed = True

            diff_result = analyze_content_diff(
                previous_snapshot["content"],
                current_snapshot["content"],
            )

            diff_report = format_change_report(
                diff_result,
                old_meta=previous_snapshot,
                new_meta=current_snapshot,
            )

    save_snapshot(current_snapshot, directory)

    return {
        "current_snapshot": current_snapshot,
        "previous_snapshot": previous_snapshot,
        "change_status": change_status,
        "has_changed": has_changed,
        "previous_hash": prev_hash,
        "previous_timestamp": prev_time,
        "diff_report": diff_report,
    }


if __name__ == "__main__":
    result = monitor_source()

    print("=" * 60)
    print("SOURCE MONITORING RESULT")
    print("=" * 60)
    print(f"Status       : {result['change_status']}")
    print(f"Changed      : {result['has_changed']}")
    print(f"Current Hash : {result['current_snapshot']['content_hash']}")

    if result["diff_report"]:
        print("\nCHANGE REPORT")
        print(result["diff_report"])