"""Deterministic content normalization and meaningful chunk creation for V2."""

import hashlib
import json
import re

from source_monitor import extract_clean_text


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalizes whitespace without changing the wording or values."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def text_chunks(text: str) -> list[dict]:
    """Creates meaningful evidence chunks while keeping short headings with their following content."""
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", text)
        if unit.strip()
    ]

    merged_units = []
    index = 0

    while index < len(units):
        current = units[index]

        # Short heading-like lines are not useful evidence by themselves.
        # Keep them together with the following content.
        is_heading_like = (
            len(current) <= 80
            and not re.search(r"[.!?]$", current)
            and index + 1 < len(units)
        )

        if is_heading_like:
            current = f"{current} {units[index + 1]}"
            index += 2
        else:
            index += 1

        merged_units.append(current)

    chunks = []
    offset = 0

    for chunk_index, unit in enumerate(merged_units, 1):
        start = text.find(unit, offset)

        if start < 0:
            start = offset

        end = start + len(unit)

        chunks.append(
            {
                "chunk_id": f"text-unit-{chunk_index:04d}",
                "text": unit,
                "location": {
                    "kind": "text_unit",
                    "start_char": start,
                    "end_char": end,
                },
            }
        )

        offset = end

    return chunks


def _flatten_json(value, path: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        pairs = []
        for key in sorted(value):
            pairs.extend(_flatten_json(value[key], f"{path}/{key}"))
        return pairs

    if isinstance(value, list):
        pairs = []
        for index, item in enumerate(value):
            pairs.extend(_flatten_json(item, f"{path}/{index}"))
        return pairs

    return [(path or "/", value)]


def normalize_web_page(raw_html: str) -> tuple[str, list[dict]]:
    clean_text = normalize_text(extract_clean_text(raw_html))
    return clean_text, text_chunks(clean_text)


def normalize_json_api(
    raw_json: str,
    ignored_fields: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Canonicalizes JSON and uses JSON Pointer paths as chunk identifiers."""
    ignored = set(ignored_fields or [])

    parsed = json.loads(raw_json)

    canonical = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    chunks = []

    for path, value in _flatten_json(parsed):
        if path.rsplit("/", 1)[-1] in ignored:
            continue

        chunks.append(
            {
                "chunk_id": f"json:{path}",
                "text": json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "location": {
                    "kind": "json_pointer",
                    "path": path,
                },
            }
        )

    return canonical, chunks


def normalize_fetch_result(
    source_config: dict,
    raw_content: str,
) -> dict:
    """Produces source-type-specific canonical content and evidence chunks."""
    if source_config["source_type"] == "web_page":
        content, chunks = normalize_web_page(raw_content)

    elif source_config["source_type"] == "json_api":
        content, chunks = normalize_json_api(
            raw_content,
            source_config.get("ignore_fields"),
        )

    else:
        raise ValueError(
            f"Unsupported source type: {source_config['source_type']}"
        )

    return {
        "source_id": source_config["source_id"],
        "source_identifier": source_config["source_identifier"],
        "source_url": source_config["source_url"],
        "source_type": source_config["source_type"],
        "normalized_content": content,
        "content_hash": compute_hash(content),
        "chunks": chunks,
    }