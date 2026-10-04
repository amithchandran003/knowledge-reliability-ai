"""Deterministic chunk comparison plus a narrow Gemini semantic-change classifier."""

import json
import os
from difflib import SequenceMatcher

import requests


GEMINI_REQUEST_TIMEOUT_SECONDS = 30


def align_and_diff_chunks(old_chunks: list[dict], new_chunks: list[dict]) -> list[dict]:
    """Aligns chunks by ID where possible, otherwise by deterministic sequence diff."""
    old_by_id = {chunk["chunk_id"]: chunk for chunk in old_chunks}
    new_by_id = {chunk["chunk_id"]: chunk for chunk in new_chunks}
    changes = []

    shared_ids = old_by_id.keys() & new_by_id.keys()
    for chunk_id in sorted(shared_ids):
        if old_by_id[chunk_id]["text"] != new_by_id[chunk_id]["text"]:
            changes.append({"change_kind": "modified", "old_chunk": old_by_id[chunk_id], "new_chunk": new_by_id[chunk_id]})

    old_remaining = [chunk for chunk in old_chunks if chunk["chunk_id"] not in shared_ids]
    new_remaining = [chunk for chunk in new_chunks if chunk["chunk_id"] not in shared_ids]
    matcher = SequenceMatcher(None, [chunk["text"] for chunk in old_remaining], [chunk["text"] for chunk in new_remaining])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            pairs = max(i2 - i1, j2 - j1)
            for offset in range(pairs):
                old_chunk = old_remaining[i1 + offset] if i1 + offset < i2 else None
                new_chunk = new_remaining[j1 + offset] if j1 + offset < j2 else None
                changes.append({"change_kind": "modified", "old_chunk": old_chunk, "new_chunk": new_chunk})
        elif tag == "delete":
            for old_chunk in old_remaining[i1:i2]:
                changes.append({"change_kind": "removed", "old_chunk": old_chunk, "new_chunk": None})
        elif tag == "insert":
            for new_chunk in new_remaining[j1:j2]:
                changes.append({"change_kind": "added", "old_chunk": None, "new_chunk": new_chunk})
    return changes


def classify_semantic_change(old_text: str, new_text: str, api_key: str | None = None) -> dict:
    """Classifies one already-relevant changed chunk; it never receives an API key in its URL."""
    if old_text == new_text:
        return {"change_type": "equivalent", "confidence": "high", "changed_facts": [], "explanation": "Text is identical."}

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"change_type": "uncertain", "confidence": "low", "changed_facts": [], "explanation": "Gemini API key is unavailable."}

    prompt = f"""Classify the meaning change between two evidence excerpts. Return JSON only.
Use change_type exactly as one of: material, equivalent, formatting_only, uncertain.
A factual value change that could alter an answer is material. Different wording with the same factual meaning is equivalent.

OLD:\n{old_text}\n\nNEW:\n{new_text}

Return: {{"change_type":"...","confidence":"high|medium|low","changed_facts":[{{"field":"...","old_value":"...","new_value":"..."}}],"explanation":"..."}}"""
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}},
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return {"change_type": "uncertain", "confidence": "low", "changed_facts": [], "explanation": "Semantic classifier unavailable."}
    if response.status_code != 200:
        return {"change_type": "uncertain", "confidence": "low", "changed_facts": [], "explanation": f"Semantic classifier returned HTTP {response.status_code}."}
    try:
        result = json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])
        if result.get("change_type") not in {"material", "equivalent", "formatting_only", "uncertain"}:
            raise ValueError
        return result
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return {"change_type": "uncertain", "confidence": "low", "changed_facts": [], "explanation": "Semantic classifier returned an unexpected response."}
