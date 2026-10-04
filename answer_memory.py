# answer_memory.py
"""
Answer Memory Module.

Stores what evidence an AI answer depended on, linking each user question
and generated answer to its specific source document, content, content hash,
and retrieval timestamp.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

ANSWERS_DIR = os.path.join(os.path.dirname(__file__), "data", "answers")


def compute_hash(text: str) -> str:
    """Computes a SHA-256 hash for a given text snippet."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    """Creates a stable, readable identifier fragment from a source name."""
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part) or "unknown-source"


def is_uri(value: str | None) -> bool:
    """Returns True for a URL or URI with an explicit scheme."""
    return bool(value and urlparse(value).scheme)


def build_source_identity(source_identifier: str, source_url: str | None = None) -> dict:
    """
    Produces a stable source identity.

    Human-readable names belong in source_identifier.  When no actual URL or URI
    is available, policy documents use an internal URI derived from their title.
    """
    canonical_url = (
        source_url
        if is_uri(source_url)
        else f"internal://policy/{_slugify(source_identifier)}"
    )
    source_id = f"source_{compute_hash(canonical_url)[:16]}"
    return {
        "source_id": source_id,
        "source_identifier": source_identifier,
        "source_url": canonical_url,
    }


def build_source_version_id(source_id: str, source_hash: str) -> str:
    """Creates a stable identifier for one immutable source-content version."""
    return f"{source_id}@{source_hash}"
def extract_claims(generated_answer: str) -> list[dict]:
    """
    Extract simple sentence-level claims from a generated answer.

    This is intentionally deterministic and does not call an LLM.
    """
    import re

    if not generated_answer:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", generated_answer.strip())

    claims = []

    for sentence in sentences:
        text = sentence.strip()

        if not text:
            continue

        claims.append(
            {
                "claim_id": f"claim_{len(claims) + 1:03d}",
                "text": text,
                "extraction_method": "sentence_split_v2",
            }
        )

    return claims


def build_provenance(
    answer_id: str,
    generated_answer: str,
    source_identifier: str,
    source_url: str,
    source_content: str,
    source_hash: str,
    timestamp: str,
    source_id: str | None = None,
    evidence_items: list[dict] | None = None,
) -> dict:
    """
    Builds a provenance record with sentence-level claims.

    Claims are extracted deterministically without an LLM.
    Evidence is linked to claims using lexical token overlap.
    """

    import re

    identity = build_source_identity(source_identifier, source_url)

    if source_id:
        identity["source_id"] = source_id

    version_id = build_source_version_id(
        identity["source_id"],
        source_hash,
    )

    # ---------------------------------------------------------
    # BUILD EVIDENCE
    # ---------------------------------------------------------

    if evidence_items:
        evidence = []

        for index, item in enumerate(evidence_items, 1):
            text = item.get("text", "")

            evidence_entry = {
                "evidence_id": item.get(
                    "evidence_id",
                    f"evidence_{index:03d}",
                ),
                "text": text,
                "location": item.get("location", {}),
            }

            if item.get("chunk_id"):
                evidence_entry["chunk_id"] = item["chunk_id"]

            if item.get("source_version_id"):
                evidence_entry["source_version_id"] = item[
                    "source_version_id"
                ]

            evidence.append(evidence_entry)

    else:
        line_count = (
            source_content.count("\n") + 1
            if source_content
            else 0
        )

        evidence = [
            {
                "evidence_id": "evidence_001",
                "text": source_content,
                "location": {
                    "start_char": 0,
                    "end_char": len(source_content),
                    "line_start": 1 if source_content else None,
                    "line_end": line_count if source_content else None,
                },
            }
        ]

    # ---------------------------------------------------------
    # EXTRACT CLAIMS
    # ---------------------------------------------------------

    claims = extract_claims(generated_answer)

    # ---------------------------------------------------------
    # SIMPLE TOKENIZATION
    # ---------------------------------------------------------

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }

    def tokenize(text: str) -> set[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            token
            for token in tokens
            if token not in stop_words
        }

    # ---------------------------------------------------------
    # CLAIM → EVIDENCE RELATIONSHIPS
    # ---------------------------------------------------------

    claim_evidence = []

    # Preserve the old behavior for a one-claim answer:
    # every evidence item supports the single claim.
    if len(claims) == 1:

        claim_id = claims[0]["claim_id"]

        for evidence_item in evidence:
            claim_evidence.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_item["evidence_id"],
                    "relationship": "supported_by",
                }
            )

    else:

        for claim in claims:

            claim_tokens = tokenize(claim["text"])

            if not claim_tokens:
                continue

            for evidence_item in evidence:

                evidence_tokens = tokenize(
                    evidence_item.get("text", "")
                )

                if not evidence_tokens:
                    continue

                overlap = claim_tokens & evidence_tokens

                # Require at least two meaningful shared tokens.
                if len(overlap) >= 2:

                    claim_evidence.append(
                        {
                            "claim_id": claim["claim_id"],
                            "evidence_id": evidence_item["evidence_id"],
                            "relationship": "supported_by",
                        }
                    )

    # ---------------------------------------------------------
    # ANSWER → CLAIM RELATIONSHIPS
    # ---------------------------------------------------------

    answer_claims = [
        {
            "answer_id": answer_id,
            "claim_id": claim["claim_id"],
            "relationship": "contains_claim",
        }
        for claim in claims
    ]

    # ---------------------------------------------------------
    # FINAL PROVENANCE
    # ---------------------------------------------------------

    return {
        "provenance_schema_version": "2.0",

        "sources": [
            {
                **identity,
                "versions": [
                    {
                        "source_version_id": version_id,
                        "source_hash": source_hash,
                        "captured_at": timestamp,
                        "content": source_content,
                        "evidence": evidence,
                    }
                ],
            }
        ],

        "claims": claims,

        "relationships": {
            "answer_claims": answer_claims,
            "claim_evidence": claim_evidence,
        },

        "validity_history": [
            {
                "status": "current",
                "recorded_at": timestamp,
                "reason": "answer_created",
                "source_version_id": version_id,
            }
        ],
    }


def save_answer(
    question: str,
    generated_answer: str,
    source_identifier: str,
    source_content: str,
    source_hash: str = None,
    source_url: str = None,
    similarity_score: float = None,
    source_id: str = None,
    directory: str = ANSWERS_DIR,
    evidence_items: list[dict] | None = None,
) -> dict:
    """
    Saves an answer record linking the generated answer to its source evidence.
    Stores the record as an individual JSON file in data/answers/.
    """
    os.makedirs(directory, exist_ok=True)

    answer_id = f"ans_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if not source_hash:
        source_hash = compute_hash(source_content)

    identity = build_source_identity(source_identifier, source_url)
    if source_id:
        identity["source_id"] = source_id

    source_version_id = build_source_version_id(
        identity["source_id"],
        source_hash,
    )

    record = {
        "schema_version": "1.0",
        "answer_id": answer_id,
        "timestamp": timestamp,
        "question": question,
        "generated_answer": generated_answer,
        "source_identifier": identity["source_identifier"],
        "source_url": identity["source_url"],
        "source_content": source_content,
        "source_hash": source_hash,
        "source_id": identity["source_id"],
        "source_version_id": source_version_id,
        "similarity_score": (
            round(similarity_score, 4)
            if similarity_score is not None
            else None
        ),
    }

    record["provenance"] = build_provenance(
        answer_id=answer_id,
        generated_answer=generated_answer,
        source_identifier=identity["source_identifier"],
        source_url=identity["source_url"],
        source_content=source_content,
        source_hash=source_hash,
        timestamp=timestamp,
        source_id=identity["source_id"],
        evidence_items=evidence_items,
    )

    file_path = os.path.join(directory, f"{answer_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


def load_answers(directory: str = ANSWERS_DIR) -> list[dict]:
    """
    Loads all saved answer records from data/answers/ sorted by timestamp.
    """
    if not os.path.exists(directory):
        return []

    records = []

    for filename in os.listdir(directory):
        if filename.startswith("ans_") and filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    records.append(json.load(f))
                except json.JSONDecodeError:
                    continue

    records.sort(key=lambda r: r.get("timestamp", ""))

    return records


def get_answer(answer_id: str, directory: str = ANSWERS_DIR) -> dict | None:
    """
    Retrieves a single answer record by its unique answer_id.
    """
    file_path = os.path.join(directory, f"{answer_id}.json")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_validity_history(
    answer_id: str,
    status: str,
    reason: str,
    source_version_id: str,
    directory: str = ANSWERS_DIR,
) -> dict | None:
    """Appends a V1 provenance validity event for an existing answer record."""
    file_path = os.path.join(directory, f"{answer_id}.json")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    provenance = record.get("provenance")

    if not provenance:
        return None

    provenance.setdefault("validity_history", []).append(
        {
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_version_id": source_version_id,
        }
    )

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)

    return record


def list_answers(directory: str = ANSWERS_DIR) -> list[dict]:
    """
    Prints a human-readable list of all stored answers and returns the records.
    """
    answers = load_answers(directory)

    print("=" * 70)
    print(f"STORED ANSWER MEMORY ({len(answers)} records)")
    print("=" * 70)

    if not answers:
        print("No answer records found.")
        print("=" * 70)
        return []

    for i, a in enumerate(answers, 1):
        print(f"[{i}] Answer ID: {a['answer_id']}")
        print(f"    Timestamp: {a['timestamp']}")
        print(f"    Question:  {a['question']}")
        print(
            f"    Source:    {a['source_identifier']} "
            f"(Hash: {a['source_hash'][:12]}...)"
        )
        print(f"    Answer:    {a['generated_answer'][:100]}...")
        print("-" * 70)

    return answers


if __name__ == "__main__":
    list_answers()