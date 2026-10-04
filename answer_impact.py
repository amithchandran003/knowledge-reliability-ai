# answer_impact.py
"""
Answer Impact Analyzer Module.

Identifies which previously stored answers in Answer Memory depended on a source
that has now changed, flagging them as potentially stale or requiring re-evaluation.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from answer_memory import load_answers, save_answer
from change_analyzer import analyze_content_diff


def find_affected_answers(
    previous_source_hash: str,
    source_id: str | None = None,
    answers: list = None,
) -> list:
    """
    Deterministic matching: prefers source_id plus source_hash when source_id is
    available. Legacy records without source_id retain V0.8 hash-only matching.
    """
    if answers is None:
        answers = load_answers()

    affected = []
    for answer in answers:
        if answer.get("source_hash") != previous_source_hash:
            continue

        answer_source_id = answer.get("source_id")
        if source_id is None:
            affected.append(answer)
        elif answer_source_id == source_id or answer_source_id is None:
            affected.append(answer)

    return affected


def format_impact_report(
    old_snapshot: dict,
    new_snapshot: dict,
    diff_result: dict,
    affected_answers: list,
) -> str:
    """
    Produces a clear, human-readable report showing the source changes
    and listing all potentially affected previous answers.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("SOURCE CHANGE & ANSWER IMPACT REPORT")
    lines.append("=" * 65)
    lines.append("SOURCE CHANGE DETECTED")
    lines.append(f"Source:    {new_snapshot.get('source_url', 'Unknown')}")
    lines.append(f"Old Hash:  {old_snapshot.get('content_hash', 'N/A')[:16]}...")
    lines.append(f"New Hash:  {new_snapshot.get('content_hash', 'N/A')[:16]}...")
    lines.append("")

    if diff_result.get("removed"):
        lines.append("Old information (removed):")
        for r in diff_result["removed"]:
            lines.append(f"  • {r}")
        lines.append("")

    if diff_result.get("added"):
        lines.append("New information (added):")
        for a in diff_result["added"]:
            lines.append(f"  • {a}")
        lines.append("")

    lines.append("-" * 65)
    lines.append(f"POTENTIALLY AFFECTED ANSWERS: {len(affected_answers)}")
    lines.append("-" * 65)

    if not affected_answers:
        lines.append("No previous answers were found that depended on this source version.")
    else:
        for i, ans in enumerate(affected_answers, 1):
            lines.append(f"[{i}] Answer ID:       {ans['answer_id']}")
            lines.append(f"    User Question:   {ans['question']}")
            lines.append(f"    Previous Answer: {ans['generated_answer']}")
            lines.append(f"    Source Used:     {ans['source_identifier']} (Hash: {ans['source_hash'][:12]}...)")
            lines.append(f"    Timestamp:       {ans['timestamp']}")
            lines.append("    Reason:          Answer used the previous version of this source.")
            lines.append("")

    lines.append("=" * 65)
    return "\n".join(lines)


def run_controlled_test():
    """
    Controlled test demonstration:
    1. An existing answer is stored that depends on: 'Refund period is 30 days.'
    2. A new version of the source is published saying: 'Refund period is 15 days.'
    3. The system detects the change and flags the 30-day answer as affected.
    """
    # 1. Prepare old source version
    content_v1 = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 30 days.\n"
        "Contact support for billing inquiries."
    )
    hash_v1 = hashlib.sha256(content_v1.encode("utf-8")).hexdigest()
    snapshot_v1 = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": hash_v1,
        "content": content_v1,
    }

    # 2. Store an answer record that relied on this exact source v1
    answer_record = save_answer(
        question="What is the refund window for customer purchases?",
        generated_answer="According to the policy, the refund period is 30 days.",
        source_identifier="Customer Refund Policy",
        source_content=content_v1,
        source_hash=hash_v1,
        source_url=snapshot_v1["source_url"],
    )

    # 3. Prepare new source version (changed to 15 days)
    content_v2 = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 15 days.\n"
        "Contact support for billing inquiries."
    )
    hash_v2 = hashlib.sha256(content_v2.encode("utf-8")).hexdigest()
    snapshot_v2 = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": hash_v2,
        "content": content_v2,
    }

    # 4. Analyze content differences
    diff_result = analyze_content_diff(snapshot_v1["content"], snapshot_v2["content"])

    # 5. Identify affected answers
    affected = find_affected_answers(snapshot_v1["content_hash"])

    # 6. Generate and print the report
    report = format_impact_report(
        old_snapshot=snapshot_v1,
        new_snapshot=snapshot_v2,
        diff_result=diff_result,
        affected_answers=affected,
    )
    print(report)


if __name__ == "__main__":
    run_controlled_test()
