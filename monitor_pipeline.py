# monitor_pipeline.py
"""
End-to-End Monitoring Pipeline.

Integrates:
1. source_monitor: Snapshot fetching and hash tracking
2. change_analyzer: Sentence-level diff analysis
3. answer_memory: Loading and storing answers with evidence linkage
4. answer_impact: Identifying historical answers tied to changed sources
5. answer_revaluator: Semantic re-evaluation of affected claims using Gemini
6. notification: User-facing alert generation for invalidated answers
"""

import hashlib
import sys
from datetime import datetime, timezone

from source_monitor import (
    DEFAULT_URL,
    SNAPSHOT_DIR,
    fetch_source_snapshot,
    get_latest_snapshot,
    save_snapshot,
)
from change_analyzer import analyze_content_diff
from answer_memory import load_answers
from answer_impact import find_affected_answers
from answer_revaluator import reevaluate_answer
from notification import create_notification, format_notification


def run_pipeline(
    current_snapshot: dict,
    previous_snapshot: dict | None = None,
    answers: list | None = None,
) -> dict:
    """
    Executes the end-to-end monitoring pipeline on current and previous snapshots.

    Stages:
      1. Change Detection
      2. Sentence Diff Analysis
      3. Answer Impact Matching
      4. Claim-Level Re-evaluation
      5. Notification Generation
      6. Run Summary
    """
    source_url = current_snapshot.get("source_url", "Unknown Source")
    print("\n" + "=" * 65)
    print(f"PIPELINE RUN: {source_url}")
    print("=" * 65)

    # -------------------------------------------------------------
    # Stage 1: Change Detection
    # -------------------------------------------------------------
    if previous_snapshot is None:
        print("[Stage 1] Initial snapshot recorded. No previous version to compare.")
        return {
            "source_changed": False,
            "status": "initial_snapshot",
            "message": "Initial snapshot established.",
            "notifications": [],
        }

    prev_hash = previous_snapshot.get("content_hash")
    curr_hash = current_snapshot.get("content_hash")

    if curr_hash == prev_hash:
        print(f"[Stage 1] No meaningful source change detected for {source_url}.")
        print(f"         Hash: {curr_hash[:16]}... (unchanged)")
        return {
            "source_changed": False,
            "status": "unchanged",
            "message": "No meaningful source change detected. Pipeline stopped.",
            "notifications": [],
        }

    print(f"[Stage 1] Source change detected!")
    print(f"         Old Hash: {prev_hash[:16]}...")
    print(f"         New Hash: {curr_hash[:16]}...")

    # -------------------------------------------------------------
    # Stage 2: Change Analysis
    # -------------------------------------------------------------
    try:
        diff_result = analyze_content_diff(
            previous_snapshot.get("content", ""),
            current_snapshot.get("content", ""),
        )
    except Exception as err:
        raise RuntimeError(f"[Stage 2: Change Analysis Failed]: {err}") from err

    added_units = diff_result.get("added", [])
    removed_units = diff_result.get("removed", [])
    print(f"[Stage 2] Change Analysis: {len(added_units)} added, {len(removed_units)} removed units.")

    if removed_units:
        print("         Removed:")
        for r in removed_units:
            print(f"           - {r}")
    if added_units:
        print("         Added:")
        for a in added_units:
            print(f"           + {a}")

    # -------------------------------------------------------------
    # Stage 3: Answer Impact Identification
    # -------------------------------------------------------------
    try:
        affected_answers = find_affected_answers(
            prev_hash,
            source_id=current_snapshot.get("source_id"),
            answers=answers,
        )
    except Exception as err:
        raise RuntimeError(f"[Stage 3: Answer Impact Failed]: {err}") from err

    print(f"[Stage 3] Answer Impact: Found {len(affected_answers)} historical answer(s) tied to previous source.")

    if not affected_answers:
        print("         No stored answers depended on this source version. Pipeline finished.")
        return {
            "source_changed": True,
            "status": "changed_no_affected_answers",
            "added_units": added_units,
            "removed_units": removed_units,
            "potentially_affected_count": 0,
            "remained_valid_count": 0,
            "became_invalid_count": 0,
            "notifications": [],
        }

    # -------------------------------------------------------------
    # Stages 4 & 5: Semantic Re-evaluation & Notifications
    # -------------------------------------------------------------
    valid_count = 0
    invalid_count = 0
    notifications = []

    print(f"[Stage 4 & 5] Re-evaluating {len(affected_answers)} answer(s)...")

    for idx, answer_record in enumerate(affected_answers, 1):
        ans_id = answer_record.get("answer_id", f"ans_{idx}")
        question = answer_record.get("question", "")
        prev_answer = answer_record.get("generated_answer", "")
        old_source_content = answer_record.get("source_content") or previous_snapshot.get("content", "")
        new_source_content = current_snapshot.get("content", "")
        source_id = answer_record.get("source_identifier") or source_url

        print(f"\n  [{idx}/{len(affected_answers)}] Evaluating Answer ID: {ans_id}")
        print(f"      Question: {question}")
        print(f"      Answer:   {prev_answer}")

        try:
            reeval = reevaluate_answer(
                old_source=old_source_content,
                new_source=new_source_content,
                question=question,
                previous_answer=prev_answer,
            )
        except Exception as err:
            raise RuntimeError(f"[Stage 4: Re-evaluation Failed for {ans_id}]: {err}") from err

        is_valid = reeval.get("answer_still_valid", False)

        if is_valid:
            valid_count += 1
            print(f"      Status:   STILL VALID (No notification needed)")
            print(f"      Reason:   {reeval.get('explanation')}")
        else:
            invalid_count += 1
            print(f"      Status:   INVALIDATED / OUTDATED")
            print(f"      Delta:    {reeval.get('old_information')} -> {reeval.get('new_information')}")
            print(f"      Fix:      {reeval.get('corrected_answer')}")

            # Generate and format notification
            notif = create_notification(
                question=question,
                previous_answer=prev_answer,
                reevaluation_result=reeval,
                source_identifier=source_id,
            )
            if notif:
                notifications.append(notif)
                print("\n" + format_notification(notif))

    # -------------------------------------------------------------
    # Stage 6: Summary Report
    # -------------------------------------------------------------
    summary = {
        "source_changed": True,
        "additions_count": len(added_units),
        "removals_count": len(removed_units),
        "potentially_affected_count": len(affected_answers),
        "remained_valid_count": valid_count,
        "became_invalid_count": invalid_count,
        "notifications_generated": len(notifications),
        "notifications": notifications,
    }

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Source Changed:                 Yes")
    print(f"Added Units:                    {summary['additions_count']}")
    print(f"Removed Units:                  {summary['removals_count']}")
    print(f"Historical Answers Evaluated:   {summary['potentially_affected_count']}")
    print(f"  - Answers Remaining Valid:    {summary['remained_valid_count']}")
    print(f"  - Answers Becoming Invalid:   {summary['became_invalid_count']}")
    print(f"Notifications Generated:        {summary['notifications_generated']}")
    print("=" * 65)

    return summary


def run_controlled_scenario_a():
    """
    Scenario A: Policy change invalidates the previous answer.
    Old: Refund period is 30 days.
    New: Refund period is 15 days.
    Expected: Answer invalidated, 1 notification generated.
    """
    print("\n" + "#" * 65)
    print("CONTROLLED TEST A: Policy Change (30 days -> 15 days)")
    print("#" * 65)

    old_content = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 30 days.\n"
        "Contact support for billing inquiries."
    )
    new_content = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 15 days.\n"
        "Contact support for billing inquiries."
    )

    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()
    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

    old_snapshot = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": old_hash,
        "content": old_content,
    }

    new_snapshot = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": new_hash,
        "content": new_content,
    }

    # Historical answer grounded on old_snapshot
    test_answers = [
        {
            "answer_id": "ans_demo_refund",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": "What is the refund period for customer purchases?",
            "generated_answer": "According to the policy, the refund period is 30 days.",
            "source_identifier": "Customer Refund Policy",
            "source_url": "https://example.internal/refund-policy",
            "source_content": old_content,
            "source_hash": old_hash,
        }
    ]

    return run_pipeline(
        current_snapshot=new_snapshot,
        previous_snapshot=old_snapshot,
        answers=test_answers,
    )


def run_controlled_scenario_b():
    """
    Scenario B: Unrelated change in source document.
    Old: Refund period is 30 days. Contact support by email.
    New: Refund period is 30 days. Contact support through the help center.
    Expected: Answer remains valid, 0 notifications generated.
    """
    print("\n" + "#" * 65)
    print("CONTROLLED TEST B: Unrelated Change (Support Channel Updated)")
    print("#" * 65)

    old_content = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 30 days.\n"
        "Contact support by email."
    )
    new_content = (
        "Customer satisfaction is our priority.\n"
        "Refund period is 30 days.\n"
        "Contact support through the help center."
    )

    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()
    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

    old_snapshot = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": old_hash,
        "content": old_content,
    }

    new_snapshot = {
        "source_url": "https://example.internal/refund-policy",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": new_hash,
        "content": new_content,
    }

    # Historical answer grounded on old_snapshot
    test_answers = [
        {
            "answer_id": "ans_demo_support",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": "What is the refund period for customer purchases?",
            "generated_answer": "According to the policy, the refund period is 30 days.",
            "source_identifier": "Customer Refund Policy",
            "source_url": "https://example.internal/refund-policy",
            "source_content": old_content,
            "source_hash": old_hash,
        }
    ]

    return run_pipeline(
        current_snapshot=new_snapshot,
        previous_snapshot=old_snapshot,
        answers=test_answers,
    )


if __name__ == "__main__":
    print("=== Running End-to-End Monitoring Pipeline Controlled Tests ===")
    run_controlled_scenario_a()
    run_controlled_scenario_b()
