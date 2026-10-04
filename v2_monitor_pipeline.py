"""V2 vertical slice: adapter ingestion through provenance-aware notifications."""

import json
import sys

from answer_memory import ANSWERS_DIR, append_validity_history, load_answers
from answer_revaluator import reevaluate_answer
from claim_impact import select_affected_claims
from content_normalizer import normalize_fetch_result
from notification import NOTIFICATIONS_DIR, create_notification, save_notification
from semantic_change import align_and_diff_chunks, classify_semantic_change
from snapshot_store import build_snapshot, get_latest_snapshot, record_source_status, save_snapshot
from source_adapters import get_adapter
from source_registry import get_source


def run_v2_monitor(
    source_config: dict,
    answers: list[dict] | None = None,
    snapshot_root: str | None = None,
    adapter=None,
    semantic_classifier=classify_semantic_change,
    reevaluator=reevaluate_answer,
    persist_validity: bool = True,
    answer_directory: str = ANSWERS_DIR,
    persist_notifications: bool = True,
    notifications_directory: str = NOTIFICATIONS_DIR,
) -> dict:
    """Runs one source through V2 without changing the V0.8 monitor pipeline."""
    adapter = adapter or get_adapter(source_config["source_type"])
    fetch = adapter.fetch(source_config)
    root_kwargs = {"root": snapshot_root} if snapshot_root else {}

    if not fetch.success:
        record_source_status(source_config["source_id"], {"status": "unavailable", "error_code": fetch.error_code, "http_status": fetch.status_code}, **root_kwargs)
        return {"status": "unavailable", "source_id": source_config["source_id"], "error_code": fetch.error_code, "notifications": []}

    try:
        current = build_snapshot(normalize_fetch_result(source_config, fetch.raw_content))
    except (TypeError, ValueError):
        record_source_status(source_config["source_id"], {"status": "normalization_failed", "error_code": "invalid_source_content"}, **root_kwargs)
        return {"status": "normalization_failed", "source_id": source_config["source_id"], "notifications": []}
    previous = get_latest_snapshot(source_config["source_id"], **root_kwargs)
    save_snapshot(current, **root_kwargs)
    record_source_status(source_config["source_id"], {"status": "available", "source_version_id": current["source_version_id"]}, **root_kwargs)

    if previous is None:
        return {"status": "initial_snapshot", "source_id": source_config["source_id"], "notifications": []}
    if previous["content_hash"] == current["content_hash"]:
        return {"status": "unchanged", "source_id": source_config["source_id"], "notifications": []}

    changes = align_and_diff_chunks(previous["chunks"], current["chunks"])
    selections = select_affected_claims(previous, changes, answers if answers is not None else load_answers())
    notifications = []
    reevaluations = []
    semantic_results = {}
    for selection in selections:
        change = selection["change"]
        old_text = change.get("old_chunk", {}).get("text", "")
        new_text = change.get("new_chunk", {}).get("text", "")
        change_key = (old_text, new_text)
        if change_key not in semantic_results:
            semantic_results[change_key] = semantic_classifier(old_text, new_text)
        semantic = semantic_results[change_key]
        if semantic.get("change_type") not in {"material", "uncertain"}:
            continue
        answer = selection["answer"]
        result = reevaluator(
            old_source=previous["normalized_content"],
            new_source=current["normalized_content"],
            question=answer["question"],
            previous_answer=answer["generated_answer"],
        )
        reevaluations.append({"answer_id": answer["answer_id"], "claim_id": selection["claim"]["claim_id"], "semantic_change": semantic, "result": result})
        if persist_validity:
            status = "current" if result.get("answer_still_valid") else "stale"
            append_validity_history(
                answer["answer_id"],
                status=status,
                reason=f"v2_{semantic.get('change_type', 'uncertain')}_source_change",
                source_version_id=current["source_version_id"],
                directory=answer_directory,
            )
        notification = create_notification(
            question=answer["question"],
            previous_answer=answer["generated_answer"],
            reevaluation_result=result,
            source_identifier=answer.get("source_identifier") or source_config.get("source_identifier"),
            answer_id=answer.get("answer_id"),
            source_id=source_config.get("source_id"),
            source_url=source_config.get("source_url"),
        )
        if notification:
            if persist_notifications:
                save_notification(notification, directory=notifications_directory)
            notifications.append(notification)

    return {
        "status": "changed",
        "source_id": source_config["source_id"],
        "changes": changes,
        "candidate_claims": len(selections),
        "reevaluations": reevaluations,
        "notifications": notifications,
    }


def main():
    """Runs one configured V2 source from the command line."""
    if len(sys.argv) != 2:
        print("Usage: python3 v2_monitor_pipeline.py <source_id>")
        return
    source = get_source(sys.argv[1])
    if source is None:
        print(f"Unknown source_id: {sys.argv[1]}")
        return
    result = run_v2_monitor(source)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
