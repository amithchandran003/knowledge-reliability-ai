# notification.py
"""
Notification Layer Module.

Generates structured, user-facing notifications when a previously delivered AI
answer has been invalidated or made outdated by a verified source change.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from answer_revaluator import reevaluate_answer

NOTIFICATIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "notifications")


def create_notification(
    question: str,
    previous_answer: str,
    reevaluation_result: dict,
    source_identifier: str = None,
    answer_id: str = None,
    source_id: str = None,
    source_url: str = None,
    notification_id: str = None,
) -> dict | None:
    """
    Creates a notification dictionary ONLY when answer_still_valid is False.
    Returns None if the previous answer remains valid.
    """
    # 1. Check validity: If still valid, suppress notification
    if reevaluation_result.get("answer_still_valid") is True:
        return None

    notif_id = notification_id or f"notif_{uuid.uuid4().hex[:8]}"

    # 2. Construct notification record using only provided information
    notification = {
        "notification_id": notif_id,
        "title": "Your previous answer has changed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_identifier": source_identifier or "Referenced Source",
        "question": question,
        "previous_answer": previous_answer,
        "affected_claim": reevaluation_result.get("affected_claim"),
        "old_information": reevaluation_result.get("old_information"),
        "new_information": reevaluation_result.get("new_information"),
        "corrected_answer": reevaluation_result.get("corrected_answer"),
        "explanation": reevaluation_result.get("explanation"),
    }

    if answer_id:
        notification["answer_id"] = answer_id
    if source_id:
        notification["source_id"] = source_id
    if source_url:
        notification["source_url"] = source_url

    return notification


def save_notification(
    notification: dict,
    directory: str = NOTIFICATIONS_DIR,
) -> str:
    """
    Saves a notification dictionary to a JSON file in the notifications directory.
    """
    os.makedirs(directory, exist_ok=True)
    notif_id = notification.get("notification_id") or f"notif_{uuid.uuid4().hex[:8]}"
    notification["notification_id"] = notif_id
    file_path = os.path.join(directory, f"{notif_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(notification, f, indent=2)
    return file_path


def load_notifications(
    directory: str = NOTIFICATIONS_DIR,
) -> list[dict]:
    """
    Loads all saved notifications from the notification directory,
    sorted by timestamp descending (newest first).
    """
    if not os.path.exists(directory):
        return []

    notifications = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    notifications.append(json.load(f))
            except Exception:
                pass

    notifications.sort(
        key=lambda n: n.get("timestamp", ""),
        reverse=True,
    )
    return notifications


def get_notification(
    notification_id: str,
    directory: str = NOTIFICATIONS_DIR,
) -> dict | None:
    """Returns a specific notification by its ID if it exists."""
    file_path = os.path.join(directory, f"{notification_id}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_notification(notification: dict | None) -> str:
    """
    Converts a notification dictionary into a human-readable terminal alert.
    """
    if not notification:
        return "No notification generated (answer remains valid)."

    lines = [
        "=" * 65,
        f"🔔 ALERT: {notification['title']}",
        "=" * 65,
        f"Source:           {notification.get('source_identifier', 'N/A')}",
        f"Original Question: {notification['question']}",
        "",
        f"Previous Answer:   {notification['previous_answer']}",
        "",
        f"Affected Claim:    {notification.get('affected_claim') or 'N/A'}",
        f"Changed Info:      {notification.get('old_information')} → {notification.get('new_information')}",
        "",
        f"Updated Answer:    {notification.get('corrected_answer') or 'N/A'}",
        "",
        f"Explanation:       {notification.get('explanation') or 'N/A'}",
        "=" * 65,
    ]
    return "\n".join(lines)


def run_tests():
    print("=== Running Notification Layer Tests ===\n")

    # TEST A: Substantive policy change (Refund 30d -> 15d)
    print("TEST A: Policy change invalidating previous answer (30 days -> 15 days)")
    question_a = "What is the refund period for customer purchases?"
    previous_answer_a = "According to the policy, the refund period is 30 days."
    old_source_a = "Refund period is 30 days."
    new_source_a = "Refund period is 15 days."

    # Re-evaluate with Gemini
    result_a = reevaluate_answer(
        old_source=old_source_a,
        new_source=new_source_a,
        question=question_a,
        previous_answer=previous_answer_a,
    )

    # Generate notification
    notification_a = create_notification(
        question=question_a,
        previous_answer=previous_answer_a,
        reevaluation_result=result_a,
        source_identifier="Customer Refund Policy",
    )

    print(f"Result returned: {'dict created' if notification_a else 'None'}")
    print(format_notification(notification_a))
    print("\n" + "-" * 65 + "\n")

    # TEST B: Unrelated change (Support channel changed, refund stays 30d)
    print("TEST B: Unrelated change to support channel (refund remains 30 days)")
    question_b = "What is the refund period for customer purchases?"
    previous_answer_b = "According to the policy, the refund period is 30 days."
    old_source_b = "Refund period is 30 days. Contact support by email."
    new_source_b = "Refund period is 30 days. Contact support through the help center."

    # Re-evaluate with Gemini
    result_b = reevaluate_answer(
        old_source=old_source_b,
        new_source=new_source_b,
        question=question_b,
        previous_answer=previous_answer_b,
    )

    # Generate notification
    notification_b = create_notification(
        question=question_b,
        previous_answer=previous_answer_b,
        reevaluation_result=result_b,
        source_identifier="Customer Refund Policy",
    )

    print(f"Result returned: {notification_b}")
    print(format_notification(notification_b))


if __name__ == "__main__":
    run_tests()
