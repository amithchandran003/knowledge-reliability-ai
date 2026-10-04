# test_notification.py
"""
Unit tests for notification creation, formatting, and persistent disk storage.
"""

import json
import os
import tempfile
import unittest

from answer_memory import save_answer
from notification import (
    create_notification,
    format_notification,
    get_notification,
    load_notifications,
    save_notification,
)
from v2_monitor_pipeline import run_v2_monitor


class NotificationPersistenceTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.notifs_dir = os.path.join(self.temp_dir.name, "notifications")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_notification_creation_and_fields(self):
        reeval_result = {
            "answer_still_valid": False,
            "affected_claim": "Refunds are 30 days.",
            "old_information": "30 days",
            "new_information": "15 days",
            "corrected_answer": "Refunds are 15 days.",
            "confidence": "high",
            "explanation": "Policy shortened to 15 days.",
        }

        notif = create_notification(
            question="What is the refund period?",
            previous_answer="The refund period is 30 days.",
            reevaluation_result=reeval_result,
            source_identifier="Customer Refund Policy",
            answer_id="ans_123",
            source_id="source_refund",
            source_url="https://example.com/refunds",
        )

        self.assertIsNotNone(notif)
        self.assertTrue(notif["notification_id"].startswith("notif_"))
        self.assertEqual(notif["answer_id"], "ans_123")
        self.assertEqual(notif["source_id"], "source_refund")
        self.assertEqual(notif["source_url"], "https://example.com/refunds")
        self.assertEqual(notif["old_information"], "30 days")
        self.assertEqual(notif["new_information"], "15 days")
        self.assertEqual(notif["corrected_answer"], "Refunds are 15 days.")

        formatted = format_notification(notif)
        self.assertIn("30 days → 15 days", formatted)
        self.assertIn("Customer Refund Policy", formatted)

    def test_notification_suppression_when_answer_valid(self):
        reeval_result = {
            "answer_still_valid": True,
            "affected_claim": None,
            "old_information": None,
            "new_information": None,
            "corrected_answer": None,
            "confidence": "high",
            "explanation": "Answer remains factually supported.",
        }

        notif = create_notification(
            question="What is the refund period?",
            previous_answer="The refund period is 30 days.",
            reevaluation_result=reeval_result,
            source_identifier="Customer Refund Policy",
        )
        self.assertIsNone(notif)

    def test_save_and_load_notifications(self):
        notif1 = {
            "notification_id": "notif_001",
            "title": "Your previous answer has changed",
            "timestamp": "2026-09-30T10:00:00Z",
            "question": "Question 1",
            "previous_answer": "Answer 1",
            "old_information": "Old 1",
            "new_information": "New 1",
            "corrected_answer": "Corrected 1",
            "explanation": "Reason 1",
        }
        notif2 = {
            "notification_id": "notif_002",
            "title": "Your previous answer has changed",
            "timestamp": "2026-09-30T11:00:00Z",
            "question": "Question 2",
            "previous_answer": "Answer 2",
            "old_information": "Old 2",
            "new_information": "New 2",
            "corrected_answer": "Corrected 2",
            "explanation": "Reason 2",
        }

        path1 = save_notification(notif1, directory=self.notifs_dir)
        path2 = save_notification(notif2, directory=self.notifs_dir)

        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))

        loaded = load_notifications(directory=self.notifs_dir)
        self.assertEqual(len(loaded), 2)
        # Should be sorted descending by timestamp
        self.assertEqual(loaded[0]["notification_id"], "notif_002")
        self.assertEqual(loaded[1]["notification_id"], "notif_001")

        single = get_notification("notif_001", directory=self.notifs_dir)
        self.assertIsNotNone(single)
        self.assertEqual(single["question"], "Question 1")

    def test_v2_monitor_pipeline_persists_notifications_on_material_change(self):
        with tempfile.TemporaryDirectory() as snapshot_root, tempfile.TemporaryDirectory() as answers_dir:
            source_cfg = {
                "source_id": "source_persist_test",
                "source_type": "web_page",
                "source_identifier": "Cancellation Terms",
                "source_url": "https://example.com/cancellation",
            }

            stored_answer = save_answer(
                question="What is the cancellation notice period?",
                generated_answer="Notice must be given 30 days in advance.",
                source_identifier="Cancellation Terms",
                source_content="Notice must be given 30 days in advance.",
                source_url="https://example.com/cancellation",
                source_id="source_persist_test",
                directory=answers_dir,
            )

            class SequenceAdapter:
                def __init__(self, contents):
                    self.contents = list(contents)

                def fetch(self, config):
                    from source_adapters.base import FetchResult
                    return FetchResult(
                        success=True,
                        raw_content=self.contents.pop(0),
                        content_type="text/html",
                        status_code=200,
                    )

            old_html = "<p>Notice must be given 30 days in advance.</p>"
            new_html = "<p>Notice must be given 10 days in advance.</p>"
            adapter = SequenceAdapter([old_html, new_html])

            mock_classifier = lambda old_t, new_t, api_key=None: {
                "change_type": "material",
                "confidence": "high",
                "changed_facts": [{"field": "notice_period", "old_value": "30 days", "new_value": "10 days"}],
                "explanation": "Notice period reduced to 10 days.",
            }
            mock_reevaluator = lambda old_source, new_source, question, previous_answer, api_key=None: {
                "answer_still_valid": False,
                "affected_claim": "Notice must be given 30 days in advance.",
                "old_information": "30 days",
                "new_information": "10 days",
                "corrected_answer": "Notice must be given 10 days in advance.",
                "confidence": "high",
                "explanation": "Policy updated from 30 to 10 days.",
            }

            # 1. Initial snapshot run
            res1 = run_v2_monitor(
                source_cfg,
                [stored_answer],
                snapshot_root,
                adapter,
                mock_classifier,
                mock_reevaluator,
                answer_directory=answers_dir,
                notifications_directory=self.notifs_dir,
            )
            self.assertEqual(res1["status"], "initial_snapshot")
            self.assertEqual(len(load_notifications(self.notifs_dir)), 0)

            # 2. Material change run
            res2 = run_v2_monitor(
                source_cfg,
                [stored_answer],
                snapshot_root,
                adapter,
                mock_classifier,
                mock_reevaluator,
                answer_directory=answers_dir,
                notifications_directory=self.notifs_dir,
            )
            self.assertEqual(res2["status"], "changed")
            self.assertEqual(len(res2["notifications"]), 1)

            # 3. Verify notification was persisted to disk and loaded back
            persisted = load_notifications(self.notifs_dir)
            self.assertEqual(len(persisted), 1)
            notif = persisted[0]
            self.assertEqual(notif["answer_id"], stored_answer["answer_id"])
            self.assertEqual(notif["source_id"], "source_persist_test")
            self.assertEqual(notif["old_information"], "30 days")
            self.assertEqual(notif["new_information"], "10 days")
            self.assertEqual(notif["corrected_answer"], "Notice must be given 10 days in advance.")


if __name__ == "__main__":
    unittest.main()
