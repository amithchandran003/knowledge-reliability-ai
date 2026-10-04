import json
import os
import tempfile
import unittest

from answer_memory import append_validity_history, save_answer


class ValidityHistoryTests(unittest.TestCase):

    def test_invalidated_answer_is_recorded_as_stale(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            answer = save_answer(
                question="What is the refund period?",
                generated_answer="The refund period is 30 days.",
                source_identifier="Customer Refund Policy",
                source_content="Refund period is 30 days.",
                source_hash="old_hash_123",
                source_url="https://example.internal/refund-policy",
                source_id="refund_policy",
                directory=temp_dir,
            )

            updated = append_validity_history(
                answer_id=answer["answer_id"],
                status="stale",
                reason="source_changed_and_answer_re_evaluated",
                source_version_id="refund_policy@new_hash_456",
                directory=temp_dir,
            )

            self.assertIsNotNone(updated)

            history = updated["provenance"]["validity_history"]

            self.assertEqual(len(history), 2)

            self.assertEqual(
                history[-1]["status"],
                "stale"
            )

            self.assertEqual(
                history[-1]["reason"],
                "source_changed_and_answer_re_evaluated"
            )

            self.assertEqual(
                history[-1]["source_version_id"],
                "refund_policy@new_hash_456"
            )

            # Confirm it was actually persisted to disk.
            file_path = os.path.join(
                temp_dir,
                f"{answer['answer_id']}.json"
            )

            with open(file_path, "r", encoding="utf-8") as file:
                saved_record = json.load(file)

            self.assertEqual(
                saved_record["provenance"]["validity_history"][-1]["status"],
                "stale"
            )

    def test_v2_monitor_pipeline_records_stale_on_material_change(self):
        """Regression test: v2_monitor_pipeline records 'stale' status upon material source change."""
        from v2_monitor_pipeline import run_v2_monitor

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as answers_dir:
            source_cfg = {
                "source_id": "source_test",
                "source_type": "web_page",
                "source_identifier": "Test Source",
                "source_url": "https://example.com/policy",
            }

            stored_answer = save_answer(
                question="What is the refund period?",
                generated_answer="Refunds are accepted within 30 days.",
                source_identifier="Test Source",
                source_content="Refunds are accepted within 30 days.",
                source_url="https://example.com/policy",
                source_id="source_test",
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

            old_html = "<p>Refunds are accepted within 30 days.</p>"
            new_html = "<p>Refunds are accepted within 15 days.</p>"
            adapter = SequenceAdapter([old_html, new_html])

            mock_classifier = lambda old_t, new_t, api_key=None: {
                "change_type": "material",
                "confidence": "high",
                "changed_facts": [{"field": "refund_period", "old_value": "30 days", "new_value": "15 days"}],
                "explanation": "Refund period changed from 30 to 15 days.",
            }
            mock_reevaluator = lambda old_source, new_source, question, previous_answer, api_key=None: {
                "answer_still_valid": False,
                "affected_claim": "Refunds are accepted within 30 days.",
                "old_information": "30 days",
                "new_information": "15 days",
                "corrected_answer": "Refunds are accepted within 15 days.",
                "confidence": "high",
                "explanation": "Policy reduced to 15 days.",
            }

            # 1. Initial snapshot
            res1 = run_v2_monitor(
                source_cfg,
                [stored_answer],
                root,
                adapter,
                mock_classifier,
                mock_reevaluator,
                answer_directory=answers_dir,
            )
            self.assertEqual(res1["status"], "initial_snapshot")

            # 2. Material change run
            res2 = run_v2_monitor(
                source_cfg,
                [stored_answer],
                root,
                adapter,
                mock_classifier,
                mock_reevaluator,
                answer_directory=answers_dir,
            )
            self.assertEqual(res2["status"], "changed")
            self.assertEqual(len(res2["notifications"]), 1)

            # 3. Verify validity history updated to canonical 'stale'
            saved_file = os.path.join(answers_dir, f"{stored_answer['answer_id']}.json")
            with open(saved_file, "r", encoding="utf-8") as f:
                saved_rec = json.load(f)

            history = saved_rec["provenance"]["validity_history"]
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["status"], "stale")
            self.assertEqual(history[-1]["reason"], "v2_material_source_change")


if __name__ == "__main__":
    unittest.main()