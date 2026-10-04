import hashlib
import tempfile
import unittest

from answer_memory import save_answer
from v2_monitor_pipeline import run_v2_monitor


class MultiClaimMonitorTests(unittest.TestCase):

    def test_changed_evidence_affects_only_correct_claim(self):

        with tempfile.TemporaryDirectory() as snapshot_root, \
             tempfile.TemporaryDirectory() as answers_dir, \
             tempfile.TemporaryDirectory() as notifications_dir:

            source_cfg = {
                "source_id": "multi_claim_test",
                "source_type": "web_page",
                "source_identifier": "Python Data Tools",
                "source_url": "https://example.com/python-tools",
            }

            old_content = (
                "Python is widely used for data analysis.\n"
                "Pandas provides tools for working with tabular data.\n"
                "NumPy supports numerical computing."
            )

            new_content = (
                "Python is widely used for data analysis.\n"
                "Pandas provides tools for working with time-series data.\n"
                "NumPy supports numerical computing."
            )

            old_hash = hashlib.sha256(
                old_content.encode("utf-8")
            ).hexdigest()

            answer = save_answer(
                question="What are Python, Pandas and NumPy used for?",
                generated_answer=(
                    "Python is widely used for data analysis. "
                    "Pandas provides tools for working with tabular data. "
                    "NumPy supports numerical computing."
                ),
                source_identifier="Python Data Tools",
                source_content=old_content,
                source_hash=old_hash,
                source_url="https://example.com/python-tools",
                source_id="multi_claim_test",
                directory=answers_dir,
                evidence_items=[
                    {
                        "evidence_id": "evidence_001",
                        "text": "Python is widely used for data analysis.",
                        "chunk_id": "python_chunk_001",
                        "source_version_id": f"multi_claim_test@{old_hash}",
                    },
                    {
                        "evidence_id": "evidence_002",
                        "text": "Pandas provides tools for working with tabular data.",
                        "chunk_id": "pandas_chunk_001",
                        "source_version_id": f"multi_claim_test@{old_hash}",
                    },
                    {
                        "evidence_id": "evidence_003",
                        "text": "NumPy supports numerical computing.",
                        "chunk_id": "numpy_chunk_001",
                        "source_version_id": f"multi_claim_test@{old_hash}",
                    },
                ],
            )

            class SequenceAdapter:

                def __init__(self):
                    self.contents = [
                        old_content,
                        new_content,
                    ]

                def fetch(self, config):
                    from source_adapters.base import FetchResult

                    return FetchResult(
                        success=True,
                        raw_content=self.contents.pop(0),
                        content_type="text/html",
                        status_code=200,
                    )

            adapter = SequenceAdapter()

            def classifier(old_text, new_text, api_key=None):
                return {
                    "change_type": "material",
                    "confidence": "high",
                    "changed_facts": [
                        {
                            "field": "pandas_usage",
                            "old_value": "tabular data",
                            "new_value": "time-series data",
                        }
                    ],
                    "explanation": "Pandas evidence changed.",
                }

            def reevaluator(
                old_source,
                new_source,
                question,
                previous_answer,
                api_key=None,
            ):
                return {
                    "answer_still_valid": False,
                    "affected_claim": (
                        "Pandas provides tools for working "
                        "with tabular data."
                    ),
                    "old_information": "tabular data",
                    "new_information": "time-series data",
                    "corrected_answer": (
                        "Pandas provides tools for working "
                        "with time-series data."
                    ),
                    "confidence": "high",
                    "explanation": "Pandas usage changed.",
                }

            # Initial snapshot
            result_1 = run_v2_monitor(
                source_cfg,
                [answer],
                snapshot_root,
                adapter,
                classifier,
                reevaluator,
                answer_directory=answers_dir,
                notifications_directory=notifications_dir,
            )

            self.assertEqual(
                result_1["status"],
                "initial_snapshot",
            )

            # Changed source
            result_2 = run_v2_monitor(
                source_cfg,
                [answer],
                snapshot_root,
                adapter,
                classifier,
                reevaluator,
                answer_directory=answers_dir,
                notifications_directory=notifications_dir,
            )

            self.assertEqual(
                result_2["status"],
                "changed",
            )

            self.assertEqual(
                result_2["candidate_claims"],
                1,
            )

            self.assertEqual(
                len(result_2["reevaluations"]),
                1,
            )
            

            self.assertEqual(
                result_2["reevaluations"][0]["result"]["affected_claim"],
                "Pandas provides tools for working with tabular data.",
            )

            self.assertEqual(
                len(result_2["notifications"]),
                1,
            )


if __name__ == "__main__":
    unittest.main()