import hashlib
import unittest

from answer_memory import save_answer
from claim_impact import select_affected_claims


class ClaimImpactTests(unittest.TestCase):

    def test_exact_evidence_match_for_changed_chunk(self):
        # Original source
        old_content = (
            "Customer satisfaction is our priority.\n"
            "Refund period is 30 days.\n"
            "Contact support for billing inquiries."
        )

        old_hash = hashlib.sha256(
            old_content.encode("utf-8")
        ).hexdigest()

        # Store an answer with explicit provenance
        answer = save_answer(
            question="What is the refund window?",
            generated_answer="The refund period is 30 days.",
            source_identifier="Customer Refund Policy",
            source_content=old_content,
            source_hash=old_hash,
            source_url="https://example.internal/refund-policy",
            source_id="refund_policy",
            evidence_items=[
                {
                    "evidence_id": "evidence_001",
                    "text": "Refund period is 30 days.",
                    "chunk_id": "refund_chunk_001",
                    "source_version_id": f"refund_policy@{old_hash}",
                    "location": {
                        "kind": "text_unit",
                        "start_char": 42,
                        "end_char": 67,
                    },
                }
            ],
        )

        # Simulate the source changing from 30 → 15 days
        changes = [
            {
                "change_type": "material",
                "old_chunk": {
                    "chunk_id": "refund_chunk_001",
                    "text": "Refund period is 30 days.",
                },
                "new_chunk": {
                    "chunk_id": "refund_chunk_001",
                    "text": "Refund period is 15 days.",
                },
            }
        ]

        previous_snapshot = {
            "source_id": "refund_policy",
            "content_hash": old_hash,
        }

        results = select_affected_claims(
            previous_snapshot=previous_snapshot,
            changes=changes,
            answers=[answer],
        )

        self.assertEqual(len(results), 1)

        result = results[0]

        self.assertEqual(
            result["match_type"],
            "exact_evidence",
        )

        self.assertEqual(
            result["claim"]["claim_id"],
            "claim_001",
        )

        self.assertEqual(
            result["evidence"][0]["evidence_id"],
            "evidence_001",
        )

        self.assertEqual(
            result["evidence"][0]["chunk_id"],
            "refund_chunk_001",
        )

    def test_multi_claim_maps_changed_evidence_to_correct_claim(self):
        old_content = (
            "Python is widely used for data analysis.\n"
            "Pandas provides tools for working with tabular data.\n"
            "NumPy supports numerical computing."
        )

        old_hash = hashlib.sha256(
            old_content.encode("utf-8")
        ).hexdigest()

        # Create an answer containing three separate claims.
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
            source_url="https://example.internal/python-tools",
            source_id="python_tools",
            evidence_items=[
                {
                    "evidence_id": "evidence_001",
                    "text": "Python is widely used for data analysis.",
                    "chunk_id": "python_chunk_001",
                    "source_version_id": f"python_tools@{old_hash}",
                },
                {
                    "evidence_id": "evidence_002",
                    "text": "Pandas provides tools for working with tabular data.",
                    "chunk_id": "pandas_chunk_001",
                    "source_version_id": f"python_tools@{old_hash}",
                },
                {
                    "evidence_id": "evidence_003",
                    "text": "NumPy supports numerical computing.",
                    "chunk_id": "numpy_chunk_001",
                    "source_version_id": f"python_tools@{old_hash}",
                },
            ],
        )

        # Only the Pandas evidence changes.
        changes = [
            {
                "change_type": "material",
                "old_chunk": {
                    "chunk_id": "pandas_chunk_001",
                    "text": "Pandas provides tools for working with tabular data.",
                },
                "new_chunk": {
                    "chunk_id": "pandas_chunk_001",
                    "text": "Pandas provides tools for working with time-series data.",
                },
            }
        ]

        previous_snapshot = {
            "source_id": "python_tools",
            "content_hash": old_hash,
        }

        results = select_affected_claims(
            previous_snapshot=previous_snapshot,
            changes=changes,
            answers=[answer],
        )

        self.assertEqual(len(results), 1)

        result = results[0]

        self.assertEqual(
            result["match_type"],
            "exact_evidence",
        )

        # Only claim_002 should be affected.
        self.assertEqual(
            result["claim"]["claim_id"],
            "claim_002",
        )

        self.assertEqual(
            result["evidence"][0]["evidence_id"],
            "evidence_002",
        )


if __name__ == "__main__":
    unittest.main()