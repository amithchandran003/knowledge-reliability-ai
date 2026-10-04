import unittest
from unittest.mock import patch

from impact_re_evaluation import reevaluate_affected_claims


class ImpactReEvaluationTests(unittest.TestCase):

    @patch("impact_re_evaluation.reevaluate_answer")
    def test_affected_answer_is_re_evaluated(self, mock_reevaluate):

        # Simulate Gemini's re-evaluation result.
        mock_reevaluate.return_value = {
            "answer_still_valid": False,
            "affected_claim": "The refund period is 30 days.",
            "old_information": "30 days",
            "new_information": "15 days",
            "corrected_answer": "The refund period is 15 days.",
            "confidence": "high",
            "explanation": "The refund period changed from 30 days to 15 days."
        }

        # This represents an answer that our provenance system
        # has already identified as affected.
        selections = [
            {
                "answer": {
                    "answer_id": "ans_refund_001",
                    "question": "What is the refund period?",
                    "generated_answer": "The refund period is 30 days."
                },
                "claim": {
                    "claim_id": "claim_001",
                    "text": "The refund period is 30 days."
                },
                "match_type": "exact_evidence",
                "evidence": [
                    {
                        "evidence_id": "evidence_001",
                        "chunk_id": "refund_chunk_001"
                    }
                ]
            }
        ]

        old_source = "Refund period is 30 days."
        new_source = "Refund period is 15 days."

        results = reevaluate_affected_claims(
            selections=selections,
            old_source=old_source,
            new_source=new_source,
        )

        # One affected answer should have been re-evaluated.
        self.assertEqual(len(results), 1)

        result = results[0]

        # Verify the correct answer and claim were processed.
        self.assertEqual(
            result["answer_id"],
            "ans_refund_001"
        )

        self.assertEqual(
            result["claim_id"],
            "claim_001"
        )

        self.assertEqual(
            result["match_type"],
            "exact_evidence"
        )

        # Verify the re-evaluation result.
        evaluation = result["evaluation"]

        self.assertFalse(
            evaluation["answer_still_valid"]
        )

        self.assertEqual(
            evaluation["old_information"],
            "30 days"
        )

        self.assertEqual(
            evaluation["new_information"],
            "15 days"
        )

        self.assertEqual(
            evaluation["corrected_answer"],
            "The refund period is 15 days."
        )

        # Verify that the re-evaluator was actually called.
        mock_reevaluate.assert_called_once_with(
            old_source=old_source,
            new_source=new_source,
            question="What is the refund period?",
            previous_answer="The refund period is 30 days.",
        )


if __name__ == "__main__":
    unittest.main()