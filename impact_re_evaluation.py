"""Connects provenance impact detection to answer re-evaluation."""

from answer_revaluator import reevaluate_answer
from answer_memory import append_validity_history


def reevaluate_affected_claims(
    selections: list[dict],
    old_source: str,
    new_source: str,
    new_source_version_id: str | None = None,
) -> list[dict]:
    """
    Re-evaluate only answers whose claims were identified as affected.

    If the answer is no longer valid, its provenance validity history
    is updated to "stale".
    """

    results = []

    for selection in selections:
        answer = selection["answer"]
        claim = selection["claim"]

        result = reevaluate_answer(
            old_source=old_source,
            new_source=new_source,
            question=answer["question"],
            previous_answer=answer["generated_answer"],
        )

        answer_id = answer["answer_id"]

        # Use the new source version when supplied.
        # Otherwise retain the previous source version identifier.
        source_version_id = (
            new_source_version_id
            or answer.get("source_version_id")
            or answer.get("provenance", {})
            .get("validity_history", [{}])[-1]
            .get("source_version_id")
        )

        validity_record = None

        if result.get("answer_still_valid") is False:
            validity_record = append_validity_history(
                answer_id=answer_id,
                status="stale",
                reason="source_changed_and_answer_re_evaluated",
                source_version_id=source_version_id,
            )

        results.append(
            {
                "answer_id": answer_id,
                "claim_id": claim.get("claim_id"),
                "match_type": selection.get("match_type"),
                "evaluation": result,
                "validity_record": validity_record,
            }
        )

    return results


def print_re_evaluation_report(results: list[dict]) -> None:
    """Print a concise report for affected answers."""

    print("=" * 65)
    print("AFFECTED ANSWER RE-EVALUATION")
    print("=" * 65)

    if not results:
        print("No affected answers required re-evaluation.")
        return

    for result in results:
        evaluation = result["evaluation"]

        print(f"Answer ID:        {result['answer_id']}")
        print(f"Claim ID:         {result['claim_id']}")
        print(f"Match Type:       {result['match_type']}")
        print(
            f"Still Valid:      "
            f"{evaluation.get('answer_still_valid')}"
        )
        print(
            f"Confidence:       "
            f"{evaluation.get('confidence')}"
        )
        print(
            f"Old Information:  "
            f"{evaluation.get('old_information')}"
        )
        print(
            f"New Information:  "
            f"{evaluation.get('new_information')}"
        )
        print(
            f"Corrected Answer: "
            f"{evaluation.get('corrected_answer')}"
        )
        print(
            f"Validity Updated: "
            f"{result['validity_record'] is not None}"
        )
        print(
            f"Explanation:      "
            f"{evaluation.get('explanation')}"
        )
        print("-" * 65)