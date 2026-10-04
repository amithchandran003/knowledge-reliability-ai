"""Controlled tests for the V2 vertical slice; no live network or Gemini calls."""

import tempfile
from unittest.mock import Mock, patch
import json
import os

from answer_memory import save_answer
from source_adapters.base import FetchResult
from semantic_change import classify_semantic_change
from v2_monitor_pipeline import run_v2_monitor


class SequenceAdapter:
    """Returns controlled source versions in order."""

    def __init__(self, versions):
        self.versions = iter(versions)

    def fetch(self, source_config):
        value = next(self.versions)
        if isinstance(value, FetchResult):
            return value
        return FetchResult(success=True, raw_content=value, content_type="text/html", status_code=200)


def classifier(old_text, new_text, api_key=None):
    if "30 days" in old_text and "15 days" in new_text:
        return {"change_type": "material", "confidence": "high", "changed_facts": [{"field": "refund_period_days", "old_value": "30", "new_value": "15"}], "explanation": "Refund period changed."}
    if "30 days" in old_text and "30 days" in new_text:
        return {"change_type": "equivalent", "confidence": "high", "changed_facts": [], "explanation": "Refund period is unchanged."}
    return {"change_type": "uncertain", "confidence": "low", "changed_facts": [], "explanation": "Unexpected fixture."}


def reevaluator(**kwargs):
    invalid = "15 days" in kwargs["new_source"]
    return {
        "answer_still_valid": not invalid,
        "affected_claim": "Refund period is 30 days." if invalid else None,
        "old_information": "Refund period is 30 days." if invalid else None,
        "new_information": "Refund period is 15 days." if invalid else None,
        "corrected_answer": "Refund period is 15 days." if invalid else None,
        "confidence": "high",
        "explanation": "Controlled result.",
    }


def source_config():
    return {"source_id": "source_refund_policy_v2", "source_type": "web_page", "source_identifier": "Customer Refund Policy", "source_url": "https://example.test/refunds"}


def answer(directory):
    return save_answer(
        question="What is the refund period?",
        generated_answer="The refund period is 30 days.",
        source_identifier="Customer Refund Policy",
        source_url="https://example.test/refunds",
        source_content="Refunds are available within 30 days.\nContact support by email.",
        source_id="source_refund_policy_v2",
        directory=directory,
    )


def run_tests():
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"change_type":"material","confidence":"high","changed_facts":[],"explanation":"Changed."}'}]}}]}
    with patch("semantic_change.requests.post", return_value=response) as request:
        assert classify_semantic_change("Refund period is 30 days.", "Refund period is 15 days.", api_key="test-key")["change_type"] == "material"
        assert "?key=" not in request.call_args.args[0]
        assert request.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert request.call_args.kwargs["timeout"] == 30

    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as answers_dir:
        stored_answer = answer(answers_dir)
        old = "<p>Refunds are available within 30 days.</p><p>Contact support by email.</p>"
        material = "<p>Customers can request refunds within 15 days.</p><p>Contact support by email.</p>"
        pipeline = SequenceAdapter([old, material])
        assert run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator, answer_directory=answers_dir)["status"] == "initial_snapshot"
        result = run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator, answer_directory=answers_dir)
        assert result["reevaluations"][0]["semantic_change"]["change_type"] == "material"
        assert len(result["notifications"]) == 1
        record = json.load(open(os.path.join(answers_dir, f"{stored_answer['answer_id']}.json"), encoding="utf-8"))
        assert record["provenance"]["validity_history"][-1]["status"] == "stale"

    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as answers_dir:
        stored_answer = answer(answers_dir)
        old = "<p>Refunds are available within 30 days.</p><p>Contact support by email.</p>"
        equivalent = "<p>Customers can request refunds within 30 days.</p><p>Contact support by email.</p>"
        pipeline = SequenceAdapter([old, equivalent])
        run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator)
        result = run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator)
        assert result["candidate_claims"] == 1 and result["reevaluations"] == [] and result["notifications"] == []

    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as answers_dir:
        stored_answer = answer(answers_dir)
        old = "<p>Refunds are available within 30 days.</p><p>Contact support by email.</p>"
        support_change = "<p>Refunds are available within 30 days.</p><p>Contact support through the help center.</p>"
        pipeline = SequenceAdapter([old, support_change])
        run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator)
        result = run_v2_monitor(source_config(), [stored_answer], root, pipeline, classifier, reevaluator)
        assert result["candidate_claims"] == 0 and result["reevaluations"] == [] and result["notifications"] == []

    with tempfile.TemporaryDirectory() as root:
        unavailable = run_v2_monitor(source_config(), [], root, SequenceAdapter([FetchResult(success=False, error_code="timeout")]), classifier, reevaluator)
        assert unavailable["status"] == "unavailable" and unavailable["error_code"] == "timeout"

    print("V2 controlled tests: passed")


if __name__ == "__main__":
    run_tests()
