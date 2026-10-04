# test_dynamic_rag.py
"""
Focused, fully-mocked unit tests for Dynamic RAG Integration.
Tests run completely offline without calling live external websites or Gemini APIs.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock

from dynamic_rag import DynamicRetriever, SourceChunk, query_dynamic_rag
from snapshot_store import build_snapshot, save_snapshot
from source_adapters.base import FetchResult


def mock_embed_fn(text: str, api_key: str | None = None) -> list[float]:
    """
    Deterministic mock embedding function:
    Creates a simple fixed-length vector highlighting keywords for deterministic similarity.
    """
    text_lower = text.lower()
    vec = [0.0] * 8
    if "refund" in text_lower or "30 days" in text_lower or "15 days" in text_lower:
        vec[0] = 1.0
    if "support" in text_lower or "email" in text_lower:
        vec[1] = 1.0
    if "privacy" in text_lower or "encrypt" in text_lower or "security" in text_lower:
        vec[2] = 1.0
    if "unrelated" in text_lower or "football" in text_lower or "weather" in text_lower:
        vec[3] = 1.0
    # Add a base non-zero element so vectors are non-zero
    vec[7] = 0.5
    return vec


class MockAdapter:
    def __init__(self, raw_content: str, success: bool = True, error_code: str | None = None):
        self.raw_content = raw_content
        self.success = success
        self.error_code = error_code

    def fetch(self, source_config: dict) -> FetchResult:
        if not self.success:
            return FetchResult(success=False, error_code=self.error_code or "network_error")
        return FetchResult(
            success=True,
            raw_content=self.raw_content,
            content_type="text/html",
            status_code=200,
        )


class DynamicRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshot_root = os.path.join(self.temp_dir.name, "snapshots_v2")
        self.answers_dir = os.path.join(self.temp_dir.name, "answers")
        self.sources_file = os.path.join(self.temp_dir.name, "sources.json")

        self.sample_html = (
            "<html><body>"
            "<p>Refund requests are accepted within 30 days of purchase.</p>"
            "<p>Contact support by email for billing questions.</p>"
            "</body></html>"
        )
        self.sample_sources = [
            {
                "source_id": "source_refund_doc",
                "enabled": True,
                "source_type": "web_page",
                "source_identifier": "Customer Refund Policy",
                "source_url": "https://example.com/refunds",
            }
        ]
        with open(self.sources_file, "w", encoding="utf-8") as f:
            json.dump({"sources": self.sample_sources}, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dynamic_retriever_indexes_chunks_and_creates_snapshots(self):
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        self.assertGreater(len(retriever.chunks), 0)
        first_chunk = retriever.chunks[0]
        self.assertIsInstance(first_chunk, SourceChunk)
        self.assertEqual(first_chunk.source_id, "source_refund_doc")
        self.assertEqual(first_chunk.source_identifier, "Customer Refund Policy")
        self.assertEqual(first_chunk.source_url, "https://example.com/refunds")
        self.assertIn("source_refund_doc@", first_chunk.source_version_id)
        self.assertTrue(len(first_chunk.content_hash) == 64)

        # Verify snapshot was written to disk
        source_snap_dir = os.path.join(self.snapshot_root, "source_refund_doc")
        self.assertTrue(os.path.exists(source_snap_dir))
        snapshots = os.listdir(source_snap_dir)
        self.assertEqual(len(snapshots), 1)

    def test_chunk_retrieval_and_evidence_gate_pass(self):
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock(return_value="According to the policy, refunds are accepted within 30 days.")
        question = "What is the refund period?"

        result = query_dynamic_rag(
            question=question,
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertTrue(result["answered"])
        self.assertTrue(result["gate_passed"])
        self.assertIn("30 days", result["answer"])
        self.assertGreaterEqual(result["similarity_score"], 0.65)
        self.assertIsNotNone(result["retrieved_chunk"])
        self.assertIn("Refund requests", result["retrieved_chunk"].text)

        # Verify LLM was called with bounded context
        mock_llm.assert_called_once()
        context_arg = mock_llm.call_args.args[0]
        self.assertIn("Customer Refund Policy", context_arg)
        self.assertIn("30 days", context_arg)

        # Verify saved answer record and provenance
        self.assertIsNotNone(result["answer_record"])
        record = result["answer_record"]
        self.assertEqual(record["source_id"], "source_refund_doc")
        self.assertEqual(record["source_url"], "https://example.com/refunds")
        self.assertEqual(record["source_version_id"], result["retrieved_chunk"].source_version_id)
        self.assertEqual(record["source_hash"], result["retrieved_chunk"].content_hash)

        # Verify record exists in answer directory
        ans_file = os.path.join(self.answers_dir, f"{record['answer_id']}.json")
        self.assertTrue(os.path.exists(ans_file))
        with open(ans_file, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["answer_id"], record["answer_id"])
        self.assertEqual(saved_json["provenance"]["sources"][0]["source_id"], "source_refund_doc")

    def test_evidence_gate_abstains_on_irrelevant_query(self):
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock()
        question = "What is the football championship score?"

        result = query_dynamic_rag(
            question=question,
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertFalse(result["answered"])
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["reason"], "insufficient_evidence")
        self.assertIn("don't have enough reliable evidence", result["answer"])
        self.assertIsNone(result["answer_record"])
        # Verify LLM was NOT invoked
        mock_llm.assert_not_called()

    def test_fallback_to_cached_snapshot_when_live_fetch_fails(self):
        # Pre-seed snapshot store with a snapshot
        from content_normalizer import normalize_fetch_result

        norm = normalize_fetch_result(self.sample_sources[0], self.sample_html)
        snap = build_snapshot(norm)
        save_snapshot(snap, root=self.snapshot_root)

        # Now adapter fails
        failing_adapter = MockAdapter("", success=False, error_code="timeout")
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: failing_adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        # Retriever should successfully load chunks from cached snapshot
        self.assertGreater(len(retriever.chunks), 0)
        self.assertEqual(retriever.chunks[0].source_id, "source_refund_doc")

    def test_empty_sources_graceful_handling(self):
        retriever = DynamicRetriever(
            sources=[],
            api_key="mock-key",
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )
        self.assertEqual(len(retriever.chunks), 0)

        mock_llm = Mock()
        result = query_dynamic_rag(
            question="Any question",
            retriever=retriever,
            api_key="mock-key",
            llm_fn=mock_llm,
        )
        self.assertFalse(result["answered"])
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["reason"], "no_indexed_sources")
        mock_llm.assert_not_called()

    def test_dynamic_retriever_reads_api_key_from_environ(self):
        # Verify DynamicRetriever picks up GEMINI_API_KEY from environment when api_key is omitted
        with unittest.mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-configured-key"}):
            retriever = DynamicRetriever(
                sources=[],
                embed_fn=mock_embed_fn,
                snapshot_root=self.snapshot_root,
            )
            self.assertEqual(retriever.api_key, "env-configured-key")

    def test_domain_neutral_generation_prompt(self):
        from dynamic_rag import generate_grounded_answer
        from unittest.mock import patch

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Python is a programming language."}]}}]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            ans = generate_grounded_answer(
                context="Python is an interpreted programming language.",
                question="What is Python?",
                api_key="test-key",
            )
            self.assertEqual(ans, "Python is a programming language.")
            payload = mock_post.call_args.kwargs["json"]
            prompt_text = payload["contents"][0]["parts"][0]["text"]
            self.assertIn("provided source content", prompt_text)
            self.assertNotIn("company policies", prompt_text)

    def test_multi_chunk_evidence_inclusion(self):
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock(
            return_value="Refunds are accepted within 30 days and support can be reached by email."
        )
        result = query_dynamic_rag(
            question="What is the refund and support policy?",
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["answered"])
        self.assertTrue(result["answerable"])
        self.assertGreaterEqual(len(result["retrieved_chunks"]), 2)
        
        # Verify context passed to LLM contains both evidence items
        context_arg = mock_llm.call_args.args[0]
        self.assertIn("30 days", context_arg)
        self.assertIn("email", context_arg)

        # Verify provenance record contains both chunk evidence items
        self.assertIsNotNone(result["answer_record"])
        record = result["answer_record"]
        evidence_list = record["provenance"]["sources"][0]["versions"][0]["evidence"]
        self.assertGreaterEqual(len(evidence_list), 2)

    def test_high_similarity_but_answer_absent_in_evidence(self):
        """Scenario 1: High similarity / keyword match passes gate, but LLM determines evidence is insufficient."""
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock(
            return_value="The provided source does not contain enough information to answer this question."
        )
        result = query_dynamic_rag(
            question="What is the refund processing fee?",
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        # Gate passed during retrieval, but Gemini found evidence insufficient
        self.assertTrue(result["gate_passed"])
        self.assertFalse(result["answered"])
        self.assertFalse(result["answerable"])
        self.assertEqual(result["reason"], "llm_found_insufficient_evidence")
        self.assertIsNone(result["answer_record"])

    def test_multiple_chunks_collectively_support_answer(self):
        """Scenario 2: Multiple chunks from the source collectively provide sufficient context."""
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock(
            return_value="Refunds are accepted within 30 days and support is available by email."
        )
        result = query_dynamic_rag(
            question="Tell me about refunds and support contact email",
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["answered"])
        self.assertTrue(result["answerable"])
        self.assertIsNone(result["reason"])
        self.assertGreaterEqual(len(result["retrieved_chunks"]), 2)
        self.assertIsNotNone(result["answer_record"])

    def test_score_below_threshold_abstains_without_llm(self):
        """Scenario 3: Score below threshold triggers immediate abstention without calling LLM."""
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock()
        result = query_dynamic_rag(
            question="Who won the international football championship?",
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["answered"])
        self.assertFalse(result["answerable"])
        self.assertEqual(result["reason"], "insufficient_evidence")
        self.assertIsNone(result["answer_record"])
        mock_llm.assert_not_called()

    def test_supported_grounded_answer_saves_provenance(self):
        """Scenario 4: Sufficient evidence yields a grounded answer with provenance saved."""
        adapter = MockAdapter(self.sample_html)
        retriever = DynamicRetriever(
            sources=self.sample_sources,
            api_key="mock-key",
            adapter_factory=lambda st: adapter,
            embed_fn=mock_embed_fn,
            snapshot_root=self.snapshot_root,
        )

        mock_llm = Mock(return_value="Refund requests are accepted within 30 days of purchase.")
        result = query_dynamic_rag(
            question="What is the refund period?",
            retriever=retriever,
            api_key="mock-key",
            save_to_memory=True,
            answer_dir=self.answers_dir,
            llm_fn=mock_llm,
        )

        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["answered"])
        self.assertTrue(result["answerable"])
        self.assertIsNone(result["reason"])
        self.assertIsNotNone(result["answer_record"])
        self.assertIn("30 days", result["answer"])
        self.assertEqual(result["answer_record"]["source_id"], "source_refund_doc")


if __name__ == "__main__":
    unittest.main()

