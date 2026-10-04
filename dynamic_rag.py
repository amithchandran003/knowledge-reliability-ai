# dynamic_rag.py
"""
Dynamic RAG Integration Module.
Enables Retrieval-Augmented Generation over external web sources registered in
data/sources.json using V2 adapters, deterministic normalization/chunking,
Gemini embeddings, evidence gating, and provenance-tracked answer memory.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable

from answer_memory import ANSWERS_DIR, save_answer
from content_normalizer import normalize_fetch_result
from evidence_gate import SIMILARITY_THRESHOLD, evaluate_evidence
from rag_basic import ask_gemini
from retrieval import cosine_similarity, get_embedding
from snapshot_store import (
    SNAPSHOT_ROOT,
    build_snapshot,
    get_latest_snapshot,
    record_source_status,
    save_snapshot,
)
from source_adapters import get_adapter
from source_registry import SOURCES_FILE, get_source, load_sources

# Load environment variables from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(
                        key.strip(),
                        val.strip().strip("'").strip('"'),
                    )


@dataclass
class SourceChunk:
    """Represents a fine-grained chunk of content with source provenance."""

    chunk_id: str
    text: str
    source_id: str
    source_identifier: str
    source_url: str
    source_version_id: str
    content_hash: str
    location: dict

    @property
    def title(self) -> str:
        """Duck-typing compatibility with Document objects."""
        return self.source_identifier

    @property
    def content(self) -> str:
        """Duck-typing compatibility with Document objects."""
        return self.text


@dataclass
class RetrievalMatch:
    """Represents a retrieved chunk with its hybrid retrieval score and pure semantic similarity."""

    chunk: SourceChunk
    retrieval_score: float
    semantic_score: float

    def __iter__(self):
        yield self.chunk
        yield self.retrieval_score

    def __getitem__(self, index):
        if index == 0:
            return self.chunk
        elif index == 1:
            return self.retrieval_score
        elif index == 2:
            return self.semantic_score
        raise IndexError("RetrievalMatch index out of range")


class DynamicRetriever:
    """
    In-memory chunk-level semantic retriever for registered external sources.
    Fetches, normalizes, chunks, snapshots, and embeds content from active sources.
    """

    def __init__(
        self,
        sources: list[dict] | None = None,
        sources_path: str = SOURCES_FILE,
        api_key: str | None = None,
        adapter_factory: Callable = get_adapter,
        embed_fn: Callable = get_embedding,
        snapshot_root: str = SNAPSHOT_ROOT,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.adapter_factory = adapter_factory
        self.embed_fn = embed_fn
        self.snapshot_root = snapshot_root
        self.chunks: list[SourceChunk] = []
        self.chunk_embeddings: list[list[float]] = []

        if sources is None:
            sources = load_sources(
                sources_path,
                enabled_only=True,
            )

        self._index_sources(sources)

    def _index_sources(self, sources: list[dict]) -> None:
        """Fetches and creates chunk embeddings for all configured external sources."""

        for source_config in sources:
            source_id = source_config.get(
                "source_id",
                "unknown_source",
            )

            adapter = self.adapter_factory(
                source_config.get(
                    "source_type",
                    "web_page",
                )
            )

            snapshot = None

            try:
                fetch = adapter.fetch(source_config)

                if fetch.success and fetch.raw_content:
                    normalized = normalize_fetch_result(
                        source_config,
                        fetch.raw_content,
                    )

                    snapshot = build_snapshot(normalized)

                    save_snapshot(
                        snapshot,
                        root=self.snapshot_root,
                    )

                    record_source_status(
                        source_id,
                        {
                            "status": "available",
                            "source_version_id": snapshot[
                                "source_version_id"
                            ],
                        },
                        root=self.snapshot_root,
                    )

                else:
                    record_source_status(
                        source_id,
                        {
                            "status": "unavailable",
                            "error_code": getattr(
                                fetch,
                                "error_code",
                                "fetch_failed",
                            ),
                        },
                        root=self.snapshot_root,
                    )

            except Exception:
                # On live fetch failure, attempt to fall back to cached snapshot.
                pass

            if snapshot is None:
                snapshot = get_latest_snapshot(
                    source_id,
                    root=self.snapshot_root,
                )

            if not snapshot or not snapshot.get("chunks"):
                continue

            for chunk_data in snapshot["chunks"]:
                chunk = SourceChunk(
                    chunk_id=chunk_data["chunk_id"],
                    text=chunk_data["text"],
                    source_id=snapshot["source_id"],
                    source_identifier=snapshot["source_identifier"],
                    source_url=snapshot["source_url"],
                    source_version_id=snapshot["source_version_id"],
                    content_hash=snapshot["content_hash"],
                    location=chunk_data.get(
                        "location",
                        {},
                    ),
                )

                self.chunks.append(chunk)

        # Generate embeddings for all indexed chunks.
        for chunk in self.chunks:
            embedding = self.embed_fn(
                chunk.text,
                self.api_key,
            )

            self.chunk_embeddings.append(embedding)

    def retrieve_top_k(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> list[RetrievalMatch]:
        """
        Hybrid retrieval combining:
        1. Semantic cosine similarity (relevance measure)
        2. Exact keyword overlap (ranking bonus)
        3. Subject keyword alignment (ranking bonus)

        Returns list of RetrievalMatch objects with both retrieval_score and semantic_score.
        """

        if not self.chunks:
            return []

        query_embedding = self.embed_fn(
            query,
            self.api_key,
        )

        stop_words = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "to",
            "for",
            "and",
            "how",
            "why",
            "when",
            "where",
            "does",
            "do",
            "can",
            "tell",
            "me",
            "about",
        }

        query_terms = {
            word.lower().strip(".,?!:;()[]{}\"'")
            for word in query.split()
            if len(
                word.strip(".,?!:;()[]{}\"'")
            ) > 2
            and word.lower().strip(
                ".,?!:;()[]{}\"'"
            ) not in stop_words
        }

        scored: list[RetrievalMatch] = []

        for chunk, chunk_emb in zip(
            self.chunks,
            self.chunk_embeddings,
        ):
            semantic_score = cosine_similarity(
                query_embedding,
                chunk_emb,
            )

            chunk_text = chunk.text.lower()

            matched_terms = sum(
                1
                for term in query_terms
                if term in chunk_text
            )

            lexical_bonus = min(
                matched_terms * 0.08,
                0.20,
            )

            subject_bonus = 0.0

            if (
                "python" in query_terms
                and "python" in chunk_text
            ):
                subject_bonus = 0.10

            # Semantic similarity measures fundamental topical relevance.
            # Lexical and subject bonuses provide a small ranking aid.
            final_score = (
                semantic_score * 0.85
                + lexical_bonus * 0.10
                + subject_bonus * 0.05
            )

            if final_score >= min_score:
                scored.append(
                    RetrievalMatch(
                        chunk=chunk,
                        retrieval_score=final_score,
                        semantic_score=semantic_score,
                    )
                )

        scored.sort(
            key=lambda x: x.retrieval_score,
            reverse=True,
        )

        return scored[:top_k]

    def retrieve(
        self,
        query: str,
    ) -> tuple[SourceChunk | None, float]:
        """
        Embeds the query and computes hybrid retrieval score
        against indexed chunks.

        Returns:
            (best_chunk, retrieval_score)
        """

        top_matches = self.retrieve_top_k(
            query,
            top_k=1,
            min_score=-1.0,
        )

        if not top_matches:
            return None, 0.0

        match = top_matches[0]
        return match.chunk, match.retrieval_score


def generate_grounded_answer(
    context: str,
    question: str,
    api_key: str,
) -> str:
    """Sends the context and question to Gemini with domain-neutral strict grounding instructions."""
    import time
    import requests

    api_url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.6-flash:generateContent"
    )

    prompt = f"""You are an assistant answering questions based strictly on the provided source content.
Use ONLY the source text provided below to answer the user's question.
If the source text does not contain enough information to answer the question, state:
"The provided source does not contain enough information to answer this question."
Do not extrapolate, assume, or invent details not present in the source text.

Source Content:
\"\"\"
{context}
\"\"\"

User Question:
{question}
"""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ]
    }

    response = None

    for attempt in range(3):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                break

            if response.status_code in (
                429,
                500,
                502,
                503,
                504,
            ) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue

        except requests.RequestException as err:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue

            return f"Gemini API network error: {type(err).__name__}: {err}"
    if response is not None and response.status_code != 200:
        print(
            f"Gemini generation HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    if response is None or response.status_code != 200:
        status_code = response.status_code if response is not None else "timeout"
        return f"Gemini API request failed (HTTP {status_code})."

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return "Gemini API returned an unexpected response format."


def query_dynamic_rag(
    question: str,
    retriever: DynamicRetriever,
    api_key: str | None = None,
    save_to_memory: bool = True,
    answer_dir: str = ANSWERS_DIR,
    llm_fn: Callable = generate_grounded_answer,
) -> dict:
    """
    Executes the dynamic RAG pipeline for a given user question:
    1. Multi-chunk retrieval and ranking via hybrid scoring (semantic cosine similarity + lexical/subject bonuses).
    2. Candidate evidence collection across multiple relevant chunks.
    3. Evidence sufficiency gate check via SIMILARITY_THRESHOLD (abstaining without LLM if evidence is insufficient).
    4. Domain-neutral strictly grounded LLM generation with collective evidence excerpts.
    5. Provenance tracking with source_id, source_version_id, and multi-chunk evidence items.
    """
    api_key = (
        api_key
        or retriever.api_key
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )

    # 1. Retrieve top-k candidate chunks
    top_matches = retriever.retrieve_top_k(
        question,
        top_k=5,
        min_score=-1.0,
    )

    if not top_matches:
        source_id_label = "No available sources"
        gate_result = evaluate_evidence(
            0.0,
            source_id_label,
        )

        return {
            "answered": False,
            "answer": (
                gate_result["answer"]
                if gate_result
                else "No sources available to answer."
            ),
            "retrieved_chunk": None,
            "retrieved_chunks": [],
            "similarity_score": 0.0,
            "retrieval_score": 0.0,
            "semantic_similarity": 0.0,
            "gate_passed": False,
            "answer_record": None,
            "reason": "no_indexed_sources",
            "answerable": False,
        }

    best_match = top_matches[0]
    best_chunk = best_match.chunk
    retrieval_score = best_match.retrieval_score
    semantic_similarity = best_match.semantic_score

    # 2. Gate check on best retrieval score
    gate_result = evaluate_evidence(
        retrieval_score,
        best_chunk.source_identifier,
    )

    if gate_result is not None:
        # Insufficient retrieval evidence — abstain without invoking Gemini
        return {
            "answered": False,
            "answer": gate_result["answer"],
            "retrieved_chunk": best_chunk,
            "retrieved_chunks": [best_chunk],
            "similarity_score": retrieval_score,
            "retrieval_score": retrieval_score,
            "semantic_similarity": semantic_similarity,
            "gate_passed": False,
            "answer_record": None,
            "reason": gate_result["reason"],
            "answerable": False,
        }

    # 3. Gate passed — collect all relevant candidate chunks from the primary source
    selected_chunks = [best_chunk]

    for match in top_matches:
        c = match.chunk
        s = match.retrieval_score
        if (
            c.chunk_id != best_chunk.chunk_id
            and c.source_id == best_chunk.source_id
            and s >= 0.50
        ):
            selected_chunks.append(c)

    # 4. Check collective evidence sufficiency (ensure non-empty, substantive content)
    collective_text = " ".join(c.text.strip() for c in selected_chunks)
    if len(collective_text.strip()) < 15:
        return {
            "answered": False,
            "answer": (
                "I don't have enough reliable evidence in the available sources "
                "to answer this question."
            ),
            "retrieved_chunk": best_chunk,
            "retrieved_chunks": selected_chunks,
            "similarity_score": retrieval_score,
            "retrieval_score": retrieval_score,
            "semantic_similarity": semantic_similarity,
            "gate_passed": False,
            "answer_record": None,
            "reason": "insufficient_evidence",
            "answerable": False,
        }

    # 5. Build bounded context from collective evidence
    evidence_lines = [
        f"- [{c.chunk_id}]: {c.text}"
        for c in selected_chunks
    ]

    context = (
        f"Source: {best_chunk.source_identifier} "
        f"({best_chunk.source_url})\n"
        f"Evidence Excerpts:\n"
        + "\n".join(evidence_lines)
    )

    # 6. Call LLM for grounded answer generation
    answer = llm_fn(
        context,
        question,
        api_key,
    )

    # Check whether Gemini itself determined that
    # the retrieved evidence was insufficient.
    insufficient_message = (
        "The provided source does not contain enough "
        "information to answer this question."
    )

    answer_is_insufficient = (
        answer.strip().lower()
        == insufficient_message.lower()
        or answer.strip().lower().startswith(
            "the provided source does not contain enough information"
        )
    )

    if answer_is_insufficient:
        return {
            "answered": False,
            "answer": answer,
            "retrieved_chunk": best_chunk,
            "retrieved_chunks": selected_chunks,
            "similarity_score": retrieval_score,
            "retrieval_score": retrieval_score,
            "semantic_similarity": semantic_similarity,
            "gate_passed": True,
            "answer_record": None,
            "reason": "llm_found_insufficient_evidence",
            "answerable": False,
        }

    # 7. Grounded answer generated — save provenance
    answer_record = None

    if save_to_memory and not answer.startswith(
        "Gemini API request failed"
    ):
        evidence_content = "\n".join(
            c.text for c in selected_chunks
        )

        evidence_items = [
            {
                "evidence_id": f"evidence_{index:03d}",
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source_version_id": chunk.source_version_id,
                "location": chunk.location,
            }
            for index, chunk in enumerate(
                selected_chunks,
                1,
            )
        ]

        answer_record = save_answer(
            question=question,
            generated_answer=answer,
            source_identifier=best_chunk.source_identifier,
            source_content=evidence_content,
            source_hash=best_chunk.content_hash,
            source_url=best_chunk.source_url,
            similarity_score=retrieval_score,
            source_id=best_chunk.source_id,
            directory=answer_dir,
            evidence_items=evidence_items,
        )

    return {
        "answered": True,
        "answer": answer,
        "retrieved_chunk": best_chunk,
        "retrieved_chunks": selected_chunks,
        "similarity_score": retrieval_score,
        "retrieval_score": retrieval_score,
        "semantic_similarity": semantic_similarity,
        "gate_passed": True,
        "answer_record": answer_record,
        "reason": None,
        "answerable": True,
    }


def main():
    """CLI runner for Dynamic RAG over registered external sources."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        print("Error: GEMINI_API_KEY is not set. Please configure .env.")
        return

    # Check registered sources
    enabled_sources = load_sources(enabled_only=True)

    if not enabled_sources:
        print("No enabled sources found in data/sources.json.")
        print(
            "Register or enable sources using: "
            "python3 source_registry.py add <name> <url>"
        )
        return

    print(
        f"Indexing {len(enabled_sources)} enabled dynamic source(s)..."
    )

    retriever = DynamicRetriever(
        sources=enabled_sources,
        api_key=api_key,
    )

    print(
        f"Indexed {len(retriever.chunks)} total chunks across sources.\n"
    )

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        try:
            user_input = input("Enter your question: ").strip()
            question = (
                user_input
                if user_input
                else "What is described in the documentation?"
            )
        except EOFError:
            question = "What is described in the documentation?"

    result = query_dynamic_rag(
        question,
        retriever,
        api_key=api_key,
    )

    print("=" * 65)
    print(f"User Question:\n{question}\n")

    if result["retrieved_chunk"]:
        chunk = result["retrieved_chunk"]

        print(
            f"Retrieved Source : "
            f"[{chunk.source_identifier}] ({chunk.source_url})"
        )

        print(f"Chunk ID         : {chunk.chunk_id}")

        if (
            result.get("retrieved_chunks")
            and len(result["retrieved_chunks"]) > 1
        ):
            print(
                f"Evidence Units   : "
                f"{len(result['retrieved_chunks'])} chunks"
            )

            for c in result["retrieved_chunks"]:
                print(f"  • [{c.chunk_id}]: {c.text}")
        else:
            print(f"Evidence Excerpt : {chunk.text}")

        print(
            f"Retrieval Score  : "
            f"{result.get('retrieval_score', result.get('similarity_score', 0)):.4f}"
        )
        if "semantic_similarity" in result:
            print(
                f"Semantic Sim.    : "
                f"{result['semantic_similarity']:.4f}"
            )

    print(f"Threshold        : {SIMILARITY_THRESHOLD}")

    if not result["gate_passed"]:
        print(
            "\nGate Verdict     : "
            "INSUFFICIENT RETRIEVAL EVIDENCE — Gemini generation suppressed"
        )
        print(
            f"System Response  :\n{result['answer']}"
        )
    elif not result.get("answerable", True):
        print(
            "\nGate Verdict     : "
            "EVIDENCE RELEVANT, BUT ANSWER NOT SUPPORTED"
        )
        print(
            f"LLM Response     :\n{result['answer']}"
        )
    else:
        print(
            "\nGate Verdict     : "
            "GROUNDED ANSWER GENERATED"
        )
        print(
            f"LLM Answer       :\n{result['answer']}"
        )

        if result["answer_record"]:
            rec = result["answer_record"]

            print(
                f"\nSaved Provenance Record: "
                f"ID={rec['answer_id']}"
            )

            print(
                f"  Source Version ID:    "
                f"{rec.get('source_version_id')}"
            )

            print(
                f"  Content Hash:         "
                f"{rec.get('source_hash')[:16]}..."
            )

    print("=" * 65)


if __name__ == "__main__":
    main()