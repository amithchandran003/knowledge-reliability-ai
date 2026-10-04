# rag_basic.py
"""
Retrieval-Augmented Generation (RAG) using Semantic Retrieval (Embeddings).

Workflow:
1. Retrieve:  Embed the query and find the closest policy document via cosine similarity.
2. Gate:      Check evidence sufficiency — if score < threshold, abstain (Gemini not called).
3. Augment:   Combine the retrieved policy and similarity score with the user question.
4. Generate:  Ask Gemini (gemini-3.6-flash) to answer strictly from the retrieved context.
5. Remember:  Store question, answer, and evidence metadata in Answer Memory.
"""

import os
import sys
import requests
from policy_documents import load_policies
from retrieval import SemanticRetriever
from answer_memory import build_source_identity, save_answer
from evidence_gate import evaluate_evidence, SIMILARITY_THRESHOLD

GEMINI_REQUEST_TIMEOUT_SECONDS = 30

# 1. Load environment variables from .env
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
                    os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def ask_gemini(policy_content: str, question: str, api_key: str) -> str:
    """Sends the context and question to Gemini with strict grounding instructions."""
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

    prompt = f"""You are an assistant answering questions based strictly on the provided source content.
Use ONLY the source text provided below to answer the user's question.
If the source text does not contain enough information to answer the question, state:
"The provided source does not contain enough information to answer this question."
Do not extrapolate, assume, or invent details not present in the source text.

Source Content:
\"\"\"
{policy_content}
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
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return "Gemini API request failed due to a network or service error."

    if response.status_code != 200:
        return f"Gemini API request failed (HTTP {response.status_code})."

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return "Gemini API returned an unexpected response format."


def query(question: str, retriever, api_key: str, save_to_memory: bool = True) -> dict:
    """
    Programmatic entry point for the RAG pipeline.
    Returns a result dict instead of printing; used by evaluation.py and other callers.

    Returns dict with keys:
        answered        (bool)   — True if Gemini was called, False if abstained
        answer          (str)    — LLM answer or abstention message
        retrieved_doc   (Document) — best-matching document
        similarity_score (float) — cosine similarity
        gate_passed     (bool)   — True when score >= threshold
        answer_record   (dict|None) — saved memory record, or None
        reason          (str|None) — "insufficient_evidence" on abstention, else None
    """
    retrieved_doc, similarity_score = retriever.retrieve(question)
    source_identifier = retrieved_doc.title
    # Document.source is currently a human-readable label (for example,
    # "Internal Compliance"), not a URL.  Convert it to a stable internal URI.
    source_identity = build_source_identity(
        source_identifier,
        getattr(retrieved_doc, "source_url", None),
    )
    source_url = source_identity["source_url"]

    gate_result = evaluate_evidence(similarity_score, source_identifier)

    if gate_result is not None:
        # Abstain — Gemini not called
        return {
            "answered": False,
            "answer": gate_result["answer"],
            "retrieved_doc": retrieved_doc,
            "similarity_score": similarity_score,
            "gate_passed": False,
            "answer_record": None,
            "reason": gate_result["reason"],
        }

    # Gate passed — call Gemini
    context = f"Title: {retrieved_doc.title}\nContent: {retrieved_doc.content}"
    answer = ask_gemini(context, question, api_key)

    answer_record = None
    if save_to_memory and not answer.startswith("API Error"):
        answer_record = save_answer(
            question=question,
            generated_answer=answer,
            source_identifier=source_identifier,
            source_content=retrieved_doc.content,
            source_url=source_url,
            similarity_score=similarity_score,
            source_id=source_identity["source_id"],
        )

    return {
        "answered": True,
        "answer": answer,
        "retrieved_doc": retrieved_doc,
        "similarity_score": similarity_score,
        "gate_passed": True,
        "answer_record": answer_record,
        "reason": None,
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Error: GEMINI_API_KEY is not set. Please configure your key in .env.")
        return

    # Check for dynamic RAG mode
    use_dynamic = "--dynamic" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--dynamic"]

    if use_dynamic:
        from dynamic_rag import DynamicRetriever, query_dynamic_rag
        from source_registry import load_sources

        enabled_sources = load_sources(enabled_only=True)
        if not enabled_sources:
            print("No enabled sources found in data/sources.json.")
            print("Register or enable sources using: python3 source_registry.py add <name> <url>")
            return

        print(f"Indexing {len(enabled_sources)} enabled dynamic source(s)...")
        retriever = DynamicRetriever(sources=enabled_sources, api_key=api_key)
        question = " ".join(args) if args else "What is described in the documentation?"
        result = query_dynamic_rag(question, retriever, api_key=api_key)

        print("=" * 60)
        print(f"User Question:\n{question}\n")
        if result["retrieved_chunk"]:
            chunk = result["retrieved_chunk"]
            print(f"Retrieved Source : [{chunk.source_identifier}] ({chunk.source_url})")
            print(f"Chunk ID         : {chunk.chunk_id}")
            print(f"Evidence Excerpt : {chunk.text}")
        print(f"Similarity Score : {result['similarity_score']:.4f}")
        print(f"Threshold        : {SIMILARITY_THRESHOLD}")

        if not result["gate_passed"]:
            print(f"\nGate Verdict     : INSUFFICIENT — Gemini not called\n")
            print(f"System Response:\n{result['answer']}")
            print(f"\n[reason: {result['reason']} | score: {result['similarity_score']:.4f}]")
        else:
            print(f"\nGate Verdict     : SUFFICIENT — proceeding to Gemini\n")
            print(f"LLM Answer:\n{result['answer']}")
            if result["answer_record"]:
                print(f"\nSaved to Answer Memory: ID={result['answer_record']['answer_id']}")
        print("=" * 60)
        return

    # Default: Load existing policy documents
    documents = load_policies()

    # Initialize semantic retriever (embeds documents into memory)
    retriever = SemanticRetriever(documents=documents, api_key=api_key)

    # Allow passing question via CLI args or interactive input
    if args:
        question = " ".join(args)
    else:
        try:
            user_input = input("Enter your policy question: ").strip()
            question = user_input if user_input else "How must personal data be encrypted?"
        except EOFError:
            question = "How must personal data be encrypted?"

    # Run pipeline and display
    result = query(question, retriever, api_key, save_to_memory=True)

    print("=" * 60)
    print(f"User Question:\n{question}\n")
    print(f"Retrieved Source : [{result['retrieved_doc'].title}]")
    print(f"Similarity Score : {result['similarity_score']:.4f}")
    print(f"Threshold        : {SIMILARITY_THRESHOLD}")

    if not result["gate_passed"]:
        print(f"Gate Verdict     : INSUFFICIENT — Gemini not called\n")
        print(f"System Response:\n{result['answer']}")
        print(f"\n[reason: {result['reason']} | score: {result['similarity_score']:.4f}]")
    else:
        print(f"Gate Verdict     : SUFFICIENT — proceeding to Gemini\n")
        print(f"LLM Answer:\n{result['answer']}")
        if result["answer_record"]:
            print(f"\nSaved to Answer Memory: ID={result['answer_record']['answer_id']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
