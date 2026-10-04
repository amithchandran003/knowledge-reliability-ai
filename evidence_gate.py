# evidence_gate.py
"""
Evidence Sufficiency Gate — V1.

Before calling the LLM, check whether the retrieved document is
semantically close enough to the user question to justify generating an
answer.  If it is not, return a structured abstention result so that
rag_basic.py never invokes Gemini for unsupported questions.

Threshold rationale (V1):
  Observed similarity scores in this project span ~0.63 – 0.79 for
  genuinely on-topic questions.  A threshold of 0.65 allows all
  policy-related questions to pass while cleanly rejecting queries that
  are clearly outside the knowledge base (e.g. flight-upgrade policy,
  sports results, etc. which typically fall below 0.55).

  Change SIMILARITY_THRESHOLD here to tune the gate without touching
  any other module.
"""

# ── Configurable threshold ────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.60
# ─────────────────────────────────────────────────────────────────────────────


def is_evidence_sufficient(score: float, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """
    Returns True when the similarity score meets or exceeds the threshold,
    meaning the retrieved evidence is considered relevant enough to answer.
    """
    return score >= threshold


def abstain(score: float, source: str | None = None) -> dict:
    """
    Builds the structured abstention result returned when evidence is
    insufficient.  Gemini is NOT called when this path is taken.
    """
    return {
        "answered": False,
        "answer": (
            "I don't have enough reliable evidence in the available sources "
            "to answer this question."
        ),
        "reason": "insufficient_evidence",
        "retrieval_score": round(score, 4),
        "threshold": SIMILARITY_THRESHOLD,
        "source": source,
    }


def evaluate_evidence(score: float, source: str | None = None) -> dict | None:
    """
    Convenience wrapper used by rag_basic.py.

    Returns:
        None          — evidence is sufficient; proceed with Gemini.
        dict          — abstention result; skip Gemini and return this to caller.
    """
    if is_evidence_sufficient(score):
        return None          # green light
    return abstain(score, source)
