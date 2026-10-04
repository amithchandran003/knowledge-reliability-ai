"""
Semantic retrieval component using Google Gemini Embeddings and Cosine Similarity.

Features:
- Gemini semantic embeddings
- Cosine similarity
- Local embedding cache
- Query embedding cache
- Avoids repeated Gemini embedding requests
"""

import hashlib
import json
import math
import os
import requests


GEMINI_REQUEST_TIMEOUT_SECONDS = 30

CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "embedding_cache",
)

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(text: str, model: str) -> str:
    """
    Creates a stable cache key based on model + text.
    """
    raw = f"{model}\n{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(text: str, model: str) -> str:
    """
    Returns the local cache file path for an embedding.
    """
    return os.path.join(
        CACHE_DIR,
        f"{_cache_key(text, model)}.json",
    )


def get_embedding(
    text: str,
    api_key: str,
    model: str = "gemini-embedding-001",
) -> list:
    """
    Generates an embedding using Gemini.

    If the same text was embedded previously,
    the cached embedding is returned instead of
    making another Gemini API request.
    """

    cache_file = _cache_path(text, model)

    # --------------------------------------------------
    # 1. Try local cache
    # --------------------------------------------------

    if os.path.exists(cache_file):

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)

            if isinstance(cached, list):
                return cached

        except (OSError, json.JSONDecodeError):
            pass

    # --------------------------------------------------
    # 2. Call Gemini only when not cached
    # --------------------------------------------------

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:embedContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        }
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as err:

        raise RuntimeError(
            "Gemini embedding request failed due to "
            "a network or service error."
        ) from err

    # --------------------------------------------------
    # 3. Handle rate limiting clearly
    # --------------------------------------------------

    if response.status_code == 429:

        raise RuntimeError(
            "Gemini embedding rate limit reached (HTTP 429). "
            "Cached embeddings are available for previously "
            "processed content; please wait before embedding "
            "new content."
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"Gemini embedding request failed "
            f"(HTTP {response.status_code})."
        )

    # --------------------------------------------------
    # 4. Parse response
    # --------------------------------------------------

    try:

        data = response.json()

        embedding = data["embedding"]["values"]

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as err:

        raise RuntimeError(
            "Gemini embedding API returned "
            "an unexpected response format."
        ) from err

    # --------------------------------------------------
    # 5. Save embedding locally
    # --------------------------------------------------

    try:

        with open(
            cache_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                embedding,
                f,
            )

    except OSError:
        # Cache failure should never break retrieval.
        pass

    return embedding


def cosine_similarity(
    vec_a: list,
    vec_b: list,
) -> float:
    """
    Calculates cosine similarity between two vectors.
    """

    dot_product = sum(
        a * b
        for a, b in zip(vec_a, vec_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vec_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vec_b)
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


class SemanticRetriever:
    """
    In-memory semantic retriever using
    cached Gemini embeddings.
    """

    def __init__(
        self,
        documents: list,
        api_key: str,
    ):

        self.documents = documents

        self.api_key = api_key

        self.document_embeddings = []

        for doc in self.documents:

            text_to_embed = (
                f"{doc.title}\n"
                f"{doc.content}"
            )

            embedding = get_embedding(
                text_to_embed,
                self.api_key,
            )

            self.document_embeddings.append(
                embedding
            )

    def retrieve(
        self,
        query: str,
    ):

        query_embedding = get_embedding(
            query,
            self.api_key,
        )

        best_doc = None

        best_score = -1.0

        for (
            doc,
            doc_emb,
        ) in zip(
            self.documents,
            self.document_embeddings,
        ):

            score = cosine_similarity(
                query_embedding,
                doc_emb,
            )

            if score > best_score:

                best_score = score

                best_doc = doc

        return (
            best_doc,
            best_score,
        )