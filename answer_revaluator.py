# answer_revaluator.py
"""
Claim-Level Answer Re-evaluation Module.

Uses Gemini (gemini-3.6-flash) to evaluate whether an AI answer generated
from an older source version remains factually accurate after the source is updated,
or whether specific factual claims have become outdated or contradicted.
"""

import json
import os
import time
import requests


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


def reevaluate_answer(
    old_source: str,
    new_source: str,
    question: str,
    previous_answer: str,
    api_key: str = None,
) -> dict:
    """
    Evaluates whether previous_answer is still factually valid under new_source.
    Returns structured JSON with fields:
      - answer_still_valid: bool
      - affected_claim: str or None
      - old_information: str or None
      - new_information: str or None
      - corrected_answer: str or None
      - confidence: str ("high" | "medium" | "low")
      - explanation: str
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please configure .env.")

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

    prompt = f"""You are a factual consistency and claim re-evaluation evaluator.
Your task is to analyze whether a previously generated AI answer is still factually valid given that the underlying source document has been updated.

[CONTEXT]
User Question:
{question}

Previous AI Answer:
{previous_answer}

Previous Source Content:
\"\"\"
{old_source}
\"\"\"

New Updated Source Content:
\"\"\"
{new_source}
\"\"\"

[INSTRUCTIONS]
1. Identify the specific factual claim(s) made in the Previous AI Answer.
2. Determine if those claims are still completely supported by the New Updated Source Content.
3. If the source changes do NOT affect the claims in the previous answer (e.g. modifications were in unrelated sentences or details not asked about), set answer_still_valid to true.
4. If the new source contradicts, changes, or invalidates the claims in the previous answer, set answer_still_valid to false.
5. If the new source does not contain enough information to determine the answer to the question, set answer_still_valid to false and explain that verification is insufficient.
6. Formulate a corrected_answer using ONLY the new updated source content (or null if the answer is still valid).
7. Do NOT invent, assume, or extrapolate any information.

Return a JSON object with these EXACT keys:
{{
  "answer_still_valid": <boolean>,
  "affected_claim": <string or null>,
  "old_information": <string or null>,
  "new_information": <string or null>,
  "corrected_answer": <string or null>,
  "confidence": <"high" | "medium" | "low">,
  "explanation": <string>
}}
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
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

        # Retry temporary Gemini/API failures.
    # 429 = rate limit
    # 500/502/503/504 = temporary server/service failures
    retryable_status_codes = {429, 500, 502, 503, 504}

    max_attempts = 4
    response = None

    for attempt in range(1, max_attempts + 1):

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException as err:

            if attempt == max_attempts:
                raise RuntimeError(
                    "Gemini re-evaluation request failed after "
                    f"{max_attempts} attempts due to a network error."
                ) from err

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"Gemini request failed on attempt {attempt}. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)
            continue

        # Successful response
        if response.status_code == 200:
            break

        # Temporary failure — retry
        if response.status_code in retryable_status_codes:

            if attempt == max_attempts:
                raise RuntimeError(
                    "Gemini re-evaluation request failed after "
                    f"{max_attempts} attempts "
                    f"(HTTP {response.status_code})."
                )

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"Gemini returned HTTP {response.status_code} "
                f"on attempt {attempt}. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)
            continue

        # Permanent/unexpected HTTP error
        raise RuntimeError(
            f"Gemini re-evaluation request failed "
            f"(HTTP {response.status_code})."
        )

    data = response.json()
    try:
        reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(reply_text)
        return result
    except (KeyError, IndexError, json.JSONDecodeError) as err:
        raise RuntimeError("Gemini re-evaluation API returned an unexpected response format.") from err


def print_evaluation_report(test_name: str, test_data: dict, result: dict):
    """Formats and prints the re-evaluation report clearly."""
    print("=" * 65)
    print(f"TEST: {test_name}")
    print("=" * 65)
    print(f"Question:        {test_data['question']}")
    print(f"Previous Answer: {test_data['previous_answer']}")
    print(f"Old Source:      {test_data['old_source']}")
    print(f"New Source:      {test_data['new_source']}")
    print("-" * 65)
    print("RE-EVALUATION RESULT:")
    print(f"Answer Still Valid: {result.get('answer_still_valid')}")
    print(f"Confidence:         {result.get('confidence')}")
    print(f"Affected Claim:     {result.get('affected_claim')}")
    print(f"Old Information:    {result.get('old_information')}")
    print(f"New Information:    {result.get('new_information')}")
    print(f"Corrected Answer:   {result.get('corrected_answer')}")
    print(f"Explanation:        {result.get('explanation')}")
    print("=" * 65 + "\n")


def run_tests():
    """Runs the two controlled test scenarios."""
    # TEST A: Substantive claim change (Refund 30d -> 15d)
    test_a_data = {
        "question": "What is the refund period for customer purchases?",
        "previous_answer": "According to the policy, the refund period is 30 days.",
        "old_source": "Refund period is 30 days.",
        "new_source": "Refund period is 15 days.",
    }

    print("Running Test A (Substantive change: 30 days -> 15 days)...")
    result_a = reevaluate_answer(
        old_source=test_a_data["old_source"],
        new_source=test_a_data["new_source"],
        question=test_a_data["question"],
        previous_answer=test_a_data["previous_answer"],
    )
    print_evaluation_report("TEST A (Policy Changed: 30d -> 15d)", test_a_data, result_a)

    # TEST B: Unrelated change (Support channel changed, refund stays 30d)
    test_b_data = {
        "question": "What is the refund period for customer purchases?",
        "previous_answer": "According to the policy, the refund period is 30 days.",
        "old_source": "Refund period is 30 days. Contact support by email.",
        "new_source": "Refund period is 30 days. Contact support through the help center.",
    }

    print("Running Test B (Unrelated change: Support channel changed, refund unchanged)...")
    result_b = reevaluate_answer(
        old_source=test_b_data["old_source"],
        new_source=test_b_data["new_source"],
        question=test_b_data["question"],
        previous_answer=test_b_data["previous_answer"],
    )
    print_evaluation_report("TEST B (Unrelated Change: Support channel updated)", test_b_data, result_b)


if __name__ == "__main__":
    run_tests()
