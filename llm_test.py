# llm_test.py
"""Minimal LLM API experiment using Google Gemini.

Loads the API key securely from the .env file.
"""

import os
import requests


GEMINI_REQUEST_TIMEOUT_SECONDS = 30

# 1. Load .env file
# Try using python-dotenv if installed, otherwise fallback to standard library reading
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


def send_prompt(prompt: str) -> str:
    """Sends a single text prompt to the Gemini API and returns the response text."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        return (
            "Error: GEMINI_API_KEY is not configured.\n"
            "Please open .env and replace 'your_gemini_api_key_here' with your valid Gemini API key."
        )

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
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
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return reply
    except (KeyError, IndexError, TypeError):
        return "Gemini API returned an unexpected response format."


if __name__ == "__main__":
    test_prompt = "In one concise sentence, what is an AI hallucination?"
    print(f"Prompt: {test_prompt}\n")

    response_text = send_prompt(test_prompt)
    print("LLM Response:")
    print(response_text)
