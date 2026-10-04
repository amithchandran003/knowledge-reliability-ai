"""Adapter for public JSON HTTP APIs."""

import requests

from .base import FetchResult, SourceAdapter


class JsonApiAdapter(SourceAdapter):
    source_type = "json_api"
    REQUEST_TIMEOUT_SECONDS = 15

    def fetch(self, source_config: dict) -> FetchResult:
        try:
            response = requests.get(
                source_config["source_url"],
                headers={"Accept": "application/json", "User-Agent": "HallucinationDetector/2.0"},
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            return FetchResult(success=False, error_code="timeout")
        except requests.RequestException:
            return FetchResult(success=False, error_code="network_error")

        if not response.ok:
            return FetchResult(success=False, status_code=response.status_code, error_code="http_error")

        return FetchResult(
            success=True,
            raw_content=response.text,
            content_type=response.headers.get("Content-Type", "application/json"),
            status_code=response.status_code,
        )
