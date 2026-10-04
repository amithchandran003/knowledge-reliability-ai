"""Small, source-type-specific retrieval contract for V2."""

from dataclasses import dataclass


@dataclass
class FetchResult:
    success: bool
    raw_content: str | None = None
    content_type: str | None = None
    status_code: int | None = None
    error_code: str | None = None


class SourceAdapter:
    """Base interface implemented by each supported source type."""

    source_type = "base"

    def fetch(self, source_config: dict) -> FetchResult:
        raise NotImplementedError
