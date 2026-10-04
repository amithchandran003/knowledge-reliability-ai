"""Adapters for retrieving different real-world source types."""

from .json_api import JsonApiAdapter
from .web_page import WebPageAdapter


def get_adapter(source_type: str):
    """Returns the adapter supported by this V2 vertical slice."""
    adapters = {
        "web_page": WebPageAdapter,
        "json_api": JsonApiAdapter,
    }
    try:
        return adapters[source_type]()
    except KeyError as err:
        raise ValueError(f"Unsupported source type: {source_type}") from err
