"""File-based source registry and safe CLI registration for V3 web sources."""

import argparse
import hashlib
import ipaddress
import json
import os
import tempfile
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlsplit, urlunsplit


SOURCES_FILE = os.path.join(os.path.dirname(__file__), "data", "sources.json")
SENSITIVE_QUERY_PARAMETERS = {
    "api_key", "api-key", "apikey", "token", "access_token", "access-token",
    "authorization", "password", "secret",
}


def validate_web_url(url: str) -> str:
    """Validates and canonicalizes a public HTTP(S) URL without its fragment."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL is required.")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as err:
        raise ValueError("URL has an invalid port.") from err

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs with embedded credentials are not allowed.")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname.")

    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("Localhost URLs are not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_loopback or address.is_private or address.is_link_local
        or address.is_unspecified or address.is_multicast or address.is_reserved
    ):
        raise ValueError("Private, loopback, and link-local IP URLs are not allowed.")

    query_names = {name.lower() for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}
    if query_names & SENSITIVE_QUERY_PARAMETERS:
        raise ValueError("URLs containing credential query parameters are not allowed.")

    host_for_url = f"[{host}]" if address is not None and address.version == 6 else host
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host_for_url = f"{host_for_url}:{port}"
    return urlunsplit((scheme, host_for_url, parsed.path or "/", parsed.query, ""))


def _read_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {"sources": []}
    with open(path, encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, dict) or not isinstance(registry.get("sources", []), list):
        raise ValueError("Source registry must contain a sources list.")
    return registry


def _write_registry(registry: dict, path: str) -> None:
    """Writes the registry atomically to avoid partially-written JSON files."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix="sources-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def load_sources(path: str = SOURCES_FILE, enabled_only: bool = True) -> list[dict]:
    """Loads configured sources; disabled examples are excluded by default."""
    return [source for source in _read_registry(path)["sources"] if source.get("enabled", False) or not enabled_only]


def get_source(source_id: str, path: str = SOURCES_FILE) -> dict | None:
    """Returns one configured source by its stable source_id."""
    return next((source for source in load_sources(path, enabled_only=False) if source["source_id"] == source_id), None)


def register_web_source(name: str, url: str, path: str = SOURCES_FILE) -> dict:
    """Registers a user-configured web page, rejecting duplicate canonical URLs."""
    source_identifier = name.strip() if isinstance(name, str) else ""
    if not source_identifier:
        raise ValueError("Source name is required.")
    canonical_url = validate_web_url(url)
    registry = _read_registry(path)
    for source in registry["sources"]:
        existing_url = source.get("source_url")
        if existing_url:
            try:
                duplicate = validate_web_url(existing_url) == canonical_url
            except ValueError:
                duplicate = False
            if duplicate:
                raise ValueError(f"Source URL is already registered as {source['source_id']}.")

    record = {
        "source_id": f"source_{hashlib.sha256(canonical_url.encode('utf-8')).hexdigest()[:16]}",
        "enabled": True,
        "source_type": "web_page",
        "source_identifier": source_identifier,
        "source_url": canonical_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["sources"].append(record)
    _write_registry(registry, path)
    return record


def main(argv: list[str] | None = None, path: str = SOURCES_FILE) -> None:
    """Runs the V3 source-registration CLI."""
    parser = argparse.ArgumentParser(description="Manage hallucination-detector web sources.")
    commands = parser.add_subparsers(dest="command", required=True)
    add_parser = commands.add_parser("add", help="Register a public web page.")
    add_parser.add_argument("name")
    add_parser.add_argument("url")
    commands.add_parser("list", help="List all registered sources.")
    show_parser = commands.add_parser("show", help="Display one source configuration.")
    show_parser.add_argument("source_id")
    args = parser.parse_args(argv)

    try:
        if args.command == "add":
            record = register_web_source(args.name, args.url, path)
            print(f"Registered source: {record['source_identifier']}")
            print(f"Source ID: {record['source_id']}")
        elif args.command == "list":
            for source in load_sources(path, enabled_only=False):
                print(f"{source['source_id']}\t{source['source_type']}\t{source['source_identifier']}\t{source['source_url']}")
        elif args.command == "show":
            source = get_source(args.source_id, path)
            if source is None:
                raise ValueError(f"Unknown source_id: {args.source_id}")
            print(json.dumps(source, indent=2))
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
