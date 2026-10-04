"""Orchestrates one monitoring cycle across enabled sources."""

from answer_memory import load_answers
from source_registry import load_sources
from v2_monitor_pipeline import run_v2_monitor


def run_monitoring_cycle() -> list[dict]:
    """Run V2 monitoring for every enabled registered source."""

    sources = load_sources(enabled_only=True)
    answers = load_answers()

    results = []

    for source in sources:
        try:
            result = run_v2_monitor(
                source_config=source,
                answers=answers,
            )

            results.append(
                {
                    "source_id": source.get("source_id"),
                    "source_identifier": source.get(
                        "source_identifier"
                    ),
                    "source_url": source.get("source_url"),
                    "result": result,
                }
            )

        except Exception as error:
            results.append(
                {
                    "source_id": source.get("source_id"),
                    "source_identifier": source.get(
                        "source_identifier"
                    ),
                    "source_url": source.get("source_url"),
                    "result": {
                        "status": "error",
                        "error": str(error),
                        "notifications": [],
                    },
                }
            )

    return results


if __name__ == "__main__":
    import json

    results = run_monitoring_cycle()

    print(
        json.dumps(
            results,
            indent=2,
            default=str,
        )
    )