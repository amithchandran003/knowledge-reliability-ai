"""Background scheduler for Knowledge Reliability AI."""

import os
import time
from datetime import datetime

from monitor_worker import run_monitoring_cycle


INTERVAL_MINUTES = int(
    os.environ.get("MONITOR_INTERVAL_MINUTES", "60")
)


def run_once():
    """Run one complete monitoring cycle."""

    timestamp = datetime.now().astimezone().isoformat()

    print()
    print("=" * 70)
    print(f"Knowledge Reliability AI - Monitoring Cycle")
    print(f"Started: {timestamp}")
    print("=" * 70)

    try:
        results = run_monitoring_cycle()

        for item in results:
            source = item.get("source_identifier", "Unknown source")
            result = item.get("result", {})
            status = result.get("status", "unknown")

            print(f"\nSource: {source}")
            print(f"Status: {status}")

            if status == "changed":
                print(
                    f"Changed chunks: "
                    f"{len(result.get('changes', []))}"
                )
                print(
                    f"Candidate claims: "
                    f"{result.get('candidate_claims', 0)}"
                )
                print(
                    f"Re-evaluations: "
                    f"{len(result.get('reevaluations', []))}"
                )
                print(
                    f"Notifications: "
                    f"{len(result.get('notifications', []))}"
                )

            elif status == "unchanged":
                print("No source change detected.")

            elif status == "initial_snapshot":
                print("Initial snapshot created.")

            elif status == "unavailable":
                print("Source unavailable.")

            elif status == "error":
                print(
                    f"Error: {result.get('error', 'Unknown error')}"
                )

    except Exception as error:
        print(f"Monitoring cycle failed: {error}")

    print()
    print(
        f"Next monitoring cycle in "
        f"{INTERVAL_MINUTES} minute(s)."
    )
    print("=" * 70)


def main():
    print("🛡️ Knowledge Reliability AI Scheduler")
    print(
        f"Monitoring interval: "
        f"{INTERVAL_MINUTES} minute(s)"
    )

    while True:
        run_once()
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
