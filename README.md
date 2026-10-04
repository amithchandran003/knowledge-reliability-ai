# Knowledge Reliability AI

A reliability-aware AI system that tracks the evidence behind AI answers, monitors source changes, identifies affected claims, selectively re-evaluates impacted answers, and generates notifications when previously reliable information becomes stale.

## Problem

Traditional RAG systems can retrieve current information when a user asks a question, but previously generated answers can become outdated when their underlying sources change.

**Source → Evidence → Claim → Answer**

## Core Workflow

1. Retrieve information from registered sources.
2. Generate a grounded AI answer.
3. Store answer provenance.
4. Track individual claims and their supporting evidence.
5. Periodically monitor registered sources.
6. Detect changes between source snapshots.
7. Identify affected claims.
8. Re-evaluate only impacted answers.
9. Mark invalid answers as stale.
10. Generate a reliability notification when required.

## Key Features

### Dynamic RAG
Retrieves content from registered web pages and JSON/API sources and uses retrieved evidence to generate grounded answers.

### Evidence Gate
Evaluates retrieved evidence before accepting an answer.

### Multi-Claim Provenance
Tracks individual claims and their supporting evidence.

### Source Monitoring
Periodically checks registered sources and detects content changes using snapshots.

### Selective Re-evaluation
A source change is traced through:

**Changed Evidence → Affected Claim → Affected Answer**

Only impacted answers are re-evaluated.

### Notifications
Material changes that invalidate stored answers can generate notifications containing the relevant change and corrected information.

### Background Scheduler
`scheduler.py` provides a lightweight background monitoring loop.

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- Dynamic RAG
- Web and JSON/API source adapters
- Snapshot-based change detection
- Semantic change classification
- Automated background monitoring

## Running the Application

Create a `.env` file using `.env.example` as a template and configure the required Gemini API key.

Do not commit `.env` to Git.

Run the application:

    python -m streamlit run app.py

Run the monitoring scheduler:

    python scheduler.py

For a quick scheduler demonstration:

    MONITOR_INTERVAL_MINUTES=1 python scheduler.py

## Testing

The project includes tests covering:

- Dynamic RAG
- Evidence gating
- Source registry
- Claim/evidence impact analysis
- Multi-claim provenance
- Answer re-evaluation
- Notifications
- Validity history
- Source monitoring

Current verification: **26 tests passed, 0 failed.**

Run the tests with:

    python -m unittest discover -s . -p "test_*.py"

Python compilation was also verified successfully.

## Demonstrated Scenarios

### Material Change

A controlled policy change:

**30 days → 15 days**

resulted in:

- affected answer identified
- answer re-evaluated
- previous answer invalidated
- corrected answer generated
- notification generated

### Unrelated Change

An unrelated source change resulted in:

- existing answer remaining valid
- no unnecessary re-evaluation
- no notification

This demonstrates selective impact analysis.

## Limitations

This is a research/prototype implementation.

It does not guarantee that every possible real-world source change or semantic dependency will be detected.

The current claim extraction approach uses deterministic sentence-level extraction rather than a full semantic claim decomposition system.

The scheduler is a lightweight local process rather than a production distributed task system.

Source availability, network conditions, external APIs, and model behavior can affect the system.

## Future Improvements

- More advanced semantic claim extraction
- Confidence-based staleness prediction
- Production background workers
- Database-backed provenance
- More source adapters
- Improved source authority and conflict analysis
- Larger evaluation datasets
- Cost and latency optimization

## Project Goal

The long-term goal is to move from:

> "Can an AI answer this question?"

toward:

> "Can an AI know when an answer it gave earlier is no longer reliable?"
