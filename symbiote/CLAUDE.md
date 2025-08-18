# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Symbiote is a **Cognitive Prosthetic** - a local-first desktop system designed for near-zero friction thought capture and context assembly. It implements a sophisticated architecture based on racing search strategies, WAL-based durability, and privacy-first design with optional cloud escalation.

## Key Architectural Principles

### Operating Modes (Parallel, Not Hierarchical)
1. **Reflex Mode** - Always-on capture with <200ms p99 latency target
2. **Synthesis Mode** - Background pattern detection every ~5 minutes
3. **Deliberation Mode** - On-demand complex reasoning with consent-gated cloud

### Performance SLOs (Critical)
- **Capture**: p99 ≤ 200ms (WAL write + fsync)
- **Context Search**: p50 ≤ 100ms, p95 ≤ 300ms (first useful result)
- **Memory Budget**: < 1.5GB steady-state
- **Availability**: ≥ 99.9% when OS is up

### Core Design Decisions
- **Files as truth** (Markdown vault) > DB as truth (portability, Git-friendly)
- **Racing search** > single best index (FTS/Vector/Recents race, return first useful)
- **Receipts everywhere** - Every suggestion includes explainability with sources
- **Consent-gated cloud** - Local-only by default, explicit consent per event

## Development Commands

```bash
# Initial setup
make install          # Install dependencies (tries poetry first, falls back to pip)
make setup           # Initialize vault with example data
python scripts/setup.py --vault /custom/path  # Custom vault location

# Development cycle
make run             # Start daemon (python -m symbiote.cli.sym daemon start)
make test            # Run test suite (pytest tests/ -v)
make lint            # Check code quality (ruff + mypy)
make format          # Auto-format code (black + ruff --fix)

# Testing specific components
pytest tests/test_capture.py -v              # Test capture service
pytest tests/test_event_bus.py::test_wildcard_subscription -v  # Single test

# Operations
sym daemon start --config custom.yaml        # Start with custom config
sym daemon status                            # Check daemon health
sym-doctor                                   # Run health checks
sym-doctor --reindex                         # Rebuild all indexes
sym-doctor --stats                           # Show performance metrics

# API testing (daemon must be running on localhost:8765)
curl -X POST http://localhost:8765/capture -d '{"text":"test"}'
curl http://localhost:8765/metrics?format=prometheus
curl http://localhost:8765/health
```

## Critical Code Paths

### Capture Pipeline (daemon/capture.py)
1. Validate payload (≤8k chars)
2. Append WAL record with fsync (critical path - must be <200ms)
3. Materialize to vault asynchronously
4. Emit `capture.written` event for indexers

### Search Racing (daemon/search.py + algorithms.py)
1. Launch FTS/Vector/Recents in parallel
2. Calculate utility score for each result:
   - FTS: base_score * 1.00 + project_match * 0.10 + recency * 0.05
   - Vector: base_score * 0.95 + project_match * 0.10 + recency * 0.05  
   - Recents: base_score * 0.80 + project_match * 0.20
3. Return first result with utility ≥ 0.55 (usefulness threshold)
4. Continue merging late results up to 1s

### Consent & Privacy (daemon/consent.py)
- ConsentScope levels: DENY → ALLOW_TITLES → ALLOW_BULLETS → ALLOW_EXCERPTS
- RedactionEngine detects PII (emails, phones, names)
- Every cloud call requires preview + explicit consent
- Audit log tracks all outbound data

## API Contract (Part 3 Spec)

### Error Model
```json
{
  "error": {
    "code": "vault_locked|storage_full|queue_overflow|invalid_request",
    "message": "human readable",
    "retry_after_ms": 2000
  }
}
```

### Key Endpoints
- `POST /capture` → 201 with `{id, status, ts}`
- `GET /context?q=query` → Racing search with latency tracking
- `POST /suggest` → Suggestion with receipts_id
- `POST /deliberate` → May require consent flow
- `GET /metrics?format=prometheus|json` → Observability

## Configuration (symbiote.yaml)

Key settings that affect behavior:
- `vault_path`: Source of truth location
- `indices.fts/vector/analytics`: Enable/disable indexers
- `privacy.allow_cloud`: Default false
- `privacy.mask_pii_default`: Default true
- `performance.max_search_workers`: Concurrency limit
- `synthesis.suggest_link_threshold`: 0.70 default

## Metrics & Observability

The system tracks latency histograms for all operations. Check SLO compliance:
```python
# In code
from daemon.metrics import LatencyTimer, get_metrics

with LatencyTimer("operation_name"):
    # operation

# Check SLOs
metrics = get_metrics()
slos = metrics.check_slos()  # Returns dict of SLO -> pass/fail
```

## Pending Implementations

Components with stubs that need completion:
1. **FTS Indexer** (Tantivy) - See schema in Part 2 spec
2. **Vector Indexer** (LanceDB) - 512-token chunks, 384-dim embeddings
3. **Voice Capture** (Whisper.cpp) - Child process via stdin/stdout
4. **LLM Adapter** (Ollama) - Local inference for synthesis

## Testing Strategy

- **Unit tests**: Fast, isolated component tests
- **Integration tests**: Spawn daemon, hit APIs
- **Performance tests**: Verify latency SLOs with mock data
- **Recovery drills**: Simulate crashes, verify WAL replay

## Important Invariants

1. **Capture never blocks on indexing** - WAL write is the only critical path
2. **No suggestion without receipts** - Explainability is mandatory
3. **Receipts are immutable** - Corrections create new versions
4. **Search races return first useful** - Don't wait for all strategies
5. **Privacy by default** - No network calls without explicit consent