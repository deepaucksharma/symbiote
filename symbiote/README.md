# Symbiote - Cognitive Prosthetic

A local-first desktop system for capturing thoughts with near-zero friction and assembling actionable context in <500ms.

## Features

- **Near-zero latency capture** (<200ms) with WAL-based durability
- **Racing search strategy** returns first useful context in <100ms p50
- **Explainable suggestions** with machine-readable receipts
- **Local-first** with optional cloud escalation on explicit consent
- **Graceful degradation** to plain Markdown files

## Architecture

### Operating Modes

1. **Reflex Mode** (Always On): Instant capture and context assembly
2. **Synthesis Mode** (Background): Pattern detection every ~5 minutes
3. **Deliberation Mode** (On-Demand): Complex reasoning with optional cloud

### Core Components

- **Capture Service**: WAL-based lossless writes
- **Search Orchestrator**: Parallel FTS/Vector/Recents with racing
- **Analytics Indexer**: DuckDB for receipts and structured queries
- **Event Bus**: Async pub/sub for inter-module communication

## Quick Start

### Installation

```bash
# Install dependencies with Poetry
poetry install

# Or with pip
pip install -r requirements.txt
```

### Configuration

Edit `symbiote.yaml` to configure your vault path and preferences:

```yaml
vault_path: "/path/to/your/vault"
hotkeys:
  capture: "Ctrl+Shift+Space"
  search: "Ctrl+Shift+F"
```

### Usage

```bash
# Start the daemon
sym daemon start

# Capture a thought
sym "Email Priya about API change"

# Search for context
sym search "Q3 strategy"

# Get suggestions
sym suggest

# Check health
sym-doctor
```

## CLI Commands

### Basic Capture
```bash
sym "text"                    # Quick capture
sym capture "text" -t task    # Capture as task
sym capture "text" -p project # Assign to project
```

### Search
```bash
sym search "query"            # Search for context
sym search "query" -p project # Filter by project
```

### Daemon Management
```bash
sym daemon start              # Start daemon
sym daemon stop               # Stop daemon
sym daemon status             # Check status
```

### Diagnostics
```bash
sym-doctor                    # Run health checks
sym-doctor --reindex          # Rebuild indexes
sym-doctor --stats            # Show statistics
```

## Performance Targets

- **Capture**: p99 ≤ 200ms
- **Context Card**: p50 ≤ 100ms, p95 ≤ 300ms
- **Memory**: < 1.5GB steady-state
- **Availability**: ≥ 99.9% when OS is up

## Data Model

### Vault Structure
```
vault/
  journal/YYYY/MM/DD.md      # Daily journal
  tasks/task-<ulid>.md       # Individual tasks
  notes/<slug>-<ulid>.md     # Notes
  .sym/
    wal/                     # Write-ahead logs
    analytics.db             # DuckDB indexes
```

### Front-matter Schema
```yaml
---
id: <ulid>
type: task|note
captured: <ISO-8601>
title: <string>
project: <string>
tags: [<string>]
---
```

## Development

### Project Structure
```
symbiote/
  daemon/                    # Core daemon
    capture.py              # WAL-based capture
    search.py               # Search orchestrator
    bus.py                  # Event bus
    indexers/               # FTS/Vector/Analytics
  cli/                      # CLI tools
    sym.py                  # Main CLI
  scripts/
    doctor.py               # Diagnostics
```

### Running Tests
```bash
poetry run pytest
```

## Roadmap

- [ ] Full-text search with Tantivy
- [ ] Vector embeddings with LanceDB
- [ ] Synthesis worker for pattern detection
- [ ] Voice capture with Whisper.cpp
- [ ] Platform-specific hotkey support
- [ ] Deliberation mode with LLM integration

## License

MIT