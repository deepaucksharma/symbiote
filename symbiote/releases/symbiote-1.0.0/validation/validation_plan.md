# End-to-End Validation Plan for Symbiote Cognitive Prosthetic

## Executive Summary
This validation plan ensures the Symbiote implementation meets all PRD requirements, focusing on:
- Performance SLOs (capture <200ms p99, context <100ms p50)
- Privacy-first with explicit consent
- Explainability (receipts for every suggestion)
- Graceful degradation
- Zero-friction capture

## 1. Requirements Traceability Matrix

| PRD Requirement | Implementation Component | Validation Test |
|-----------------|-------------------------|-----------------|
| Capture p99 ≤ 200ms | WAL-based CaptureService | PERF-01 |
| Context p50 ≤ 100ms | Racing SearchOrchestrator | PERF-02 |
| Decisions with receipts | Receipts system in algorithms.py | EXPL-01 |
| Local-first, explicit consent | ConsentManager, RedactionEngine | PRIV-01 |
| Graceful degradation | Fallback to recents, WAL recovery | RESIL-01 |
| Zero capture friction | Global hotkey, no focus steal | UX-01 |
| Daily synthesis | SynthesisWorker | SYNTH-01 |
| FTS + Vector racing | FTSIndexer, VectorIndexer | SEARCH-01 |

## 2. Critical Path Validation Tests

### 2.1 Capture Flow Validation

#### TEST: CAPTURE-01 - End-to-End Capture
```bash
# Test capture through API
curl -X POST http://localhost:8765/capture \
  -H "Content-Type: application/json" \
  -d '{"text": "Test capture for validation", "type_hint": "note"}'

# Verify WAL entry
ls -la vault/.sym/wal/*.log | tail -1
grep "Test capture for validation" vault/.sym/wal/*.log

# Check materialization
sleep 1
grep "Test capture for validation" vault/daily/*.md
```

**Expected Results:**
- API returns 201 with capture ID within 200ms
- WAL file contains entry immediately
- Daily note updated within 1s

#### TEST: CAPTURE-02 - Crash Recovery
```bash
# Start capture and kill daemon mid-write
python scripts/chaos_inject.py --scenario kill_during_capture

# Restart daemon
python -m symbiote.daemon.main &

# Verify WAL replay
curl http://localhost:8765/health
```

**Expected Results:**
- WAL preserves incomplete capture
- Daemon replays WAL on startup
- No data loss

### 2.2 Search & Context Assembly Validation

#### TEST: SEARCH-01 - Racing Strategy
```bash
# Generate test data
python scripts/gen_vault.py --size 5000 --path vault

# Test search latency
python scripts/run_benchmarks.py --operation search --iterations 100

# Verify racing behavior
curl "http://localhost:8765/context?q=strategy&debug=true"
```

**Expected Results:**
- p50 latency < 100ms
- p95 latency < 300ms
- Debug shows first_useful_ms < total_ms
- Results from multiple sources (fts/vector/recents)

#### TEST: SEARCH-02 - Utility Scoring
```python
# Test utility calculation
import asyncio
from symbiote.daemon.algorithms import SearchCandidate
from datetime import datetime, timedelta

# High-value candidate (FTS, project match, recent)
candidate = SearchCandidate(
    id="test1",
    title="Q3 Strategy",
    path="notes/q3.md",
    snippet="Strategy outline",
    base_score=0.8,
    source="fts",
    project="planning",
    modified=datetime.utcnow() - timedelta(hours=1)
)

utility = candidate.calculate_utility(query_project="planning")
assert utility > 0.8, f"Expected high utility, got {utility}"

# Verify source weights
assert candidate.source_weight == 1.0  # FTS
```

### 2.3 Suggestion & Receipts Validation

#### TEST: EXPL-01 - Receipts Generation
```bash
# Request suggestion
curl -X POST http://localhost:8765/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "situation": {
      "query": "Q3 planning",
      "free_minutes": 30,
      "project": "strategy"
    }
  }'

# Fetch receipt
RECEIPT_ID=$(curl ... | jq -r '.suggestion.receipts_id')
curl http://localhost:8765/receipts/$RECEIPT_ID
```

**Expected Results:**
- Suggestion includes receipts_id
- Receipt contains:
  - sources (with IDs and titles)
  - heuristics (rules applied)
  - confidence level (high/medium/low)
  - version number

### 2.4 Privacy & Consent Validation

#### TEST: PRIV-01 - Consent Gate
```bash
# Request deliberation with cloud
curl -X POST http://localhost:8765/deliberate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare team expansion options",
    "allow_cloud": true
  }'

# Should return consent request
# Verify redaction preview
```

**Expected Results:**
- requires_consent: true
- redaction_preview shows [EMAIL_REDACTED], [PHONE_REDACTED]
- action_id for consent decision
- No data sent without consent

#### TEST: PRIV-02 - PII Redaction
```python
from symbiote.daemon.consent import RedactionEngine

text = "Contact john@example.com at 555-1234, SSN: 123-45-6789"
redacted, counts = RedactionEngine.redact_text(
    text, 
    ["emails", "phones", "ssns"]
)

assert "[EMAIL_REDACTED]" in redacted
assert "[PHONE_REDACTED]" in redacted
assert "[SSN_REDACTED]" in redacted
assert counts["emails"] == 1
assert counts["phones"] == 1
assert counts["ssns"] == 1
```

### 2.5 Synthesis Validation

#### TEST: SYNTH-01 - Background Synthesis
```bash
# Force synthesis run
curl -X POST http://localhost:8765/admin/synthesis/run

# Check patterns
curl http://localhost:8765/admin/synthesis/patterns

# Verify daily note creation
ls -la vault/synthesis/$(date +%Y-%m-%d).md
```

**Expected Results:**
- Themes extracted from recent activity
- Suggested links with scores > 0.4
- Daily synthesis note created
- Patterns stored in analytics.db

## 3. Performance Validation Suite

### 3.1 Latency SLO Tests
```bash
# Run comprehensive performance tests
python scripts/run_benchmarks.py --all --duration 300

# Expected output format:
# Operation      p50    p95    p99    SLO    Status
# capture        45ms   120ms  185ms  200ms  ✓ PASS
# search_first   78ms   250ms  450ms  100ms  ✓ PASS (p50)
# suggest        890ms  1800ms 2100ms 1000ms ✓ PASS (p50)
```

### 3.2 Memory & Resource Tests
```bash
# Monitor memory during operation
python scripts/chaos_inject.py --scenario high_memory

# Check memory usage
curl http://localhost:8765/metrics | jq '.memory_mb'
```

**Expected:** Memory < 1.5GB under normal load

### 3.3 Concurrent Load Tests
```bash
# Simulate concurrent users
python scripts/chaos_inject.py --scenario concurrent_load

# Check SLO compliance under load
curl http://localhost:8765/health | jq '.slos'
```

## 4. User Journey Validation

### 4.1 Happy Path E2E Test
```python
# Complete user journey test
import asyncio
import httpx
import time

async def test_happy_path():
    async with httpx.AsyncClient() as client:
        # 1. Capture a thought
        capture_resp = await client.post(
            "http://localhost:8765/capture",
            json={"text": "Research WebRTC for Q3 project"}
        )
        assert capture_resp.status_code == 201
        capture_id = capture_resp.json()["id"]
        
        # 2. Search for context
        await asyncio.sleep(0.5)  # Let indexing complete
        search_resp = await client.get(
            "http://localhost:8765/context",
            params={"q": "WebRTC Q3"}
        )
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        assert any(capture_id in r.get("id", "") for r in results)
        
        # 3. Get suggestion
        suggest_resp = await client.post(
            "http://localhost:8765/suggest",
            json={
                "situation": {
                    "query": "WebRTC",
                    "free_minutes": 25
                }
            }
        )
        assert suggest_resp.status_code == 200
        suggestion = suggest_resp.json()["suggestion"]
        
        # 4. Verify receipt
        if suggestion:
            receipt_id = suggestion["receipts_id"]
            receipt_resp = await client.get(
                f"http://localhost:8765/receipts/{receipt_id}"
            )
            assert receipt_resp.status_code == 200
            receipt = receipt_resp.json()
            assert "sources" in receipt
            assert "confidence" in receipt

asyncio.run(test_happy_path())
```

## 5. Resilience & Degradation Tests

### 5.1 Index Corruption Recovery
```bash
# Test graceful degradation
python scripts/chaos_inject.py --scenario index_corruption

# Verify search still works (via recents)
curl "http://localhost:8765/context?q=test"
```

### 5.2 Disk Full Handling
```bash
# Simulate disk pressure
python scripts/chaos_inject.py --scenario disk_full

# Verify appropriate error handling
curl -X POST http://localhost:8765/capture \
  -d '{"text": "test"}' \
  | jq '.status'
```

## 6. Integration Test Suite

### 6.1 Run Full Integration Tests
```bash
# Run all integration tests
pytest symbiote/tests/integration/ -v --tb=short

# Expected coverage:
# - test_capture_appears_in_search ✓
# - test_capture_with_receipts ✓
# - test_early_return ✓
# - test_deliberate_requires_consent ✓
# - test_capture_latency_slo ✓
# - test_search_latency_slo ✓
```

### 6.2 Retrieval Quality Tests
```bash
# Generate test queries and ground truth
python scripts/gen_vault.py --with-queries

# Run retrieval evaluation
python scripts/eval_retrieval.py \
  --queries vault/.sym/test_queries.json \
  --k 5

# Expected metrics:
# Recall@5: > 0.8
# MRR: > 0.7
```

## 7. Acceptance Criteria Checklist

### Core Functionality
- [ ] Capture writes to WAL in <200ms p99
- [ ] Context assembly returns in <100ms p50
- [ ] Suggestions include complete receipts
- [ ] Daily synthesis generates themes
- [ ] Search races FTS/Vector/Recents

### Privacy & Security
- [ ] No data leaves device without consent
- [ ] PII is redacted in previews
- [ ] Audit log tracks all cloud calls
- [ ] Local-only mode fully functional

### Resilience
- [ ] WAL survives crashes
- [ ] Search degrades to recents if indexes fail
- [ ] Daemon recovers from kill
- [ ] Memory stays under 1.5GB

### User Experience
- [ ] Capture doesn't steal focus
- [ ] Visual confirmation "✓ Captured"
- [ ] Suggestions are actionable
- [ ] Receipts are readable

## 8. Validation Execution Plan

### Phase 1: Component Tests (Day 1)
1. Run unit tests: `pytest symbiote/tests/ -v`
2. Verify 95% code coverage
3. Check individual component SLOs

### Phase 2: Integration Tests (Day 2)
1. Start daemon with test vault
2. Run integration test suite
3. Execute chaos tests
4. Verify resilience

### Phase 3: Performance Tests (Day 3)
1. Generate 20k document vault
2. Run benchmark suite for 1 hour
3. Monitor memory and CPU
4. Validate all SLOs

### Phase 4: User Journey Tests (Day 4)
1. Complete happy path scenarios
2. Test edge cases
3. Verify accessibility (keyboard-only)
4. Check receipt quality

### Phase 5: Security & Privacy (Day 5)
1. Attempt data exfiltration (should fail)
2. Verify consent gates
3. Test redaction completeness
4. Audit log review

## 9. Go/No-Go Criteria

### Must Pass (P0)
- Capture latency p99 ≤ 200ms ✅
- Context assembly p50 ≤ 100ms ✅
- Zero data loss on crash ✅
- Consent required for cloud ✅
- Receipts on all suggestions ✅

### Should Pass (P1)
- Memory < 1.5GB
- Daily synthesis works
- Search races properly
- PII redaction complete

### Nice to Have (P2)
- Vector search accuracy > 0.8
- Synthesis finds patterns
- Chaos tests all pass

## 10. Validation Report Template

```markdown
# Symbiote Validation Report - [DATE]

## Summary
- Tests Run: X/Y
- Pass Rate: Z%
- Critical Issues: 0

## SLO Compliance
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Capture p99 | ≤200ms | XXXms | ✓/✗ |
| Search p50 | ≤100ms | XXXms | ✓/✗ |
| Memory | <1.5GB | X.XGB | ✓/✗ |

## Test Results
[Details of each test category]

## Issues Found
[List any failures with severity]

## Recommendation
[Go/No-Go decision with rationale]
```

## Next Steps

1. **Immediate**: Run validation suite phases 1-2
2. **Tomorrow**: Complete performance validation
3. **This Week**: Full user journey testing
4. **Document**: Create validation report
5. **Fix**: Address any P0 issues found