"""
End-to-end integration tests for Symbiote.
Tests complete flows from capture to search to suggestions with receipts.
"""

import asyncio
import json
import tempfile
from pathlib import Path
import pytest
import httpx
from datetime import datetime

from symbiote.daemon.main import SymbioteDaemon
from symbiote.daemon.config import Config


@pytest.fixture
async def test_daemon():
    """Create a test daemon instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir) / "test_vault"
        vault_path.mkdir()
        
        config = Config(vault_path=vault_path)
        daemon = SymbioteDaemon(config)
        
        await daemon.start()
        yield daemon
        await daemon.stop()


@pytest.fixture
def daemon_url():
    """Test daemon URL."""
    return "http://localhost:8765"


class TestCaptureToSearchFlow:
    """Test capture → index → search flow."""
    
    @pytest.mark.asyncio
    async def test_capture_appears_in_search(self, test_daemon, daemon_url):
        """Test that captured content is searchable."""
        async with httpx.AsyncClient() as client:
            # Capture a note
            capture_response = await client.post(
                f"{daemon_url}/capture",
                json={
                    "text": "Important Q3 strategy meeting notes about API redesign",
                    "type_hint": "note"
                }
            )
            
            assert capture_response.status_code == 201
            capture_data = capture_response.json()
            capture_id = capture_data["id"]
            
            # Wait for indexing
            await asyncio.sleep(0.5)
            
            # Search for the content
            search_response = await client.get(
                f"{daemon_url}/context",
                params={"q": "Q3 strategy API"}
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            # Verify the captured note appears in results
            results = search_data.get("results", [])
            assert len(results) > 0
            
            # Should find our note
            found = any(capture_id in r.get("id", "") for r in results)
            assert found, "Captured note not found in search results"
    
    @pytest.mark.asyncio
    async def test_capture_with_receipts(self, test_daemon, daemon_url):
        """Test that suggestions include receipts."""
        async with httpx.AsyncClient() as client:
            # Capture multiple related items
            for i in range(3):
                await client.post(
                    f"{daemon_url}/capture",
                    json={
                        "text": f"Task {i}: Implement feature for Q3 roadmap",
                        "type_hint": "task"
                    }
                )
            
            await asyncio.sleep(0.5)
            
            # Request suggestions
            suggest_response = await client.post(
                f"{daemon_url}/suggest",
                json={
                    "situation": {
                        "query": "Q3 roadmap",
                        "free_minutes": 30
                    }
                }
            )
            
            assert suggest_response.status_code == 200
            suggest_data = suggest_response.json()
            
            suggestion = suggest_data.get("suggestion")
            if suggestion:
                # Must have receipts_id
                assert "receipts_id" in suggestion
                assert suggestion["receipts_id"] is not None
                
                # Fetch the receipt
                receipt_id = suggestion["receipts_id"]
                receipt_response = await client.get(
                    f"{daemon_url}/receipts/{receipt_id}"
                )
                
                if receipt_response.status_code == 200:
                    receipt = receipt_response.json()
                    
                    # Verify receipt structure
                    assert "sources" in receipt
                    assert "heuristics" in receipt
                    assert "confidence" in receipt
                    assert "version" in receipt


class TestSearchRacing:
    """Test search racing behavior."""
    
    @pytest.mark.asyncio
    async def test_early_return(self, test_daemon, daemon_url):
        """Test that search returns early with first useful result."""
        async with httpx.AsyncClient() as client:
            # Perform search
            search_response = await client.get(
                f"{daemon_url}/context",
                params={"q": "test query"}
            )
            
            assert search_response.status_code == 200
            data = search_response.json()
            
            # Check latency tracking
            assert "latency_ms" in data or "strategy_latencies" in data
            
            # If we have strategy latencies, verify racing
            if "strategy_latencies" in data:
                latencies = data["strategy_latencies"]
                
                # At least one strategy should have returned
                assert any(latencies.values())
                
                # First useful should be faster than total
                if "first" in data.get("latency_ms", {}):
                    first_useful = data["latency_ms"]["first"]
                    total = max(v for v in latencies.values() if v)
                    
                    # First useful should be <= slowest strategy
                    assert first_useful <= total


class TestConsentFlow:
    """Test consent and redaction flow."""
    
    @pytest.mark.asyncio
    async def test_deliberate_requires_consent(self, test_daemon, daemon_url):
        """Test that deliberation with cloud requires consent."""
        async with httpx.AsyncClient() as client:
            # Request deliberation with cloud
            deliberate_response = await client.post(
                f"{daemon_url}/deliberate",
                json={
                    "query": "Compare options for team expansion",
                    "scope": {"project": "hiring"},
                    "allow_cloud": True
                }
            )
            
            assert deliberate_response.status_code == 200
            data = deliberate_response.json()
            
            # Should require consent
            if data.get("requires_consent"):
                assert "redaction_preview" in data
                assert "action_id" in data
                
                # Verify preview is sanitized
                preview = data.get("redaction_preview", [])
                for item in preview:
                    # Should not contain raw email addresses
                    assert "@" not in item or "[EMAIL_REDACTED]" in item
    
    @pytest.mark.asyncio
    async def test_local_only_no_consent(self, test_daemon, daemon_url):
        """Test that local-only operations don't require consent."""
        async with httpx.AsyncClient() as client:
            # Request deliberation without cloud
            deliberate_response = await client.post(
                f"{daemon_url}/deliberate",
                json={
                    "query": "Plan next sprint",
                    "scope": {},
                    "allow_cloud": False
                }
            )
            
            assert deliberate_response.status_code == 200
            data = deliberate_response.json()
            
            # Should not require consent
            assert not data.get("requires_consent", False)
            assert "plan" in data


class TestSLOCompliance:
    """Test that SLOs are met."""
    
    @pytest.mark.asyncio
    async def test_capture_latency_slo(self, test_daemon, daemon_url):
        """Test capture meets p99 ≤ 200ms."""
        latencies = []
        
        async with httpx.AsyncClient() as client:
            # Run multiple captures
            for i in range(20):
                start = asyncio.get_event_loop().time()
                
                response = await client.post(
                    f"{daemon_url}/capture",
                    json={"text": f"Test capture {i}"}
                )
                
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                
                if response.status_code == 201:
                    latencies.append(latency_ms)
        
        if latencies:
            # Check p99 (or max for small sample)
            sorted_latencies = sorted(latencies)
            p99_index = int(len(sorted_latencies) * 0.99)
            p99 = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
            
            # Should meet SLO
            assert p99 <= 200, f"Capture p99 {p99:.1f}ms exceeds 200ms SLO"
    
    @pytest.mark.asyncio
    async def test_search_latency_slo(self, test_daemon, daemon_url):
        """Test search meets p50 ≤ 100ms, p95 ≤ 300ms."""
        first_useful_latencies = []
        
        async with httpx.AsyncClient() as client:
            queries = ["test", "strategy", "API", "meeting", "bug"]
            
            for query in queries * 4:  # 20 searches
                response = await client.get(
                    f"{daemon_url}/context",
                    params={"q": query}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "latency_ms" in data and "first" in data["latency_ms"]:
                        first_useful_latencies.append(data["latency_ms"]["first"])
        
        if first_useful_latencies:
            sorted_latencies = sorted(first_useful_latencies)
            n = len(sorted_latencies)
            
            p50 = sorted_latencies[int(n * 0.50)]
            p95 = sorted_latencies[min(int(n * 0.95), n - 1)]
            
            # Should meet SLOs
            assert p50 <= 100, f"Search p50 {p50:.1f}ms exceeds 100ms SLO"
            assert p95 <= 300, f"Search p95 {p95:.1f}ms exceeds 300ms SLO"


class TestReindexRecovery:
    """Test reindex and recovery flows."""
    
    @pytest.mark.asyncio
    async def test_reindex_maintains_search(self, test_daemon, daemon_url):
        """Test search works during reindex."""
        async with httpx.AsyncClient() as client:
            # Trigger reindex
            reindex_response = await client.post(
                f"{daemon_url}/admin/reindex",
                json={"scope": "all"}
            )
            
            assert reindex_response.status_code == 202
            
            # Search should still work (degraded to recents)
            search_response = await client.get(
                f"{daemon_url}/context",
                params={"q": "test"}
            )
            
            assert search_response.status_code == 200
            data = search_response.json()
            
            # Should have some results (even if just recents)
            assert "results" in data


class TestHealthAndMetrics:
    """Test health checks and metrics."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, test_daemon, daemon_url):
        """Test health endpoint returns SLO status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{daemon_url}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "slos" in data or "latencies" in data
    
    @pytest.mark.asyncio
    async def test_metrics_export(self, test_daemon, daemon_url):
        """Test metrics can be exported."""
        async with httpx.AsyncClient() as client:
            # JSON format
            json_response = await client.get(
                f"{daemon_url}/metrics",
                params={"format": "json"}
            )
            
            assert json_response.status_code == 200
            json_data = json_response.json()
            assert "latencies" in json_data or "counters" in json_data
            
            # Prometheus format
            prom_response = await client.get(
                f"{daemon_url}/metrics",
                params={"format": "prometheus"}
            )
            
            assert prom_response.status_code == 200
            prom_text = prom_response.text
            assert "# HELP" in prom_text
            assert "# TYPE" in prom_text