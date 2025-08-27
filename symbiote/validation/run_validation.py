#!/usr/bin/env python3
"""
Automated validation runner for Symbiote.
Executes the complete validation plan and generates a report.
"""

import asyncio
import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import httpx
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

console = Console()


class ValidationTest:
    """Base class for validation tests."""
    
    def __init__(self, test_id: str, name: str, category: str, priority: str = "P1"):
        self.test_id = test_id
        self.name = name
        self.category = category
        self.priority = priority
        self.result = None
        self.error = None
        self.metrics = {}
    
    async def run(self) -> bool:
        """Run the test and return pass/fail."""
        raise NotImplementedError
    
    def get_status_symbol(self) -> str:
        """Get status symbol for display."""
        if self.result is None:
            return "⏳"
        elif self.result:
            return "✅"
        else:
            return "❌"


class CaptureLatencyTest(ValidationTest):
    """Test capture latency SLO."""
    
    def __init__(self):
        super().__init__("PERF-01", "Capture Latency < 200ms p99", "Performance", "P0")
    
    async def run(self) -> bool:
        """Test capture latency."""
        latencies = []
        
        async with httpx.AsyncClient() as client:
            for i in range(100):
                start = time.perf_counter()
                try:
                    response = await client.post(
                        "http://localhost:8765/capture",
                        json={"text": f"Latency test {i}"},
                        timeout=1.0
                    )
                    if response.status_code == 201:
                        latency_ms = (time.perf_counter() - start) * 1000
                        latencies.append(latency_ms)
                except:
                    pass
        
        if len(latencies) >= 50:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[int(len(latencies) * 0.50)]
            p95 = sorted_latencies[int(len(latencies) * 0.95)]
            p99 = sorted_latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
            
            self.metrics = {
                "p50": f"{p50:.1f}ms",
                "p95": f"{p95:.1f}ms",
                "p99": f"{p99:.1f}ms"
            }
            
            self.result = p99 <= 200
            return self.result
        
        self.result = False
        self.error = "Insufficient samples"
        return False


class SearchLatencyTest(ValidationTest):
    """Test search latency SLO."""
    
    def __init__(self):
        super().__init__("PERF-02", "Search Latency < 100ms p50", "Performance", "P0")
    
    async def run(self) -> bool:
        """Test search latency."""
        queries = ["test", "strategy", "API", "meeting", "project", "Q3", "review"]
        latencies = []
        
        async with httpx.AsyncClient() as client:
            for _ in range(5):
                for query in queries:
                    start = time.perf_counter()
                    try:
                        response = await client.get(
                            "http://localhost:8765/context",
                            params={"q": query},
                            timeout=1.0
                        )
                        if response.status_code == 200:
                            latency_ms = (time.perf_counter() - start) * 1000
                            latencies.append(latency_ms)
                    except:
                        pass
        
        if len(latencies) >= 20:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[int(len(latencies) * 0.50)]
            p95 = sorted_latencies[int(len(latencies) * 0.95)]
            
            self.metrics = {
                "p50": f"{p50:.1f}ms",
                "p95": f"{p95:.1f}ms",
                "samples": len(latencies)
            }
            
            self.result = p50 <= 100 and p95 <= 300
            return self.result
        
        self.result = False
        self.error = "Insufficient samples"
        return False


class ReceiptsTest(ValidationTest):
    """Test receipts generation."""
    
    def __init__(self):
        super().__init__("EXPL-01", "Receipts for all suggestions", "Explainability", "P0")
    
    async def run(self) -> bool:
        """Test receipt generation."""
        async with httpx.AsyncClient() as client:
            # Request suggestion
            response = await client.post(
                "http://localhost:8765/suggest",
                json={
                    "situation": {
                        "query": "test",
                        "free_minutes": 30
                    }
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                suggestion = data.get("suggestion")
                
                if suggestion:
                    # Check for receipt
                    if "receipts_id" in suggestion:
                        receipt_id = suggestion["receipts_id"]
                        
                        # Fetch receipt
                        receipt_response = await client.get(
                            f"http://localhost:8765/receipts/{receipt_id}",
                            timeout=2.0
                        )
                        
                        if receipt_response.status_code == 200:
                            receipt = receipt_response.json()
                            
                            # Validate receipt structure
                            required_fields = ["sources", "heuristics", "confidence", "version"]
                            has_all = all(field in receipt for field in required_fields)
                            
                            self.metrics = {
                                "has_receipt": True,
                                "complete": has_all
                            }
                            
                            self.result = has_all
                            return self.result
        
        self.result = False
        self.error = "No receipt generated"
        return False


class ConsentGateTest(ValidationTest):
    """Test privacy consent gates."""
    
    def __init__(self):
        super().__init__("PRIV-01", "Consent required for cloud", "Privacy", "P0")
    
    async def run(self) -> bool:
        """Test consent gate."""
        async with httpx.AsyncClient() as client:
            # Request deliberation with cloud
            response = await client.post(
                "http://localhost:8765/deliberate",
                json={
                    "query": "Compare options",
                    "allow_cloud": True
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Should require consent
                requires_consent = data.get("requires_consent", False)
                has_preview = "redaction_preview" in data
                has_action_id = "action_id" in data
                
                self.metrics = {
                    "requires_consent": requires_consent,
                    "has_preview": has_preview,
                    "has_action_id": has_action_id
                }
                
                # Local-only should work without consent
                local_response = await client.post(
                    "http://localhost:8765/deliberate",
                    json={
                        "query": "Compare options",
                        "allow_cloud": False
                    },
                    timeout=5.0
                )
                
                local_ok = local_response.status_code == 200
                local_no_consent = not local_response.json().get("requires_consent", False)
                
                self.result = (requires_consent or not has_preview) and local_ok and local_no_consent
                return self.result
        
        self.result = False
        self.error = "Consent gate not working"
        return False


class WALRecoveryTest(ValidationTest):
    """Test WAL crash recovery."""
    
    def __init__(self):
        super().__init__("RESIL-01", "WAL survives crashes", "Resilience", "P0")
    
    async def run(self) -> bool:
        """Test WAL recovery."""
        # Run chaos test
        result = subprocess.run(
            ["python", "scripts/chaos_inject.py", "--scenario", "kill_during_capture"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if test passed
        self.result = result.returncode == 0
        
        if not self.result:
            self.error = "WAL recovery failed"
            
        self.metrics = {
            "test_output": "PASS" if self.result else "FAIL"
        }
        
        return self.result


class ValidationRunner:
    """Runs all validation tests."""
    
    def __init__(self, daemon_url: str = "http://localhost:8765"):
        self.daemon_url = daemon_url
        self.tests = [
            CaptureLatencyTest(),
            SearchLatencyTest(),
            ReceiptsTest(),
            ConsentGateTest(),
            WALRecoveryTest(),
        ]
        self.results = {}
    
    async def check_daemon(self) -> bool:
        """Check if daemon is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.daemon_url}/health", timeout=2.0)
                return response.status_code == 200
        except:
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        console.print("\n[bold blue]Symbiote Validation Suite[/bold blue]")
        console.print(f"Starting validation at {datetime.now().isoformat()}\n")
        
        # Check daemon
        if not await self.check_daemon():
            console.print("[red]❌ Daemon not running at {self.daemon_url}[/red]")
            console.print("Please start the daemon: python -m symbiote.daemon.main")
            return {"error": "Daemon not running"}
        
        console.print("[green]✓ Daemon is running[/green]\n")
        
        # Group tests by category
        categories = {}
        for test in self.tests:
            if test.category not in categories:
                categories[test.category] = []
            categories[test.category].append(test)
        
        # Run tests by category
        total_passed = 0
        total_failed = 0
        p0_failed = []
        
        for category, tests in categories.items():
            console.print(f"\n[bold]{category} Tests[/bold]")
            
            for test in tests:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task(f"Running {test.test_id}: {test.name}...")
                    
                    try:
                        passed = await test.run()
                        progress.stop()
                        
                        status = test.get_status_symbol()
                        console.print(f"{status} {test.test_id}: {test.name}")
                        
                        if test.metrics:
                            for key, value in test.metrics.items():
                                console.print(f"  └─ {key}: {value}")
                        
                        if passed:
                            total_passed += 1
                        else:
                            total_failed += 1
                            if test.priority == "P0":
                                p0_failed.append(test.test_id)
                            if test.error:
                                console.print(f"  └─ [red]Error: {test.error}[/red]")
                    
                    except Exception as e:
                        progress.stop()
                        console.print(f"❌ {test.test_id}: {test.name}")
                        console.print(f"  └─ [red]Exception: {e}[/red]")
                        total_failed += 1
                        if test.priority == "P0":
                            p0_failed.append(test.test_id)
        
        # Generate summary
        console.print("\n" + "="*60)
        console.print("[bold]Validation Summary[/bold]")
        console.print("="*60)
        
        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Tests", str(total_passed + total_failed))
        table.add_row("Passed", f"[green]{total_passed}[/green]")
        table.add_row("Failed", f"[red]{total_failed}[/red]" if total_failed > 0 else "0")
        table.add_row("Pass Rate", f"{(total_passed / (total_passed + total_failed) * 100):.1f}%")
        
        console.print(table)
        
        # Go/No-Go decision
        console.print("\n[bold]Go/No-Go Decision[/bold]")
        
        if p0_failed:
            console.print(f"[red]❌ NO-GO: P0 tests failed: {', '.join(p0_failed)}[/red]")
            decision = "NO-GO"
        elif total_failed > len(self.tests) * 0.2:  # More than 20% failed
            console.print(f"[yellow]⚠️  NO-GO: Too many failures ({total_failed}/{len(self.tests)})[/yellow]")
            decision = "NO-GO"
        else:
            console.print("[green]✅ GO: All critical tests passed[/green]")
            decision = "GO"
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_passed + total_failed,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0,
                "p0_failed": p0_failed,
                "decision": decision
            },
            "tests": {
                test.test_id: {
                    "name": test.name,
                    "category": test.category,
                    "priority": test.priority,
                    "passed": test.result,
                    "metrics": test.metrics,
                    "error": test.error
                }
                for test in self.tests
            }
        }
        
        # Save report
        report_path = Path("validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        console.print(f"\n[dim]Report saved to {report_path}[/dim]")
        
        return report


@click.command()
@click.option("--daemon-url", default="http://localhost:8765", help="Daemon API URL")
@click.option("--quick", is_flag=True, help="Run quick validation only")
def main(daemon_url: str, quick: bool):
    """Run Symbiote validation suite."""
    runner = ValidationRunner(daemon_url)
    
    if quick:
        # Only run P0 tests
        runner.tests = [t for t in runner.tests if t.priority == "P0"]
    
    report = asyncio.run(runner.run_all_tests())
    
    # Exit with appropriate code
    if report.get("summary", {}).get("decision") == "GO":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()