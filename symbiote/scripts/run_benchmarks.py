#!/usr/bin/env python3
"""
Performance benchmarking suite for Symbiote.
Tests capture latency, search performance, and resource usage against synthetic vaults.
"""

import os
import time
import asyncio
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from statistics import mean, median, stdev
from datetime import datetime
import click
import httpx
from tabulate import tabulate
from loguru import logger


class PerformanceBenchmark:
    """Run performance benchmarks against the daemon."""
    
    def __init__(self, daemon_url: str = "http://localhost:8765"):
        self.daemon_url = daemon_url
        self.process = psutil.Process()
        
    async def benchmark_capture(
        self,
        num_captures: int = 100,
        warmup: int = 5
    ) -> Dict[str, Any]:
        """Benchmark capture latency."""
        logger.info(f"Benchmarking capture with {num_captures} iterations...")
        
        latencies = []
        errors = 0
        
        async with httpx.AsyncClient() as client:
            # Warmup
            for i in range(warmup):
                await client.post(
                    f"{self.daemon_url}/capture",
                    json={"text": f"Warmup capture {i}"},
                    timeout=5.0
                )
            
            # Actual benchmark
            for i in range(num_captures):
                text = f"Benchmark capture {i}: {' '.join(['word'] * 50)}"
                
                start = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.daemon_url}/capture",
                        json={"text": text, "type_hint": "note"},
                        timeout=5.0
                    )
                    
                    if response.status_code == 201:
                        latency_ms = (time.perf_counter() - start) * 1000
                        latencies.append(latency_ms)
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    logger.debug(f"Capture error: {e}")
                
                if (i + 1) % 20 == 0:
                    logger.debug(f"  Completed {i + 1}/{num_captures} captures")
        
        if not latencies:
            return {"error": "No successful captures"}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "operation": "capture",
            "iterations": num_captures,
            "successful": len(latencies),
            "errors": errors,
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            "mean": mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "stdev": stdev(latencies) if len(latencies) > 1 else 0
        }
    
    async def benchmark_search(
        self,
        queries: List[str],
        warmup: int = 5
    ) -> Dict[str, Any]:
        """Benchmark search latency and racing behavior."""
        logger.info(f"Benchmarking search with {len(queries)} queries...")
        
        total_latencies = []
        first_useful_latencies = []
        fts_latencies = []
        vector_latencies = []
        recents_latencies = []
        errors = 0
        
        async with httpx.AsyncClient() as client:
            # Warmup
            for i in range(min(warmup, len(queries))):
                await client.get(
                    f"{self.daemon_url}/context",
                    params={"q": queries[i % len(queries)]},
                    timeout=5.0
                )
            
            # Actual benchmark
            for i, query in enumerate(queries):
                start = time.perf_counter()
                
                try:
                    response = await client.get(
                        f"{self.daemon_url}/context",
                        params={"q": query, "limit": 10},
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        total_ms = (time.perf_counter() - start) * 1000
                        total_latencies.append(total_ms)
                        
                        data = response.json()
                        
                        # Extract strategy latencies
                        if "latency_ms" in data:
                            if "first" in data["latency_ms"]:
                                first_useful_latencies.append(data["latency_ms"]["first"])
                        
                        if "strategy_latencies" in data:
                            strat = data["strategy_latencies"]
                            if "fts" in strat and strat["fts"]:
                                fts_latencies.append(strat["fts"])
                            if "vector" in strat and strat["vector"]:
                                vector_latencies.append(strat["vector"])
                            if "recents" in strat and strat["recents"]:
                                recents_latencies.append(strat["recents"])
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    logger.debug(f"Search error: {e}")
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Completed {i + 1}/{len(queries)} searches")
        
        results = {
            "operation": "search",
            "iterations": len(queries),
            "successful": len(total_latencies),
            "errors": errors
        }
        
        # Add percentiles for each metric
        for name, latencies in [
            ("total", total_latencies),
            ("first_useful", first_useful_latencies),
            ("fts", fts_latencies),
            ("vector", vector_latencies),
            ("recents", recents_latencies)
        ]:
            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                results[f"{name}_p50"] = sorted_lat[int(n * 0.50)]
                results[f"{name}_p95"] = sorted_lat[int(n * 0.95)]
                results[f"{name}_p99"] = sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1]
                results[f"{name}_mean"] = mean(latencies)
        
        return results
    
    async def benchmark_reindex(self, vault_path: Path) -> Dict[str, Any]:
        """Benchmark index rebuild time."""
        logger.info("Benchmarking index rebuild...")
        
        start = time.perf_counter()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.daemon_url}/admin/reindex",
                    json={"scope": "all"},
                    timeout=600.0  # 10 minute timeout
                )
                
                if response.status_code == 202:
                    # Poll for completion
                    while True:
                        await asyncio.sleep(1)
                        status_response = await client.get(
                            f"{self.daemon_url}/status",
                            timeout=5.0
                        )
                        
                        # Check if reindexing is complete
                        # (This would need a proper status endpoint)
                        elapsed = time.perf_counter() - start
                        if elapsed > 600:  # Max 10 minutes
                            break
                        
                        # For now, just wait a reasonable time based on vault size
                        vault_size = sum(1 for _ in vault_path.rglob("*.md"))
                        expected_time = vault_size * 0.01  # 10ms per file estimate
                        if elapsed > max(expected_time, 10):
                            break
                    
                    rebuild_time = time.perf_counter() - start
                    
                    return {
                        "operation": "reindex",
                        "vault_size": sum(1 for _ in vault_path.rglob("*.md")),
                        "time_seconds": rebuild_time,
                        "files_per_second": sum(1 for _ in vault_path.rglob("*.md")) / rebuild_time
                    }
                    
        except Exception as e:
            logger.error(f"Reindex benchmark failed: {e}")
            return {"error": str(e)}
    
    def measure_resources(self) -> Dict[str, Any]:
        """Measure current resource usage."""
        try:
            memory_info = self.process.memory_info()
            return {
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": self.process.memory_percent(),
                "cpu_percent": self.process.cpu_percent(interval=1),
                "num_threads": self.process.num_threads()
            }
        except Exception as e:
            logger.error(f"Resource measurement failed: {e}")
            return {}
    
    async def run_full_benchmark(
        self,
        vault_path: Path,
        capture_iterations: int = 100,
        search_queries: List[str] = None
    ) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "vault_path": str(vault_path),
            "vault_stats": {}
        }
        
        # Count vault files
        notes = list(vault_path.glob("notes/*.md"))
        tasks = list(vault_path.glob("tasks/*.md"))
        results["vault_stats"] = {
            "notes": len(notes),
            "tasks": len(tasks),
            "total": len(notes) + len(tasks)
        }
        
        # Resource usage before
        results["resources_before"] = self.measure_resources()
        
        # Capture benchmark
        logger.info("Running capture benchmark...")
        results["capture"] = await self.benchmark_capture(capture_iterations)
        
        # Search benchmark
        if not search_queries:
            # Generate default queries
            search_queries = [
                "strategy", "implementation", "bug fix", "performance",
                "meeting notes", "architecture", "API design", "testing",
                "deployment", "security", "documentation", "review"
            ] * 5  # 60 queries total
        
        logger.info("Running search benchmark...")
        results["search"] = await self.benchmark_search(search_queries)
        
        # Resource usage after
        results["resources_after"] = self.measure_resources()
        
        # Memory delta
        if "memory_mb" in results["resources_before"] and "memory_mb" in results["resources_after"]:
            results["memory_delta_mb"] = (
                results["resources_after"]["memory_mb"] - 
                results["resources_before"]["memory_mb"]
            )
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate benchmark report."""
        report = []
        report.append("# Performance Benchmark Report")
        report.append(f"\n**Date:** {results['timestamp']}")
        report.append(f"**Vault:** {results['vault_path']}")
        report.append(f"**Vault Size:** {results['vault_stats']['total']} files "
                     f"({results['vault_stats']['notes']} notes, "
                     f"{results['vault_stats']['tasks']} tasks)")
        report.append("")
        
        # Check SLO compliance
        slo_pass = True
        
        # Capture performance
        if "capture" in results and "error" not in results["capture"]:
            cap = results["capture"]
            report.append("## Capture Performance")
            report.append("")
            
            cap_data = [
                ["Metric", "Value", "SLO", "Status"],
                ["p50", f"{cap['p50']:.1f} ms", "-", ""],
                ["p95", f"{cap['p95']:.1f} ms", "-", ""],
                ["p99", f"{cap['p99']:.1f} ms", "≤200 ms", 
                 "✅" if cap['p99'] <= 200 else "❌"],
                ["Mean", f"{cap['mean']:.1f} ms", "-", ""],
                ["Min", f"{cap['min']:.1f} ms", "-", ""],
                ["Max", f"{cap['max']:.1f} ms", "-", ""],
                ["Success Rate", f"{cap['successful']}/{cap['iterations']}", "-", ""]
            ]
            
            report.append(tabulate(cap_data, headers="firstrow", tablefmt="github"))
            report.append("")
            
            if cap['p99'] > 200:
                slo_pass = False
        
        # Search performance
        if "search" in results and "error" not in results["search"]:
            search = results["search"]
            report.append("## Search Performance")
            report.append("")
            
            # First useful latency (the key metric)
            if "first_useful_p50" in search:
                search_data = [
                    ["Metric", "Value", "SLO", "Status"],
                    ["First Useful p50", f"{search['first_useful_p50']:.1f} ms", 
                     "≤100 ms", "✅" if search['first_useful_p50'] <= 100 else "❌"],
                    ["First Useful p95", f"{search['first_useful_p95']:.1f} ms", 
                     "≤300 ms", "✅" if search['first_useful_p95'] <= 300 else "❌"],
                    ["Total p50", f"{search.get('total_p50', 0):.1f} ms", "-", ""],
                    ["Total p95", f"{search.get('total_p95', 0):.1f} ms", "-", ""],
                ]
                
                report.append(tabulate(search_data, headers="firstrow", tablefmt="github"))
                report.append("")
                
                if search['first_useful_p50'] > 100 or search['first_useful_p95'] > 300:
                    slo_pass = False
            
            # Strategy breakdown
            report.append("### Strategy Latencies")
            report.append("")
            
            strat_data = [["Strategy", "p50", "p95", "Mean"]]
            for strategy in ["fts", "vector", "recents"]:
                if f"{strategy}_p50" in search:
                    strat_data.append([
                        strategy.upper(),
                        f"{search[f'{strategy}_p50']:.1f} ms",
                        f"{search[f'{strategy}_p95']:.1f} ms",
                        f"{search[f'{strategy}_mean']:.1f} ms"
                    ])
            
            if len(strat_data) > 1:
                report.append(tabulate(strat_data, headers="firstrow", tablefmt="github"))
                report.append("")
        
        # Resource usage
        report.append("## Resource Usage")
        report.append("")
        
        if "resources_after" in results:
            res = results["resources_after"]
            res_data = [
                ["Metric", "Value", "SLO", "Status"],
                ["Memory", f"{res.get('memory_mb', 0):.1f} MB", 
                 "<1500 MB", "✅" if res.get('memory_mb', 0) < 1500 else "❌"],
                ["CPU", f"{res.get('cpu_percent', 0):.1f}%", "-", ""],
                ["Threads", f"{res.get('num_threads', 0)}", "-", ""]
            ]
            
            if "memory_delta_mb" in results:
                res_data.append(["Memory Delta", f"{results['memory_delta_mb']:.1f} MB", "-", ""])
            
            report.append(tabulate(res_data, headers="firstrow", tablefmt="github"))
            report.append("")
            
            if res.get('memory_mb', 0) >= 1500:
                slo_pass = False
        
        # Overall result
        report.append("## Summary")
        report.append("")
        report.append(f"**SLO Compliance:** {'✅ PASS' if slo_pass else '❌ FAIL'}")
        report.append("")
        
        if not slo_pass:
            report.append("### Failed SLOs")
            report.append("")
            if "capture" in results and results["capture"].get("p99", 0) > 200:
                report.append(f"- Capture p99: {results['capture']['p99']:.1f}ms > 200ms")
            if "search" in results:
                if results["search"].get("first_useful_p50", 0) > 100:
                    report.append(f"- Search p50: {results['search']['first_useful_p50']:.1f}ms > 100ms")
                if results["search"].get("first_useful_p95", 0) > 300:
                    report.append(f"- Search p95: {results['search']['first_useful_p95']:.1f}ms > 300ms")
            if results.get("resources_after", {}).get("memory_mb", 0) >= 1500:
                report.append(f"- Memory: {results['resources_after']['memory_mb']:.1f}MB >= 1500MB")
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
        
        logger.success(f"Report written to {output_path}")
        
        # Also save raw results
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


@click.command()
@click.option("--vault", type=click.Path(exists=True), required=True,
              help="Path to vault for benchmarking")
@click.option("--captures", default=100, help="Number of capture iterations")
@click.option("--out", type=click.Path(),
              help="Output report path (default: tests/benchmarks/YYYY-MM-DD.md)")
@click.option("--daemon-url", default="http://localhost:8765",
              help="Daemon API URL")
def main(vault: str, captures: int, out: str, daemon_url: str):
    """Run performance benchmarks."""
    vault_path = Path(vault)
    
    # Determine output path
    if not out:
        out = Path("tests/benchmarks") / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    else:
        out = Path(out)
    
    logger.info(f"Running performance benchmarks...")
    logger.info(f"Vault: {vault_path}")
    logger.info(f"Capture iterations: {captures}")
    
    # Check daemon is running
    try:
        response = httpx.get(f"{daemon_url}/health", timeout=2.0)
        if response.status_code != 200:
            logger.error("Daemon health check failed")
            return
    except Exception as e:
        logger.error(f"Cannot connect to daemon at {daemon_url}: {e}")
        logger.info("Start the daemon with: sym daemon start")
        return
    
    # Set CPU governor if possible (Linux)
    if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"):
        logger.info("Note: For best results, set CPU governor to 'performance':")
        logger.info("  sudo cpupower frequency-set -g performance")
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(daemon_url)
    results = asyncio.run(benchmark.run_full_benchmark(vault_path, captures))
    
    # Generate report
    benchmark.generate_report(results, out)
    
    # Check SLO compliance
    slo_pass = True
    if "capture" in results and results["capture"].get("p99", 0) > 200:
        slo_pass = False
    if "search" in results:
        if results["search"].get("first_useful_p50", 0) > 100:
            slo_pass = False
        if results["search"].get("first_useful_p95", 0) > 300:
            slo_pass = False
    if results.get("resources_after", {}).get("memory_mb", 0) >= 1500:
        slo_pass = False
    
    if slo_pass:
        logger.success("✅ All performance SLOs met!")
    else:
        logger.warning("❌ Some performance SLOs not met")


if __name__ == "__main__":
    main()