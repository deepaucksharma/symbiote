#!/usr/bin/env python3
"""
Retrieval evaluation harness for testing search quality.
Computes Recall@k, MRR, time-to-first-useful, and generates reports.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from statistics import mean, median, stdev
import click
import httpx
from tabulate import tabulate
from loguru import logger


class RetrievalEvaluator:
    """Evaluate retrieval quality against golden dataset."""
    
    def __init__(self, daemon_url: str = "http://localhost:8765"):
        self.daemon_url = daemon_url
        self.results = []
    
    async def evaluate_query(
        self,
        query: str,
        relevant_ids: List[str],
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.daemon_url}/context",
                    params={"q": query, "limit": max(k_values)},
                    timeout=5.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Query failed: {query}")
                    return None
                
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract result IDs
                result_ids = [r["id"] for r in data.get("results", [])]
                
                # Calculate metrics
                metrics = {
                    "query": query,
                    "relevant_count": len(relevant_ids),
                    "retrieved_count": len(result_ids),
                    "latency_ms": latency_ms,
                    "first_useful_ms": data.get("latency_ms", {}).get("first", latency_ms)
                }
                
                # Recall@k
                for k in k_values:
                    top_k = result_ids[:k]
                    hits = len(set(top_k) & set(relevant_ids))
                    recall = hits / len(relevant_ids) if relevant_ids else 0
                    metrics[f"recall@{k}"] = recall
                
                # MRR (Mean Reciprocal Rank)
                mrr = 0
                for i, result_id in enumerate(result_ids, 1):
                    if result_id in relevant_ids:
                        mrr = 1.0 / i
                        break
                metrics["mrr"] = mrr
                
                # Strategy latencies
                metrics["fts_ms"] = data.get("strategy_latencies", {}).get("fts", None)
                metrics["vector_ms"] = data.get("strategy_latencies", {}).get("vector", None)
                metrics["recents_ms"] = data.get("strategy_latencies", {}).get("recents", None)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return None
    
    async def evaluate_dataset(
        self,
        golden_path: Path,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate entire golden dataset."""
        queries = []
        
        # Load golden dataset
        with open(golden_path, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        
        logger.info(f"Evaluating {len(queries)} queries...")
        
        # Evaluate each query
        for i, query_data in enumerate(queries):
            result = await self.evaluate_query(
                query_data["query"],
                query_data["relevant"],
                k_values
            )
            
            if result:
                result["notes"] = query_data.get("notes", "")
                self.results.append(result)
            
            if (i + 1) % 5 == 0:
                logger.debug(f"  Evaluated {i + 1}/{len(queries)} queries")
        
        # Compute aggregate metrics
        return self.compute_aggregates(k_values)
    
    def compute_aggregates(self, k_values: List[int]) -> Dict[str, Any]:
        """Compute aggregate metrics across all queries."""
        if not self.results:
            return {}
        
        aggregates = {
            "total_queries": len(self.results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Recall@k aggregates
        for k in k_values:
            recalls = [r[f"recall@{k}"] for r in self.results if f"recall@{k}" in r]
            if recalls:
                aggregates[f"recall@{k}_mean"] = mean(recalls)
                aggregates[f"recall@{k}_median"] = median(recalls)
                aggregates[f"recall@{k}_min"] = min(recalls)
                aggregates[f"recall@{k}_max"] = max(recalls)
        
        # MRR aggregate
        mrrs = [r["mrr"] for r in self.results if "mrr" in r]
        if mrrs:
            aggregates["mrr_mean"] = mean(mrrs)
            aggregates["mrr_median"] = median(mrrs)
        
        # Latency aggregates
        latencies = [r["latency_ms"] for r in self.results if "latency_ms" in r]
        if latencies:
            aggregates["latency_p50"] = median(latencies)
            aggregates["latency_p95"] = sorted(latencies)[int(len(latencies) * 0.95)]
            aggregates["latency_mean"] = mean(latencies)
        
        first_useful = [r["first_useful_ms"] for r in self.results 
                       if "first_useful_ms" in r and r["first_useful_ms"]]
        if first_useful:
            aggregates["first_useful_p50"] = median(first_useful)
            aggregates["first_useful_p95"] = sorted(first_useful)[int(len(first_useful) * 0.95)]
        
        # Strategy win rates
        fts_wins = sum(1 for r in self.results 
                      if r.get("fts_ms") and 
                      (not r.get("vector_ms") or r["fts_ms"] < r["vector_ms"]))
        vector_wins = sum(1 for r in self.results 
                         if r.get("vector_ms") and 
                         (not r.get("fts_ms") or r["vector_ms"] < r["fts_ms"]))
        
        aggregates["fts_win_rate"] = fts_wins / len(self.results) if self.results else 0
        aggregates["vector_win_rate"] = vector_wins / len(self.results) if self.results else 0
        
        return aggregates
    
    def generate_report(self, aggregates: Dict[str, Any], output_path: Path) -> None:
        """Generate Markdown evaluation report."""
        report = []
        report.append("# Retrieval Evaluation Report")
        report.append(f"\n**Date:** {aggregates.get('timestamp', 'Unknown')}")
        report.append(f"**Total Queries:** {aggregates.get('total_queries', 0)}")
        report.append("")
        
        # Key metrics summary
        report.append("## Summary")
        report.append("")
        
        # Check if we meet SLO thresholds
        recall3 = aggregates.get("recall@3_mean", 0)
        mrr = aggregates.get("mrr_mean", 0)
        p50_latency = aggregates.get("first_useful_p50", 0)
        p95_latency = aggregates.get("first_useful_p95", 0)
        
        slo_pass = (
            recall3 >= 0.80 and
            mrr >= 0.70 and
            p50_latency <= 100 and
            p95_latency <= 300
        )
        
        status = "✅ **PASS**" if slo_pass else "❌ **FAIL**"
        report.append(f"**SLO Status:** {status}")
        report.append("")
        
        # Recall metrics table
        report.append("## Recall Metrics")
        report.append("")
        
        recall_data = []
        for k in [3, 5, 10]:
            if f"recall@{k}_mean" in aggregates:
                recall_data.append([
                    f"Recall@{k}",
                    f"{aggregates[f'recall@{k}_mean']:.3f}",
                    f"{aggregates[f'recall@{k}_median']:.3f}",
                    f"{aggregates[f'recall@{k}_min']:.3f}",
                    f"{aggregates[f'recall@{k}_max']:.3f}"
                ])
        
        if recall_data:
            report.append(tabulate(
                recall_data,
                headers=["Metric", "Mean", "Median", "Min", "Max"],
                tablefmt="github"
            ))
        report.append("")
        
        # MRR and latency metrics
        report.append("## Performance Metrics")
        report.append("")
        
        perf_data = []
        if "mrr_mean" in aggregates:
            perf_data.append(["MRR", f"{aggregates['mrr_mean']:.3f}", 
                            f"{aggregates['mrr_median']:.3f}"])
        if "latency_p50" in aggregates:
            perf_data.append(["Total Latency (p50)", f"{aggregates['latency_p50']:.1f} ms", ""])
            perf_data.append(["Total Latency (p95)", f"{aggregates['latency_p95']:.1f} ms", ""])
        if "first_useful_p50" in aggregates:
            perf_data.append(["First Useful (p50)", f"{aggregates['first_useful_p50']:.1f} ms", 
                            "🎯 Target: ≤100ms"])
            perf_data.append(["First Useful (p95)", f"{aggregates['first_useful_p95']:.1f} ms", 
                            "🎯 Target: ≤300ms"])
        
        if perf_data:
            report.append(tabulate(perf_data, headers=["Metric", "Value", "Target"], 
                                 tablefmt="github"))
        report.append("")
        
        # Strategy comparison
        report.append("## Strategy Performance")
        report.append("")
        report.append(f"- **FTS Win Rate:** {aggregates.get('fts_win_rate', 0):.1%}")
        report.append(f"- **Vector Win Rate:** {aggregates.get('vector_win_rate', 0):.1%}")
        report.append("")
        
        # Failed queries
        failed_queries = [r for r in self.results if r.get("mrr", 0) == 0]
        if failed_queries:
            report.append("## Failed Queries (MRR = 0)")
            report.append("")
            for q in failed_queries[:5]:  # Show top 5
                report.append(f"- \"{q['query']}\" - {q.get('notes', 'No notes')}")
            if len(failed_queries) > 5:
                report.append(f"- ... and {len(failed_queries) - 5} more")
            report.append("")
        
        # SLO compliance details
        report.append("## SLO Compliance")
        report.append("")
        report.append("| Requirement | Target | Actual | Status |")
        report.append("|------------|--------|--------|--------|")
        report.append(f"| Recall@3 | ≥0.80 | {recall3:.3f} | {'✅' if recall3 >= 0.80 else '❌'} |")
        report.append(f"| MRR | ≥0.70 | {mrr:.3f} | {'✅' if mrr >= 0.70 else '❌'} |")
        report.append(f"| p50 First Useful | ≤100ms | {p50_latency:.1f}ms | {'✅' if p50_latency <= 100 else '❌'} |")
        report.append(f"| p95 First Useful | ≤300ms | {p95_latency:.1f}ms | {'✅' if p95_latency <= 300 else '❌'} |")
        report.append("")
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
        
        logger.success(f"Report written to {output_path}")
        
        # Also save raw results
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "aggregates": aggregates,
                "queries": self.results
            }, f, indent=2)


@click.command()
@click.option("--golden", type=click.Path(exists=True), 
              default="tests/evals/golden.jsonl",
              help="Path to golden dataset")
@click.option("--out", type=click.Path(), 
              help="Output report path (default: tests/evals/reports/YYYY-MM-DD.md)")
@click.option("--daemon-url", default="http://localhost:8765",
              help="Daemon API URL")
@click.option("--k-values", default="3,5,10",
              help="Comma-separated k values for Recall@k")
def main(golden: str, out: str, daemon_url: str, k_values: str):
    """Run retrieval evaluation and generate report."""
    # Parse k values
    k_list = [int(k) for k in k_values.split(",")]
    
    # Determine output path
    if not out:
        out = Path("tests/evals/reports") / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    else:
        out = Path(out)
    
    logger.info(f"Running retrieval evaluation...")
    logger.info(f"Golden dataset: {golden}")
    logger.info(f"k values: {k_list}")
    
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
    
    # Run evaluation
    evaluator = RetrievalEvaluator(daemon_url)
    aggregates = asyncio.run(evaluator.evaluate_dataset(Path(golden), k_list))
    
    if not aggregates:
        logger.error("No results to report")
        return
    
    # Generate report
    evaluator.generate_report(aggregates, out)
    
    # Check if we meet thresholds
    if (aggregates.get("recall@3_mean", 0) >= 0.80 and
        aggregates.get("mrr_mean", 0) >= 0.70 and
        aggregates.get("first_useful_p50", float('inf')) <= 100 and
        aggregates.get("first_useful_p95", float('inf')) <= 300):
        logger.success("✅ All SLO thresholds met!")
    else:
        logger.warning("❌ Some SLO thresholds not met")


if __name__ == "__main__":
    main()