"""Metrics collection and observability."""

import time
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from loguru import logger


@dataclass
class LatencyHistogram:
    """Track latency distribution with percentiles."""
    name: str
    buckets: List[float] = field(default_factory=lambda: [
        10, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000  # milliseconds
    ])
    counts: Dict[float, int] = field(default_factory=dict)
    total_count: int = 0
    sum_ms: float = 0
    
    def __post_init__(self):
        for bucket in self.buckets:
            self.counts[bucket] = 0
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.total_count += 1
        self.sum_ms += latency_ms
        
        # Find appropriate bucket
        for bucket in self.buckets:
            if latency_ms <= bucket:
                self.counts[bucket] += 1
                break
        else:
            # Over max bucket
            self.counts[self.buckets[-1]] += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Get approximate percentile value."""
        if self.total_count == 0:
            return 0
        
        target_count = self.total_count * (percentile / 100)
        cumulative = 0
        
        for bucket in self.buckets:
            cumulative += self.counts[bucket]
            if cumulative >= target_count:
                return bucket
        
        return self.buckets[-1]
    
    def get_mean(self) -> float:
        """Get mean latency."""
        if self.total_count == 0:
            return 0
        return self.sum_ms / self.total_count
    
    def to_dict(self) -> Dict[str, any]:
        """Export metrics as dictionary."""
        if self.total_count == 0:
            return {
                "name": self.name,
                "count": 0,
                "mean": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }
        
        return {
            "name": self.name,
            "count": self.total_count,
            "mean": round(self.get_mean(), 1),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "buckets": {
                f"le_{bucket}": self.counts[bucket]
                for bucket in self.buckets
            }
        }


class MetricsCollector:
    """
    Central metrics collection for observability.
    Tracks latencies, counters, and resource usage.
    """
    
    def __init__(self):
        # Latency histograms
        self.histograms = {
            "capture": LatencyHistogram("capture"),
            "search.total": LatencyHistogram("search.total"),
            "search.fts": LatencyHistogram("search.fts"),
            "search.vector": LatencyHistogram("search.vector"),
            "search.recents": LatencyHistogram("search.recents"),
            "search.first_useful": LatencyHistogram("search.first_useful"),
            "suggestion": LatencyHistogram("suggestion"),
            "synthesis": LatencyHistogram("synthesis"),
        }
        
        # Counters
        self.counters = defaultdict(int)
        
        # Queue depth tracking
        self.queue_depths = defaultdict(lambda: deque(maxlen=100))
        
        # Resource usage
        self.resource_samples = deque(maxlen=60)  # Last 60 samples
        self.last_resource_check = 0
        
        # Acceptance/rejection rates
        self.suggestion_stats = {
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "follow_through": 0
        }
    
    def record_latency(self, metric_name: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        if metric_name in self.histograms:
            self.histograms[metric_name].record(latency_ms)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[counter_name] += amount
    
    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record current queue depth."""
        self.queue_depths[queue_name].append({
            "timestamp": datetime.utcnow(),
            "depth": depth
        })
    
    def record_suggestion_event(self, event_type: str) -> None:
        """Record suggestion acceptance/rejection."""
        if event_type in self.suggestion_stats:
            self.suggestion_stats[event_type] += 1
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Record resource usage sample."""
        self.resource_samples.append({
            "timestamp": datetime.utcnow(),
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent
        })
    
    def check_slos(self) -> Dict[str, bool]:
        """
        Check if we're meeting SLOs from Part 3.
        Returns dict of SLO name -> passing status.
        """
        slos = {}
        
        # Capture latency: p99 ≤ 200ms
        if self.histograms["capture"].total_count > 0:
            capture_p99 = self.histograms["capture"].get_percentile(99)
            slos["capture_p99_200ms"] = capture_p99 <= 200
        
        # Search first useful: p50 ≤ 100ms, p95 ≤ 300ms
        if self.histograms["search.first_useful"].total_count > 0:
            search_p50 = self.histograms["search.first_useful"].get_percentile(50)
            search_p95 = self.histograms["search.first_useful"].get_percentile(95)
            slos["search_p50_100ms"] = search_p50 <= 100
            slos["search_p95_300ms"] = search_p95 <= 300
        
        # Memory budget: < 1500MB
        if self.resource_samples:
            recent_memory = [s["memory_mb"] for s in list(self.resource_samples)[-10:]]
            avg_memory = sum(recent_memory) / len(recent_memory)
            slos["memory_under_1500mb"] = avg_memory < 1500
        
        return slos
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        if format == "json":
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "latencies": {
                    name: hist.to_dict()
                    for name, hist in self.histograms.items()
                },
                "counters": dict(self.counters),
                "suggestions": self.suggestion_stats,
                "slos": self.check_slos()
            }
            
            # Add queue depths (latest values)
            data["queue_depths"] = {}
            for queue_name, samples in self.queue_depths.items():
                if samples:
                    data["queue_depths"][queue_name] = samples[-1]["depth"]
            
            # Add resource usage (latest)
            if self.resource_samples:
                latest = self.resource_samples[-1]
                data["resources"] = {
                    "memory_mb": latest["memory_mb"],
                    "cpu_percent": latest["cpu_percent"]
                }
            
            return json.dumps(data, indent=2, default=str)
        
        elif format == "prometheus":
            # Prometheus text format
            lines = []
            lines.append(f"# HELP symbiote_info Symbiote daemon information")
            lines.append(f"# TYPE symbiote_info gauge")
            lines.append(f'symbiote_info{{version="0.1.0"}} 1')
            
            # Latency histograms
            for name, hist in self.histograms.items():
                metric_name = f"symbiote_{name.replace('.', '_')}_latency"
                lines.append(f"# HELP {metric_name}_ms Request latency in milliseconds")
                lines.append(f"# TYPE {metric_name}_ms histogram")
                
                cumulative = 0
                for bucket in hist.buckets:
                    cumulative += hist.counts[bucket]
                    lines.append(f'{metric_name}_ms_bucket{{le="{bucket}"}} {cumulative}')
                lines.append(f'{metric_name}_ms_bucket{{le="+Inf"}} {hist.total_count}')
                lines.append(f'{metric_name}_ms_sum {hist.sum_ms}')
                lines.append(f'{metric_name}_ms_count {hist.total_count}')
            
            # Counters
            for name, value in self.counters.items():
                metric_name = f"symbiote_{name.replace('.', '_')}_total"
                lines.append(f"# HELP {metric_name} Counter for {name}")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {value}")
            
            # Suggestion stats
            for stat, value in self.suggestion_stats.items():
                metric_name = f"symbiote_suggestion_{stat}_total"
                lines.append(f"# HELP {metric_name} Suggestion {stat} count")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {value}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        for hist in self.histograms.values():
            hist.counts = {bucket: 0 for bucket in hist.buckets}
            hist.total_count = 0
            hist.sum_ms = 0
        
        self.counters.clear()
        self.queue_depths.clear()
        self.resource_samples.clear()
        self.suggestion_stats = {
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "follow_through": 0
        }


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class LatencyTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time = None
        self.metrics = get_metrics()
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
            self.metrics.record_latency(self.metric_name, latency_ms)