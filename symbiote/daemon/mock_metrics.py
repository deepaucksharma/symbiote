"""Mock metrics implementation for demo purposes."""

import time
from typing import Dict, Any
from collections import defaultdict, deque
from contextlib import contextmanager


class MockHistogram:
    """Mock histogram that tracks latencies."""
    
    def __init__(self, name: str):
        self.name = name
        self.values = deque(maxlen=1000)  # Keep last 1000 values
    
    def record(self, value: float):
        """Record a value."""
        self.values.append(value)
    
    def get_percentile(self, percentile: int) -> float:
        """Get percentile value."""
        if not self.values:
            return 0.0
        
        sorted_values = sorted(self.values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0}
        
        values = list(self.values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99)
        }


class MockMetrics:
    """Mock metrics collector."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.histograms = {
            "capture": MockHistogram("capture"),
            "search": MockHistogram("search"),
            "search.first_useful": MockHistogram("search.first_useful"),
            "suggestion": MockHistogram("suggestion")
        }
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] += value
    
    def record_latency(self, name: str, latency_ms: float):
        """Record latency in histogram."""
        if name in self.histograms:
            self.histograms[name].record(latency_ms)
    
    def record_suggestion_event(self, event_type: str):
        """Record suggestion event."""
        self.increment_counter(f"suggestion.{event_type}")
    
    def check_slos(self) -> Dict[str, bool]:
        """Check SLO compliance."""
        slos = {}
        
        # Capture SLO: p99 <= 200ms
        capture_p99 = self.histograms["capture"].get_percentile(99)
        slos["capture_p99"] = capture_p99 <= 200
        
        # Search SLO: p50 <= 100ms
        search_p50 = self.histograms["search.first_useful"].get_percentile(50)
        slos["search_p50"] = search_p50 <= 100
        
        return slos
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        if format_type == "json":
            import json
            data = {
                "counters": dict(self.counters),
                "latencies": {
                    name: hist.get_stats()
                    for name, hist in self.histograms.items()
                }
            }
            return json.dumps(data, indent=2)
        else:
            # Simple text format
            lines = []
            for name, value in self.counters.items():
                lines.append(f"# HELP {name} Counter\n")
                lines.append(f"# TYPE {name} counter\n")
                lines.append(f"{name} {value}\n")
            return "\n".join(lines)


class LatencyTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
            get_metrics().record_latency(self.operation, latency_ms)


# Global metrics instance
_metrics = None


def get_metrics() -> MockMetrics:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MockMetrics()
    return _metrics