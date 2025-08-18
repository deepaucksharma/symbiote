"""Search orchestrator with racing strategy for fast context assembly."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
from loguru import logger

from .config import Config
from .bus import Event, get_event_bus


class SearchStrategy(Enum):
    """Available search strategies."""
    FTS = "fts"
    VECTOR = "vector"
    RECENTS = "recents"
    ANALYTICS = "analytics"


@dataclass
class SearchResult:
    """Individual search result."""
    id: str
    path: str
    title: str
    snippet: str
    score: float
    source: SearchStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextCard:
    """
    Context card returned to user.
    Contains top items, quick actions, and optional suggestions.
    """
    query: str
    results: List[SearchResult]
    quick_actions: List[Dict[str, Any]]
    suggestions: Optional[List[str]] = None
    latency_ms: float = 0
    strategy_latencies: Dict[str, float] = field(default_factory=dict)
    first_useful_strategy: Optional[SearchStrategy] = None


class SearchOrchestrator:
    """
    Orchestrates parallel searches across FTS, Vector, and Recents.
    Returns first useful result, then enriches with late arrivals.
    
    Performance targets:
    - First useful result: p50 ≤ 100ms, p95 ≤ 300ms
    - Complete enrichment: up to 1s
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._event_bus = get_event_bus()
        
        # Search workers (will be initialized with actual implementations)
        self._fts_worker = None
        self._vector_worker = None
        self._analytics_worker = None
        
        # Caching
        self._recents_cache: List[SearchResult] = []
        self._cache_updated = datetime.utcnow()
        
        # Thresholds for "useful" results
        self.usefulness_threshold = {
            SearchStrategy.FTS: 0.5,
            SearchStrategy.VECTOR: 0.7,
            SearchStrategy.RECENTS: 0.3,
            SearchStrategy.ANALYTICS: 0.4
        }
        
        # Performance tracking
        self._latency_histogram = []
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        context_hint: Optional[str] = None,
        max_results: int = 10,
        timeout_ms: int = 1000
    ) -> ContextCard:
        """
        Execute racing search strategy.
        
        Returns first useful results immediately, enriches up to timeout.
        """
        start_time = time.perf_counter()
        
        # Create context card
        card = ContextCard(
            query=query,
            results=[],
            quick_actions=[]
        )
        
        # Launch parallel searches
        search_tasks = []
        
        # Always include recents (usually fastest)
        search_tasks.append(
            self._create_search_task(
                SearchStrategy.RECENTS,
                self._search_recents(query, project_hint, max_results)
            )
        )
        
        # FTS if enabled
        if self.config.indices.fts and self._fts_worker:
            search_tasks.append(
                self._create_search_task(
                    SearchStrategy.FTS,
                    self._search_fts(query, project_hint, max_results)
                )
            )
        
        # Vector if enabled
        if self.config.indices.vector and self._vector_worker:
            search_tasks.append(
                self._create_search_task(
                    SearchStrategy.VECTOR,
                    self._search_vector(query, project_hint, max_results)
                )
            )
        
        # Analytics if enabled
        if self.config.indices.analytics and self._analytics_worker:
            search_tasks.append(
                self._create_search_task(
                    SearchStrategy.ANALYTICS,
                    self._search_analytics(query, project_hint, max_results)
                )
            )
        
        # Race for first useful result
        first_useful_found = False
        results_by_strategy = {}
        timeout = timeout_ms / 1000.0
        deadline = asyncio.get_event_loop().time() + timeout
        
        # Process results as they arrive
        pending = set(search_tasks)
        while pending and asyncio.get_event_loop().time() < deadline:
            # Wait for next result with remaining timeout
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=remaining
            )
            
            for task in done:
                try:
                    strategy, results, latency = await task
                    card.strategy_latencies[strategy.value] = latency
                    results_by_strategy[strategy] = results
                    
                    # Check if results are useful
                    if not first_useful_found and self._is_useful(
                        results, strategy, query, project_hint
                    ):
                        # First useful result found!
                        first_useful_found = True
                        card.first_useful_strategy = strategy
                        card.results = results[:max_results]
                        card.quick_actions = self._generate_quick_actions(results, query)
                        
                        # Record early return latency
                        early_latency = (time.perf_counter() - start_time) * 1000
                        card.latency_ms = early_latency
                        
                        logger.debug(
                            f"First useful result from {strategy.value} in {early_latency:.1f}ms"
                        )
                        
                        # Emit early result event
                        await self._event_bus.emit(Event(
                            type="search.early_result",
                            data={
                                "query": query,
                                "strategy": strategy.value,
                                "latency_ms": early_latency,
                                "result_count": len(results)
                            }
                        ))
                
                except Exception as e:
                    logger.error(f"Search task error: {e}")
        
        # Cancel remaining tasks if we hit timeout
        for task in pending:
            task.cancel()
        
        # If no useful result found yet, use best available
        if not first_useful_found and results_by_strategy:
            # Merge all results with scoring
            merged = self._merge_results(results_by_strategy, max_results)
            card.results = merged
            card.quick_actions = self._generate_quick_actions(merged, query)
            card.latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Generate suggestions if we have results
        if card.results:
            card.suggestions = await self._generate_suggestions(
                card.results, query, context_hint
            )
        
        # Track performance
        self._latency_histogram.append(card.latency_ms)
        if len(self._latency_histogram) > 1000:
            self._latency_histogram = self._latency_histogram[-1000:]
        
        # Emit final event
        await self._event_bus.emit(Event(
            type="search.completed",
            data={
                "query": query,
                "latency_ms": card.latency_ms,
                "result_count": len(card.results),
                "strategies_used": list(card.strategy_latencies.keys())
            }
        ))
        
        return card
    
    async def _create_search_task(self, strategy: SearchStrategy, search_coro):
        """Wrapper to track strategy and timing."""
        start = time.perf_counter()
        try:
            results = await search_coro
            latency = (time.perf_counter() - start) * 1000
            return strategy, results, latency
        except Exception as e:
            logger.error(f"Search strategy {strategy.value} failed: {e}")
            return strategy, [], 0
    
    async def _search_recents(
        self, 
        query: str, 
        project_hint: Optional[str],
        max_results: int
    ) -> List[SearchResult]:
        """
        Search recent items (from cache).
        This is usually the fastest strategy.
        """
        # Update cache if stale
        if (datetime.utcnow() - self._cache_updated).seconds > 60:
            await self._update_recents_cache()
        
        # Simple scoring based on recency and project match
        scored_results = []
        for result in self._recents_cache:
            score = 1.0  # Base score for recency
            
            # Boost for project match
            if project_hint and result.metadata.get("project") == project_hint:
                score += 0.5
            
            # Simple text match
            if query.lower() in result.title.lower():
                score += 0.3
            if query.lower() in result.snippet.lower():
                score += 0.2
            
            scored_results.append((score, result))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored_results[:max_results]]
    
    async def _search_fts(
        self,
        query: str,
        project_hint: Optional[str],
        max_results: int
    ) -> List[SearchResult]:
        """Full-text search using Tantivy (stub for now)."""
        # This would call the actual FTS indexer
        await asyncio.sleep(0.01)  # Simulate FTS latency
        return []
    
    async def _search_vector(
        self,
        query: str,
        project_hint: Optional[str],
        max_results: int
    ) -> List[SearchResult]:
        """Vector similarity search using LanceDB (stub for now)."""
        # This would call the actual vector indexer
        await asyncio.sleep(0.05)  # Simulate vector search latency
        return []
    
    async def _search_analytics(
        self,
        query: str,
        project_hint: Optional[str],
        max_results: int
    ) -> List[SearchResult]:
        """Analytics-based search using DuckDB (stub for now)."""
        # This would query DuckDB for structured searches
        await asyncio.sleep(0.02)  # Simulate DB query latency
        return []
    
    def _is_useful(
        self,
        results: List[SearchResult],
        strategy: SearchStrategy,
        query: str,
        project_hint: Optional[str]
    ) -> bool:
        """
        Determine if search results are "useful" enough for early return.
        
        Useful = any result passes threshold OR matches project + has reasonable score
        """
        if not results:
            return False
        
        threshold = self.usefulness_threshold.get(strategy, 0.5)
        
        for result in results[:3]:  # Check top 3
            # Check score threshold
            if result.score >= threshold:
                return True
            
            # Project match with lower threshold
            if project_hint and result.metadata.get("project") == project_hint:
                if result.score >= threshold * 0.7:
                    return True
            
            # Exact query match in title
            if query.lower() in result.title.lower():
                return True
        
        return False
    
    def _merge_results(
        self,
        results_by_strategy: Dict[SearchStrategy, List[SearchResult]],
        max_results: int
    ) -> List[SearchResult]:
        """
        Merge results from multiple strategies.
        De-duplicate and re-score.
        """
        seen_ids = set()
        merged = []
        
        # Strategy weights for final scoring
        strategy_weights = {
            SearchStrategy.FTS: 1.0,
            SearchStrategy.VECTOR: 0.9,
            SearchStrategy.RECENTS: 0.7,
            SearchStrategy.ANALYTICS: 0.8
        }
        
        # Collect all results with weighted scores
        for strategy, results in results_by_strategy.items():
            weight = strategy_weights.get(strategy, 1.0)
            for result in results:
                if result.id not in seen_ids:
                    result.score *= weight
                    merged.append(result)
                    seen_ids.add(result.id)
        
        # Sort by final score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:max_results]
    
    def _generate_quick_actions(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[Dict[str, Any]]:
        """Generate quick actions based on search results."""
        actions = []
        
        if results:
            # Open top result
            actions.append({
                "action": "open",
                "label": f"Open {results[0].title}",
                "target": results[0].path
            })
            
            # Create related note
            actions.append({
                "action": "create_related",
                "label": f"Create note about '{query}'",
                "template": "note",
                "links": [r.id for r in results[:3]]
            })
        
        # Always available: new capture
        actions.append({
            "action": "capture",
            "label": f"Capture thought about '{query}'",
            "prefill": query
        })
        
        return actions
    
    async def _generate_suggestions(
        self,
        results: List[SearchResult],
        query: str,
        context_hint: Optional[str]
    ) -> List[str]:
        """Generate suggestions based on results (stub for now)."""
        suggestions = []
        
        if results:
            # Simple heuristic suggestions
            if any("task" in r.metadata.get("type", "") for r in results):
                suggestions.append(f"Review pending tasks related to '{query}'")
            
            if context_hint and "code" in context_hint.lower():
                suggestions.append(f"Document implementation notes for '{query}'")
        
        return suggestions[:3]
    
    async def _update_recents_cache(self) -> None:
        """Update the recents cache from storage."""
        # This would query DuckDB or read recent files
        # For now, create some dummy data
        self._recents_cache = [
            SearchResult(
                id=f"recent-{i}",
                path=f"notes/recent-{i}.md",
                title=f"Recent Note {i}",
                snippet=f"This is a recent note snippet {i}...",
                score=0.5,
                source=SearchStrategy.RECENTS,
                metadata={"modified": datetime.utcnow() - timedelta(hours=i)}
            )
            for i in range(10)
        ]
        self._cache_updated = datetime.utcnow()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._latency_histogram:
            return {}
        
        sorted_latencies = sorted(self._latency_histogram)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            "mean": sum(sorted_latencies) / n,
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1]
        }