"""Production search orchestrator with advanced fusion and racing.

This module implements sophisticated search orchestration:
- Parallel racing of multiple search strategies
- Intelligent result fusion with learned weights
- Early termination when quality threshold met
- Adaptive timeout management
- Result caching and query understanding
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

from loguru import logger

from .algorithms import SearchCandidate
from .algorithms_production import SearchFusionEngine, TFIDFProcessor
from .indexers.fts_production import FTSIndexProduction
from .indexers.vector_production import VectorIndexProduction
from .bus import EventBus, Event


class SearchStrategy(Enum):
    """Available search strategies."""
    FTS = "fts"
    VECTOR = "vector"
    RECENTS = "recents"
    HYBRID = "hybrid"


@dataclass
class SearchRequest:
    """Encapsulates a search request with context."""
    query: str
    limit: int = 10
    project_hint: Optional[str] = None
    strategies: List[SearchStrategy] = field(default_factory=lambda: [
        SearchStrategy.FTS,
        SearchStrategy.VECTOR,
        SearchStrategy.RECENTS
    ])
    timeout_ms: int = 300
    quality_threshold: float = 0.55
    include_explanations: bool = True
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for this request."""
        key_parts = [
            self.query,
            str(self.limit),
            self.project_hint or '',
            ','.join(s.value for s in self.strategies)
        ]
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()


@dataclass
class SearchResult:
    """Complete search result with metadata."""
    candidates: List[SearchCandidate]
    total_found: int
    strategies_used: List[str]
    fusion_method: str
    latency_ms: Dict[str, float]
    cache_hit: bool = False
    explanations: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            'results': [
                {
                    'id': c.id,
                    'title': c.title,
                    'snippet': c.snippet,
                    'score': c.base_score,
                    'source': c.source,
                    'project': c.project
                }
                for c in self.candidates
            ],
            'total': self.total_found,
            'strategies': self.strategies_used,
            'fusion': self.fusion_method,
            'latency_ms': self.latency_ms,
            'cache_hit': self.cache_hit,
            'explanations': self.explanations
        }


class QueryCache:
    """LRU cache for search results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time to live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[SearchResult, datetime]] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[SearchResult]:
        """Get cached result if valid."""
        if key not in self.cache:
            return None
        
        result, timestamp = self.cache[key]
        
        # Check if expired
        if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
            del self.cache[key]
            self.access_order.remove(key)
            return None
        
        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        
        return result
    
    def put(self, key: str, result: SearchResult):
        """Store result in cache."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = (result, datetime.now())
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()


class SearchOrchestrator:
    """Orchestrates parallel search strategies with intelligent fusion."""
    
    def __init__(self,
                 vault_path: str,
                 event_bus: EventBus,
                 enable_cache: bool = True):
        """
        Initialize search orchestrator.
        
        Args:
            vault_path: Path to vault
            event_bus: Event bus for notifications
            enable_cache: Whether to enable result caching
        """
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.enable_cache = enable_cache
        
        # Initialize search engines
        self.fts_index = FTSIndexProduction(vault_path)
        self.vector_index = VectorIndexProduction(vault_path)
        
        # Initialize fusion engine
        self.fusion_engine = SearchFusionEngine()
        
        # Initialize cache
        self.cache = QueryCache() if enable_cache else None
        
        # Track performance metrics
        self.metrics = defaultdict(list)
        
        # Strategy health tracking
        self.strategy_health = {
            SearchStrategy.FTS: {'available': True, 'error_count': 0},
            SearchStrategy.VECTOR: {'available': True, 'error_count': 0},
            SearchStrategy.RECENTS: {'available': True, 'error_count': 0}
        }
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """
        Execute search with racing and fusion.
        
        Args:
            request: Search request
            
        Returns:
            Fused search results
        """
        start_time = time.time()
        
        # Check cache
        if self.cache and not request.user_context.get('force_fresh'):
            cached = self.cache.get(request.cache_key)
            if cached:
                cached.cache_hit = True
                logger.debug(f"Cache hit for query: {request.query}")
                return cached
        
        # Emit search started event
        await self.event_bus.emit(Event(
            type="search.started",
            data={'query': request.query, 'strategies': [s.value for s in request.strategies]}
        ))
        
        # Launch search strategies in parallel
        search_tasks = []
        strategy_names = []
        
        for strategy in request.strategies:
            if self.strategy_health[strategy]['available']:
                task = self._create_search_task(strategy, request)
                search_tasks.append(task)
                strategy_names.append(strategy.value)
        
        if not search_tasks:
            logger.error("No search strategies available")
            return SearchResult(
                candidates=[],
                total_found=0,
                strategies_used=[],
                fusion_method='none',
                latency_ms={'total': 0}
            )
        
        # Race strategies with timeout
        results = await self._race_strategies(
            search_tasks,
            strategy_names,
            request.timeout_ms,
            request.quality_threshold
        )
        
        # Fuse results
        fused_candidates = self._fuse_results(results, request)
        
        # Calculate latencies
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000
        
        latency_ms = {
            'total': total_latency,
            **{name: results[name]['latency'] for name in results}
        }
        
        # Generate explanations if requested
        explanations = None
        if request.include_explanations:
            explanations = self._generate_explanations(
                results,
                fused_candidates,
                request
            )
        
        # Create result
        result = SearchResult(
            candidates=fused_candidates[:request.limit],
            total_found=len(fused_candidates),
            strategies_used=list(results.keys()),
            fusion_method=self.fusion_engine.detect_query_type(request.query),
            latency_ms=latency_ms,
            explanations=explanations
        )
        
        # Cache result
        if self.cache:
            self.cache.put(request.cache_key, result)
        
        # Track metrics
        self._track_metrics(request, result)
        
        # Emit search completed event
        await self.event_bus.emit(Event(
            type="search.completed",
            data={
                'query': request.query,
                'result_count': len(result.candidates),
                'latency_ms': total_latency
            }
        ))
        
        return result
    
    def _create_search_task(self, 
                           strategy: SearchStrategy,
                           request: SearchRequest):
        """Create an async search task for a strategy."""
        if strategy == SearchStrategy.FTS:
            return self._search_fts(request)
        elif strategy == SearchStrategy.VECTOR:
            return self._search_vector(request)
        elif strategy == SearchStrategy.RECENTS:
            return self._search_recents(request)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _search_fts(self, request: SearchRequest) -> Dict[str, Any]:
        """Execute full-text search."""
        start = time.time()
        
        try:
            candidates = await self.fts_index.search(
                query=request.query,
                limit=request.limit * 2,  # Get extra for fusion
                project=request.project_hint
            )
            
            latency = (time.time() - start) * 1000
            
            return {
                'candidates': candidates,
                'latency': latency,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            self._mark_strategy_error(SearchStrategy.FTS)
            
            return {
                'candidates': [],
                'latency': (time.time() - start) * 1000,
                'error': str(e)
            }
    
    async def _search_vector(self, request: SearchRequest) -> Dict[str, Any]:
        """Execute vector similarity search."""
        start = time.time()
        
        try:
            candidates = await self.vector_index.search(
                query=request.query,
                limit=request.limit * 2,
                threshold=0.5
            )
            
            latency = (time.time() - start) * 1000
            
            return {
                'candidates': candidates,
                'latency': latency,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self._mark_strategy_error(SearchStrategy.VECTOR)
            
            return {
                'candidates': [],
                'latency': (time.time() - start) * 1000,
                'error': str(e)
            }
    
    async def _search_recents(self, request: SearchRequest) -> Dict[str, Any]:
        """Search recent documents."""
        start = time.time()
        
        try:
            # Get recent documents from vault
            from pathlib import Path
            vault_path = Path(self.vault_path)
            
            recent_files = sorted(
                vault_path.rglob("*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )[:50]  # Get 50 most recent
            
            candidates = []
            query_lower = request.query.lower()
            
            for file_path in recent_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Simple relevance check
                    if query_lower in content.lower():
                        # Calculate recency score
                        age_hours = (time.time() - file_path.stat().st_mtime) / 3600
                        recency_score = max(0.3, 1.0 - (age_hours / 168))  # Decay over 1 week
                        
                        candidate = SearchCandidate(
                            id=file_path.stem,
                            title=file_path.stem.replace('_', ' ').title(),
                            path=str(file_path.relative_to(vault_path)),
                            snippet=content[:200],
                            base_score=recency_score,
                            source='recents',
                            project=self._extract_project(file_path, vault_path),
                            tags=[],
                            modified=datetime.fromtimestamp(file_path.stat().st_mtime)
                        )
                        
                        candidates.append(candidate)
                        
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {e}")
            
            # Sort by score
            candidates.sort(key=lambda x: x.base_score, reverse=True)
            
            latency = (time.time() - start) * 1000
            
            return {
                'candidates': candidates[:request.limit * 2],
                'latency': latency,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Recents search failed: {e}")
            self._mark_strategy_error(SearchStrategy.RECENTS)
            
            return {
                'candidates': [],
                'latency': (time.time() - start) * 1000,
                'error': str(e)
            }
    
    def _extract_project(self, file_path, vault_path) -> Optional[str]:
        """Extract project from file path."""
        try:
            relative = file_path.relative_to(vault_path)
            parts = relative.parts
            
            if len(parts) > 1 and parts[0] == 'projects':
                return parts[1]
            
            return None
            
        except:
            return None
    
    async def _race_strategies(self,
                              tasks: List,
                              names: List[str],
                              timeout_ms: int,
                              quality_threshold: float) -> Dict[str, Dict]:
        """
        Race search strategies, returning early if quality met.
        
        Args:
            tasks: Search tasks to race
            names: Strategy names
            timeout_ms: Maximum time to wait
            quality_threshold: Quality threshold for early termination
            
        Returns:
            Results from completed strategies
        """
        results = {}
        pending = set(zip(tasks, names))
        timeout = timeout_ms / 1000.0
        start_time = time.time()
        
        while pending and (time.time() - start_time) < timeout:
            # Wait for next task to complete
            done, pending_tasks = await asyncio.wait(
                [task for task, _ in pending],
                timeout=min(0.05, timeout - (time.time() - start_time)),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                # Find corresponding name
                for pending_task, name in list(pending):
                    if pending_task == task:
                        pending.remove((pending_task, name))
                        
                        try:
                            result = await task
                            results[name] = result
                            
                            # Check if we have enough quality results
                            if self._check_quality_threshold(results, quality_threshold):
                                logger.debug(f"Quality threshold met, cancelling remaining tasks")
                                
                                # Cancel remaining tasks
                                for remaining_task, _ in pending:
                                    remaining_task.cancel()
                                
                                return results
                            
                        except Exception as e:
                            logger.error(f"Strategy {name} failed: {e}")
                            results[name] = {
                                'candidates': [],
                                'latency': 0,
                                'error': str(e)
                            }
                        
                        break
        
        # Cancel any remaining tasks
        for task, name in pending:
            task.cancel()
            logger.debug(f"Strategy {name} timed out")
        
        return results
    
    def _check_quality_threshold(self,
                                results: Dict[str, Dict],
                                threshold: float) -> bool:
        """Check if results meet quality threshold."""
        # Collect all candidates
        all_candidates = []
        
        for result in results.values():
            if result.get('candidates'):
                all_candidates.extend(result['candidates'][:3])
        
        if not all_candidates:
            return False
        
        # Check if top candidates meet threshold
        all_candidates.sort(key=lambda x: x.base_score, reverse=True)
        
        return len(all_candidates) >= 3 and all_candidates[0].base_score >= threshold
    
    def _fuse_results(self,
                     results: Dict[str, Dict],
                     request: SearchRequest) -> List[SearchCandidate]:
        """Fuse results from multiple strategies."""
        # Extract candidates from each strategy
        fts_candidates = results.get('fts', {}).get('candidates', [])
        vector_candidates = results.get('vector', {}).get('candidates', [])
        recents_candidates = results.get('recents', {}).get('candidates', [])
        
        # Use fusion engine
        fused = self.fusion_engine.fuse_results(
            fts_candidates,
            vector_candidates,
            recents_candidates,
            request.query,
            request.project_hint
        )
        
        return fused
    
    def _generate_explanations(self,
                              results: Dict[str, Dict],
                              candidates: List[SearchCandidate],
                              request: SearchRequest) -> Dict[str, Any]:
        """Generate explanations for search results."""
        explanations = {
            'query_type': self.fusion_engine.detect_query_type(request.query),
            'strategies_performance': {},
            'fusion_weights': {},
            'top_terms': []
        }
        
        # Strategy performance
        for name, result in results.items():
            explanations['strategies_performance'][name] = {
                'candidates_found': len(result.get('candidates', [])),
                'latency_ms': result.get('latency', 0),
                'error': result.get('error')
            }
        
        # Fusion weights used
        query_type = explanations['query_type']
        explanations['fusion_weights'] = self.fusion_engine.scenario_weights.get(
            query_type,
            self.fusion_engine.scenario_weights['balanced']
        )
        
        # Top terms from query
        try:
            # Simple term extraction
            terms = request.query.lower().split()
            term_freq = defaultdict(int)
            
            for candidate in candidates[:10]:
                content = (candidate.title + ' ' + candidate.snippet).lower()
                for term in terms:
                    if term in content:
                        term_freq[term] += 1
            
            explanations['top_terms'] = sorted(
                term_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
        except Exception as e:
            logger.debug(f"Failed to extract terms: {e}")
        
        return explanations
    
    def _mark_strategy_error(self, strategy: SearchStrategy):
        """Mark a strategy as having an error."""
        self.strategy_health[strategy]['error_count'] += 1
        
        # Disable strategy if too many errors
        if self.strategy_health[strategy]['error_count'] >= 5:
            self.strategy_health[strategy]['available'] = False
            logger.warning(f"Strategy {strategy.value} disabled due to errors")
    
    def _track_metrics(self, request: SearchRequest, result: SearchResult):
        """Track performance metrics for optimization."""
        metric = {
            'timestamp': datetime.now(),
            'query_length': len(request.query),
            'strategies_used': len(result.strategies_used),
            'results_found': len(result.candidates),
            'total_latency': result.latency_ms.get('total', 0),
            'cache_hit': result.cache_hit,
            'fusion_method': result.fusion_method
        }
        
        self.metrics['searches'].append(metric)
        
        # Keep only recent metrics
        if len(self.metrics['searches']) > 1000:
            self.metrics['searches'] = self.metrics['searches'][-1000:]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get search orchestrator statistics."""
        stats = {
            'total_searches': len(self.metrics['searches']),
            'cache_stats': {},
            'strategy_health': {},
            'average_latency_ms': 0,
            'fusion_distribution': defaultdict(int)
        }
        
        # Cache statistics
        if self.cache:
            stats['cache_stats'] = {
                'size': len(self.cache.cache),
                'max_size': self.cache.max_size,
                'ttl_seconds': self.cache.ttl_seconds
            }
        
        # Strategy health
        for strategy, health in self.strategy_health.items():
            stats['strategy_health'][strategy.value] = health
        
        # Average latency
        if self.metrics['searches']:
            total_latency = sum(m['total_latency'] for m in self.metrics['searches'])
            stats['average_latency_ms'] = total_latency / len(self.metrics['searches'])
        
        # Fusion method distribution
        for metric in self.metrics['searches']:
            stats['fusion_distribution'][metric['fusion_method']] += 1
        
        return stats