"""Production synthesis worker for pattern detection and insight generation.

This module implements sophisticated synthesis capabilities:
- Temporal pattern detection (trends, cycles, anomalies)
- Document clustering and theme extraction
- Connection discovery between documents
- Insight generation with explainability
- Proactive suggestion synthesis
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum
import numpy as np
import heapq

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

from loguru import logger

from .bus import EventBus, Event
from .algorithms_production import (
    TFIDFProcessor,
    DocumentClusterer,
    LinkSuggestionEngine,
    Theme,
    DocumentCluster,
    LinkSuggestion
)


class PatternType(Enum):
    """Types of patterns to detect."""
    TEMPORAL_TREND = "temporal_trend"
    ACTIVITY_BURST = "activity_burst"
    TOPIC_EMERGENCE = "topic_emergence"
    CONNECTION_CLUSTER = "connection_cluster"
    WORKFLOW_PATTERN = "workflow_pattern"
    KNOWLEDGE_GAP = "knowledge_gap"


@dataclass
class DetectedPattern:
    """Represents a detected pattern in the vault."""
    pattern_type: PatternType
    confidence: float
    description: str
    affected_documents: List[str]
    time_range: Tuple[datetime, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.pattern_type.value,
            'confidence': self.confidence,
            'description': self.description,
            'document_count': len(self.affected_documents),
            'time_range': [
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat()
            ],
            'metadata': self.metadata,
            'suggestions': self.suggestions
        }


@dataclass
class SynthesisInsight:
    """High-level insight from synthesis."""
    title: str
    summary: str
    confidence: float
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    priority: int  # 1-5, higher is more important
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'summary': self.summary,
            'confidence': self.confidence,
            'evidence_count': len(self.evidence),
            'recommendations': self.recommendations,
            'priority': self.priority
        }


class TemporalAnalyzer:
    """Analyzes temporal patterns in document activity."""
    
    def __init__(self, window_days: int = 30):
        """
        Initialize temporal analyzer.
        
        Args:
            window_days: Analysis window in days
        """
        self.window_days = window_days
    
    def analyze_activity(self, 
                        documents: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """
        Analyze temporal patterns in document activity.
        
        Args:
            documents: List of documents with timestamps
            
        Returns:
            Detected temporal patterns
        """
        patterns = []
        
        if not documents:
            return patterns
        
        # Convert to time series
        df = self._create_time_series(documents)
        
        # Detect activity bursts
        bursts = self._detect_bursts(df)
        patterns.extend(bursts)
        
        # Detect trends
        trends = self._detect_trends(df)
        patterns.extend(trends)
        
        # Detect cycles
        cycles = self._detect_cycles(df)
        patterns.extend(cycles)
        
        return patterns
    
    def _create_time_series(self, documents: List[Dict]) -> 'pd.DataFrame':
        """Create time series from documents."""
        if not HAS_ML_DEPS:
            return None
        
        # Extract timestamps
        timestamps = []
        for doc in documents:
            if 'modified' in doc:
                timestamps.append(doc['modified'])
            elif 'created' in doc:
                timestamps.append(doc['created'])
        
        if not timestamps:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame({'timestamp': timestamps})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to daily counts
        daily_counts = df.resample('D').size()
        
        return pd.DataFrame({'count': daily_counts})
    
    def _detect_bursts(self, df: 'pd.DataFrame') -> List[DetectedPattern]:
        """Detect activity bursts."""
        patterns = []
        
        if df is None or df.empty:
            return patterns
        
        # Calculate rolling statistics
        window = min(7, len(df) // 4)
        if window < 2:
            return patterns
        
        rolling_mean = df['count'].rolling(window).mean()
        rolling_std = df['count'].rolling(window).std()
        
        # Detect bursts (> 2 std above mean)
        threshold = rolling_mean + 2 * rolling_std
        bursts = df[df['count'] > threshold]
        
        if not bursts.empty:
            for burst_date in bursts.index:
                pattern = DetectedPattern(
                    pattern_type=PatternType.ACTIVITY_BURST,
                    confidence=0.8,
                    description=f"Activity burst detected on {burst_date.date()}",
                    affected_documents=[],  # Would need to track actual docs
                    time_range=(burst_date, burst_date + timedelta(days=1)),
                    metadata={
                        'count': int(bursts.loc[burst_date, 'count']),
                        'threshold': float(threshold.loc[burst_date])
                    },
                    suggestions=[
                        "Review documents from this period for important events",
                        "Consider creating a summary of this active period"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_trends(self, df: 'pd.DataFrame') -> List[DetectedPattern]:
        """Detect trending topics over time."""
        patterns = []
        
        if df is None or df.empty or len(df) < 7:
            return patterns
        
        # Calculate trend using linear regression
        x = np.arange(len(df))
        y = df['count'].values
        
        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Determine trend direction
        if abs(slope) > 0.1:  # Significant trend
            if slope > 0:
                trend_type = "increasing"
                suggestions = [
                    "Activity is increasing - consider organizing recent work",
                    "Create a summary of recent progress"
                ]
            else:
                trend_type = "decreasing"
                suggestions = [
                    "Activity is decreasing - consider reviewing pending tasks",
                    "Check if there are blockers or incomplete work"
                ]
            
            pattern = DetectedPattern(
                pattern_type=PatternType.TEMPORAL_TREND,
                confidence=min(0.9, abs(slope) / 2),
                description=f"Document activity shows {trend_type} trend",
                affected_documents=[],
                time_range=(df.index[0], df.index[-1]),
                metadata={
                    'slope': float(slope),
                    'trend_type': trend_type
                },
                suggestions=suggestions
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_cycles(self, df: 'pd.DataFrame') -> List[DetectedPattern]:
        """Detect cyclical patterns."""
        patterns = []
        
        if df is None or df.empty or len(df) < 14:
            return patterns
        
        # Check for weekly patterns
        df['weekday'] = df.index.dayofweek
        weekday_avg = df.groupby('weekday')['count'].mean()
        
        # Check if there's significant variation
        if weekday_avg.std() > weekday_avg.mean() * 0.3:
            peak_day = weekday_avg.idxmax()
            low_day = weekday_avg.idxmin()
            
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                   'Friday', 'Saturday', 'Sunday']
            
            pattern = DetectedPattern(
                pattern_type=PatternType.TEMPORAL_TREND,
                confidence=0.7,
                description=f"Weekly pattern: Most active on {days[peak_day]}, "
                           f"least active on {days[low_day]}",
                affected_documents=[],
                time_range=(df.index[0], df.index[-1]),
                metadata={
                    'peak_day': days[peak_day],
                    'low_day': days[low_day],
                    'peak_avg': float(weekday_avg[peak_day]),
                    'low_avg': float(weekday_avg[low_day])
                },
                suggestions=[
                    f"Schedule important work for {days[peak_day]}",
                    f"Use {days[low_day]} for planning and review"
                ]
            )
            patterns.append(pattern)
        
        return patterns


class TopicEvolutionTracker:
    """Tracks how topics evolve over time."""
    
    def __init__(self):
        """Initialize topic evolution tracker."""
        self.tfidf_processor = TFIDFProcessor()
        self.previous_themes: List[Theme] = []
    
    def track_evolution(self,
                       documents_by_time: Dict[datetime, List[Dict]]) -> List[DetectedPattern]:
        """
        Track topic evolution over time periods.
        
        Args:
            documents_by_time: Documents grouped by time period
            
        Returns:
            Detected topic evolution patterns
        """
        patterns = []
        
        if len(documents_by_time) < 2:
            return patterns
        
        # Process each time period
        time_periods = sorted(documents_by_time.keys())
        themes_by_period = {}
        
        for period in time_periods:
            docs = documents_by_time[period]
            if docs:
                # Extract themes for this period
                doc_texts = {d['id']: d.get('content', '') for d in docs}
                
                if doc_texts:
                    self.tfidf_processor.fit_transform(doc_texts)
                    themes = self.tfidf_processor.extract_themes(num_themes=3)
                    themes_by_period[period] = themes
        
        # Detect emerging topics
        emerging = self._detect_emerging_topics(themes_by_period, time_periods)
        patterns.extend(emerging)
        
        # Detect declining topics
        declining = self._detect_declining_topics(themes_by_period, time_periods)
        patterns.extend(declining)
        
        # Detect topic shifts
        shifts = self._detect_topic_shifts(themes_by_period, time_periods)
        patterns.extend(shifts)
        
        return patterns
    
    def _detect_emerging_topics(self,
                               themes_by_period: Dict,
                               periods: List[datetime]) -> List[DetectedPattern]:
        """Detect newly emerging topics."""
        patterns = []
        
        if len(periods) < 2:
            return patterns
        
        # Compare recent period to earlier
        recent_themes = themes_by_period.get(periods[-1], [])
        earlier_themes = themes_by_period.get(periods[-2], [])
        
        # Find themes that are new or growing
        for theme in recent_themes:
            is_new = True
            
            for earlier_theme in earlier_themes:
                # Check keyword overlap
                overlap = set(theme.keywords[:5]) & set(earlier_theme.keywords[:5])
                if len(overlap) >= 2:
                    is_new = False
                    
                    # Check if growing
                    if theme.relevance_score > earlier_theme.relevance_score * 1.5:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.TOPIC_EMERGENCE,
                            confidence=0.7,
                            description=f"Growing topic: {', '.join(theme.keywords[:3])}",
                            affected_documents=theme.document_ids,
                            time_range=(periods[-2], periods[-1]),
                            metadata={
                                'keywords': theme.keywords[:5],
                                'growth_rate': theme.relevance_score / earlier_theme.relevance_score
                            },
                            suggestions=[
                                f"Explore growing topic: {theme.keywords[0]}",
                                "Consider creating a dedicated project for this theme"
                            ]
                        )
                        patterns.append(pattern)
                    break
            
            if is_new and theme.relevance_score > 0.3:
                pattern = DetectedPattern(
                    pattern_type=PatternType.TOPIC_EMERGENCE,
                    confidence=0.8,
                    description=f"New topic emerged: {', '.join(theme.keywords[:3])}",
                    affected_documents=theme.document_ids,
                    time_range=(periods[-1], periods[-1]),
                    metadata={
                        'keywords': theme.keywords[:5],
                        'relevance': theme.relevance_score
                    },
                    suggestions=[
                        f"New topic detected: {theme.keywords[0]}",
                        "Review related documents for insights"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_declining_topics(self,
                                themes_by_period: Dict,
                                periods: List[datetime]) -> List[DetectedPattern]:
        """Detect declining topics."""
        patterns = []
        
        if len(periods) < 2:
            return patterns
        
        recent_themes = themes_by_period.get(periods[-1], [])
        earlier_themes = themes_by_period.get(periods[-2], [])
        
        # Find themes that are declining or disappeared
        for earlier_theme in earlier_themes:
            found_recent = False
            
            for theme in recent_themes:
                overlap = set(theme.keywords[:5]) & set(earlier_theme.keywords[:5])
                if len(overlap) >= 2:
                    found_recent = True
                    
                    # Check if declining
                    if theme.relevance_score < earlier_theme.relevance_score * 0.5:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.TOPIC_EMERGENCE,
                            confidence=0.6,
                            description=f"Declining topic: {', '.join(earlier_theme.keywords[:3])}",
                            affected_documents=earlier_theme.document_ids,
                            time_range=(periods[-2], periods[-1]),
                            metadata={
                                'keywords': earlier_theme.keywords[:5],
                                'decline_rate': theme.relevance_score / earlier_theme.relevance_score
                            },
                            suggestions=[
                                f"Topic '{earlier_theme.keywords[0]}' is declining",
                                "Consider archiving or summarizing related work"
                            ]
                        )
                        patterns.append(pattern)
                    break
            
            if not found_recent and earlier_theme.relevance_score > 0.3:
                pattern = DetectedPattern(
                    pattern_type=PatternType.TOPIC_EMERGENCE,
                    confidence=0.7,
                    description=f"Topic disappeared: {', '.join(earlier_theme.keywords[:3])}",
                    affected_documents=earlier_theme.document_ids,
                    time_range=(periods[-2], periods[-1]),
                    metadata={
                        'keywords': earlier_theme.keywords[:5],
                        'last_relevance': earlier_theme.relevance_score
                    },
                    suggestions=[
                        f"Topic '{earlier_theme.keywords[0]}' no longer active",
                        "Review if work was completed or abandoned"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_topic_shifts(self,
                            themes_by_period: Dict,
                            periods: List[datetime]) -> List[DetectedPattern]:
        """Detect shifts in topic focus."""
        patterns = []
        
        if len(periods) < 3:
            return patterns
        
        # Track topic evolution across multiple periods
        topic_trajectories = defaultdict(list)
        
        for period in periods:
            themes = themes_by_period.get(period, [])
            for theme in themes:
                # Use first keyword as topic identifier
                if theme.keywords:
                    topic_id = theme.keywords[0]
                    topic_trajectories[topic_id].append({
                        'period': period,
                        'score': theme.relevance_score,
                        'keywords': theme.keywords
                    })
        
        # Detect significant shifts
        for topic_id, trajectory in topic_trajectories.items():
            if len(trajectory) >= 2:
                # Calculate volatility
                scores = [t['score'] for t in trajectory]
                if len(scores) > 1:
                    volatility = np.std(scores) / (np.mean(scores) + 0.01)
                    
                    if volatility > 0.5:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.TOPIC_EMERGENCE,
                            confidence=0.6,
                            description=f"Volatile topic: {topic_id}",
                            affected_documents=[],
                            time_range=(trajectory[0]['period'], trajectory[-1]['period']),
                            metadata={
                                'topic': topic_id,
                                'volatility': float(volatility),
                                'trajectory_length': len(trajectory)
                            },
                            suggestions=[
                                f"Topic '{topic_id}' shows irregular attention",
                                "Consider if this needs more consistent focus"
                            ]
                        )
                        patterns.append(pattern)
        
        return patterns


class ConnectionDiscovery:
    """Discovers hidden connections between documents."""
    
    def __init__(self):
        """Initialize connection discovery."""
        self.clusterer = DocumentClusterer()
        self.link_engine = LinkSuggestionEngine()
    
    def discover_connections(self,
                            documents: List[Dict[str, Any]],
                            existing_links: Dict[str, Set[str]] = None) -> List[DetectedPattern]:
        """
        Discover connections between documents.
        
        Args:
            documents: Documents to analyze
            existing_links: Known links between documents
            
        Returns:
            Detected connection patterns
        """
        patterns = []
        
        if len(documents) < 3:
            return patterns
        
        # Create TF-IDF matrix
        doc_texts = {d['id']: d.get('content', '') for d in documents}
        
        if not doc_texts:
            return patterns
        
        processor = TFIDFProcessor()
        tfidf_matrix = processor.fit_transform(doc_texts)
        
        # Find clusters
        clusters = self.clusterer.cluster_documents(
            tfidf_matrix,
            list(doc_texts.keys())
        )
        
        # Detect cluster patterns
        for cluster in clusters:
            if cluster.coherence_score > 0.7:
                pattern = DetectedPattern(
                    pattern_type=PatternType.CONNECTION_CLUSTER,
                    confidence=cluster.coherence_score,
                    description=f"Highly connected cluster of {len(cluster.document_ids)} documents",
                    affected_documents=cluster.document_ids,
                    time_range=(datetime.now(), datetime.now()),
                    metadata={
                        'cluster_id': cluster.cluster_id,
                        'keywords': cluster.keywords,
                        'coherence': cluster.coherence_score
                    },
                    suggestions=[
                        "These documents form a natural group",
                        "Consider creating a collection or project",
                        "Review for potential synthesis opportunity"
                    ]
                )
                patterns.append(pattern)
        
        # Find missing links
        link_suggestions = self.link_engine.suggest_links(
            tfidf_matrix,
            list(doc_texts.keys()),
            existing_links
        )
        
        # Group suggestions by strength
        strong_links = [s for s in link_suggestions if s.confidence > 0.85]
        
        if len(strong_links) > 3:
            pattern = DetectedPattern(
                pattern_type=PatternType.CONNECTION_CLUSTER,
                confidence=0.8,
                description=f"Found {len(strong_links)} potential document connections",
                affected_documents=list(set(
                    [s.source_id for s in strong_links] +
                    [s.target_id for s in strong_links]
                )),
                time_range=(datetime.now(), datetime.now()),
                metadata={
                    'link_count': len(strong_links),
                    'avg_confidence': np.mean([s.confidence for s in strong_links])
                },
                suggestions=[
                    "Review suggested connections",
                    "Consider adding cross-references",
                    "These documents may benefit from integration"
                ]
            )
            patterns.append(pattern)
        
        return patterns


class WorkflowDetector:
    """Detects workflow patterns in document creation."""
    
    def detect_workflows(self, documents: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """
        Detect workflow patterns.
        
        Args:
            documents: Documents with metadata
            
        Returns:
            Detected workflow patterns
        """
        patterns = []
        
        # Group documents by type and time
        doc_sequences = self._extract_sequences(documents)
        
        # Detect common sequences
        common_sequences = self._find_common_sequences(doc_sequences)
        
        for seq, count in common_sequences:
            if count >= 3:  # Repeated at least 3 times
                pattern = DetectedPattern(
                    pattern_type=PatternType.WORKFLOW_PATTERN,
                    confidence=min(0.9, count / 10),
                    description=f"Workflow pattern: {' → '.join(seq)}",
                    affected_documents=[],
                    time_range=(datetime.now(), datetime.now()),
                    metadata={
                        'sequence': seq,
                        'occurrences': count
                    },
                    suggestions=[
                        f"Common workflow detected: {' → '.join(seq)}",
                        "Consider creating a template for this workflow",
                        "Automate or streamline this process"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_sequences(self, documents: List[Dict]) -> List[List[str]]:
        """Extract document type sequences."""
        # Sort by creation time
        sorted_docs = sorted(
            documents,
            key=lambda d: d.get('created', datetime.min)
        )
        
        # Group into sessions (documents created within 1 hour)
        sessions = []
        current_session = []
        last_time = None
        
        for doc in sorted_docs:
            doc_time = doc.get('created')
            doc_type = doc.get('type', 'note')
            
            if doc_time and last_time:
                if (doc_time - last_time).total_seconds() > 3600:
                    if current_session:
                        sessions.append(current_session)
                    current_session = []
            
            current_session.append(doc_type)
            last_time = doc_time
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _find_common_sequences(self, 
                              sequences: List[List[str]]) -> List[Tuple[Tuple[str], int]]:
        """Find common subsequences."""
        # Extract all subsequences of length 2-4
        subsequence_counts = Counter()
        
        for seq in sequences:
            for length in range(2, min(5, len(seq) + 1)):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i:i+length])
                    subsequence_counts[subseq] += 1
        
        # Return most common
        return subsequence_counts.most_common(5)


class KnowledgeGapDetector:
    """Detects gaps in knowledge coverage."""
    
    def detect_gaps(self, 
                    documents: List[Dict[str, Any]],
                    expected_topics: List[str] = None) -> List[DetectedPattern]:
        """
        Detect knowledge gaps.
        
        Args:
            documents: Current documents
            expected_topics: Expected topics to cover
            
        Returns:
            Detected knowledge gaps
        """
        patterns = []
        
        # Analyze topic coverage
        covered_topics = self._extract_topics(documents)
        
        # Detect sparse areas
        sparse_topics = self._find_sparse_topics(covered_topics)
        
        for topic, doc_count in sparse_topics:
            pattern = DetectedPattern(
                pattern_type=PatternType.KNOWLEDGE_GAP,
                confidence=0.7,
                description=f"Sparse coverage for topic: {topic}",
                affected_documents=[],
                time_range=(datetime.now(), datetime.now()),
                metadata={
                    'topic': topic,
                    'document_count': doc_count
                },
                suggestions=[
                    f"Topic '{topic}' has limited documentation",
                    "Consider expanding coverage in this area",
                    "Add more detailed notes or references"
                ]
            )
            patterns.append(pattern)
        
        # Check against expected topics
        if expected_topics:
            missing = set(expected_topics) - covered_topics.keys()
            
            for topic in missing:
                pattern = DetectedPattern(
                    pattern_type=PatternType.KNOWLEDGE_GAP,
                    confidence=0.9,
                    description=f"Missing expected topic: {topic}",
                    affected_documents=[],
                    time_range=(datetime.now(), datetime.now()),
                    metadata={'missing_topic': topic},
                    suggestions=[
                        f"No documentation found for '{topic}'",
                        "This appears to be an important gap",
                        "Consider researching and documenting this topic"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_topics(self, documents: List[Dict]) -> Dict[str, int]:
        """Extract topic coverage from documents."""
        topic_counts = Counter()
        
        for doc in documents:
            # Extract from tags
            tags = doc.get('tags', [])
            for tag in tags:
                topic_counts[tag] += 1
            
            # Extract from title
            title = doc.get('title', '')
            title_words = title.lower().split()
            for word in title_words:
                if len(word) > 4:  # Skip short words
                    topic_counts[word] += 1
        
        return dict(topic_counts)
    
    def _find_sparse_topics(self, 
                           topic_counts: Dict[str, int]) -> List[Tuple[str, int]]:
        """Find topics with sparse coverage."""
        if not topic_counts:
            return []
        
        # Calculate statistics
        counts = list(topic_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Find topics below threshold
        threshold = max(2, mean_count - std_count)
        
        sparse = [
            (topic, count) for topic, count in topic_counts.items()
            if count < threshold and count > 0
        ]
        
        # Sort by count
        sparse.sort(key=lambda x: x[1])
        
        return sparse[:5]  # Return top 5 sparse topics


class SynthesisWorker:
    """Main synthesis worker that orchestrates pattern detection."""
    
    def __init__(self,
                 vault_path: Path,
                 event_bus: EventBus,
                 interval_minutes: int = 5):
        """
        Initialize synthesis worker.
        
        Args:
            vault_path: Path to vault
            event_bus: Event bus for notifications
            interval_minutes: Synthesis interval
        """
        self.vault_path = Path(vault_path)
        self.event_bus = event_bus
        self.interval_minutes = interval_minutes
        
        # Initialize analyzers
        self.temporal_analyzer = TemporalAnalyzer()
        self.topic_tracker = TopicEvolutionTracker()
        self.connection_discovery = ConnectionDiscovery()
        self.workflow_detector = WorkflowDetector()
        self.gap_detector = KnowledgeGapDetector()
        
        # State
        self.last_synthesis = datetime.now()
        self.detected_patterns: List[DetectedPattern] = []
        self.insights: List[SynthesisInsight] = []
        
        self.running = False
        self.task = None
    
    async def start(self):
        """Start synthesis worker."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._synthesis_loop())
        logger.info("Synthesis worker started")
    
    async def stop(self):
        """Stop synthesis worker."""
        self.running = False
        
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info("Synthesis worker stopped")
    
    async def _synthesis_loop(self):
        """Main synthesis loop."""
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval_minutes * 60)
                
                # Run synthesis
                await self.synthesize()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def synthesize(self) -> Dict[str, Any]:
        """
        Run synthesis and pattern detection.
        
        Returns:
            Synthesis results
        """
        start_time = time.time()
        
        logger.info("Starting synthesis run")
        
        # Load documents
        documents = await self._load_documents()
        
        if not documents:
            logger.warning("No documents to synthesize")
            return {'patterns': [], 'insights': []}
        
        # Clear previous patterns
        self.detected_patterns = []
        
        # Run pattern detection in parallel
        tasks = [
            self._run_temporal_analysis(documents),
            self._run_topic_evolution(documents),
            self._run_connection_discovery(documents),
            self._run_workflow_detection(documents),
            self._run_gap_detection(documents)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect patterns
        for result in results:
            if isinstance(result, list):
                self.detected_patterns.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Pattern detection failed: {result}")
        
        # Generate insights
        self.insights = self._generate_insights(self.detected_patterns)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        
        # Emit synthesis complete event
        await self.event_bus.emit(Event(
            type="synthesis.complete",
            data={
                'pattern_count': len(self.detected_patterns),
                'insight_count': len(self.insights),
                'document_count': len(documents),
                'elapsed_seconds': elapsed
            }
        ))
        
        # Update last synthesis time
        self.last_synthesis = datetime.now()
        
        logger.info(f"Synthesis complete: {len(self.detected_patterns)} patterns, "
                   f"{len(self.insights)} insights in {elapsed:.2f}s")
        
        return {
            'patterns': [p.to_dict() for p in self.detected_patterns],
            'insights': [i.to_dict() for i in self.insights],
            'document_count': len(documents),
            'elapsed_seconds': elapsed
        }
    
    async def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from vault."""
        documents = []
        
        for md_file in self.vault_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
                
                doc = {
                    'id': md_file.stem,
                    'title': md_file.stem.replace('_', ' ').title(),
                    'content': content,
                    'path': str(md_file.relative_to(self.vault_path)),
                    'created': datetime.fromtimestamp(md_file.stat().st_ctime),
                    'modified': datetime.fromtimestamp(md_file.stat().st_mtime),
                    'size': len(content),
                    'type': self._infer_document_type(md_file, content)
                }
                
                # Extract tags
                import re
                tags = re.findall(r'#(\w+)', content)
                if tags:
                    doc['tags'] = list(set(tags))
                
                documents.append(doc)
                
            except Exception as e:
                logger.debug(f"Failed to load {md_file}: {e}")
        
        return documents
    
    def _infer_document_type(self, path: Path, content: str) -> str:
        """Infer document type from path and content."""
        path_str = str(path).lower()
        
        if 'task' in path_str or '- [ ]' in content:
            return 'task'
        elif 'note' in path_str:
            return 'note'
        elif 'project' in path_str:
            return 'project'
        elif 'idea' in path_str:
            return 'idea'
        else:
            return 'note'
    
    async def _run_temporal_analysis(self, documents: List[Dict]) -> List[DetectedPattern]:
        """Run temporal pattern analysis."""
        return self.temporal_analyzer.analyze_activity(documents)
    
    async def _run_topic_evolution(self, documents: List[Dict]) -> List[DetectedPattern]:
        """Run topic evolution tracking."""
        # Group documents by week
        documents_by_week = defaultdict(list)
        
        for doc in documents:
            week_start = doc['modified'] - timedelta(days=doc['modified'].weekday())
            documents_by_week[week_start].append(doc)
        
        return self.topic_tracker.track_evolution(dict(documents_by_week))
    
    async def _run_connection_discovery(self, documents: List[Dict]) -> List[DetectedPattern]:
        """Run connection discovery."""
        return self.connection_discovery.discover_connections(documents)
    
    async def _run_workflow_detection(self, documents: List[Dict]) -> List[DetectedPattern]:
        """Run workflow detection."""
        return self.workflow_detector.detect_workflows(documents)
    
    async def _run_gap_detection(self, documents: List[Dict]) -> List[DetectedPattern]:
        """Run knowledge gap detection."""
        return self.gap_detector.detect_gaps(documents)
    
    def _generate_insights(self, patterns: List[DetectedPattern]) -> List[SynthesisInsight]:
        """Generate high-level insights from patterns."""
        insights = []
        
        # Group patterns by type
        patterns_by_type = defaultdict(list)
        for pattern in patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)
        
        # Generate insights based on pattern combinations
        
        # Activity insights
        if PatternType.ACTIVITY_BURST in patterns_by_type:
            bursts = patterns_by_type[PatternType.ACTIVITY_BURST]
            if len(bursts) > 2:
                insight = SynthesisInsight(
                    title="Frequent Activity Bursts Detected",
                    summary=f"You've had {len(bursts)} periods of intense activity. "
                           f"This suggests a reactive rather than proactive workflow.",
                    confidence=0.8,
                    evidence=[{'pattern': p.to_dict()} for p in bursts[:3]],
                    recommendations=[
                        "Consider more consistent daily capture habits",
                        "Review what triggers these bursts",
                        "Plan regular synthesis sessions"
                    ],
                    priority=3
                )
                insights.append(insight)
        
        # Topic insights
        if PatternType.TOPIC_EMERGENCE in patterns_by_type:
            emergent = patterns_by_type[PatternType.TOPIC_EMERGENCE]
            new_topics = [p for p in emergent if 'New topic' in p.description]
            
            if new_topics:
                insight = SynthesisInsight(
                    title="New Areas of Interest Emerging",
                    summary=f"{len(new_topics)} new topics have emerged in your recent work.",
                    confidence=0.9,
                    evidence=[{'pattern': p.to_dict()} for p in new_topics],
                    recommendations=[
                        "Review new topics for potential projects",
                        "Consider dedicating focused time to explore",
                        "Connect new topics to existing knowledge"
                    ],
                    priority=4
                )
                insights.append(insight)
        
        # Connection insights
        if PatternType.CONNECTION_CLUSTER in patterns_by_type:
            clusters = patterns_by_type[PatternType.CONNECTION_CLUSTER]
            high_coherence = [p for p in clusters if p.confidence > 0.8]
            
            if high_coherence:
                insight = SynthesisInsight(
                    title="Strong Document Clusters Found",
                    summary=f"Discovered {len(high_coherence)} groups of highly related documents.",
                    confidence=0.85,
                    evidence=[{'pattern': p.to_dict()} for p in high_coherence],
                    recommendations=[
                        "Consider merging or linking related documents",
                        "Create overview documents for each cluster",
                        "Use clusters to identify potential projects"
                    ],
                    priority=3
                )
                insights.append(insight)
        
        # Knowledge gap insights
        if PatternType.KNOWLEDGE_GAP in patterns_by_type:
            gaps = patterns_by_type[PatternType.KNOWLEDGE_GAP]
            if len(gaps) > 3:
                insight = SynthesisInsight(
                    title="Multiple Knowledge Gaps Identified",
                    summary=f"Found {len(gaps)} areas with insufficient documentation.",
                    confidence=0.75,
                    evidence=[{'pattern': p.to_dict()} for p in gaps[:3]],
                    recommendations=[
                        "Prioritize documenting critical gaps",
                        "Schedule research time for missing topics",
                        "Consider if gaps align with goals"
                    ],
                    priority=2
                )
                insights.append(insight)
        
        # Sort insights by priority
        insights.sort(key=lambda x: x.priority, reverse=True)
        
        return insights
    
    def get_latest_insights(self) -> Dict[str, Any]:
        """Get the latest synthesis results."""
        return {
            'last_synthesis': self.last_synthesis.isoformat(),
            'patterns': [p.to_dict() for p in self.detected_patterns],
            'insights': [i.to_dict() for i in self.insights],
            'pattern_summary': {
                pattern_type.value: len([
                    p for p in self.detected_patterns 
                    if p.pattern_type == pattern_type
                ])
                for pattern_type in PatternType
            }
        }