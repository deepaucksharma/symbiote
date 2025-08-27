"""Production Analytics Engine using DuckDB.

This module provides real analytics capabilities:
- Time-series analysis of document activity
- Relationship graphs and network analysis
- Aggregated metrics and statistics
- Query performance tracking
- User behavior analytics
- Predictive analytics and forecasting
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict

try:
    import duckdb
    import pandas as pd
    import numpy as np
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from loguru import logger
from ...daemon.algorithms import SearchCandidate


@dataclass
class AnalyticsEvent:
    """Represents an analytics event."""
    event_id: str
    event_type: str
    timestamp: datetime
    document_id: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    metric: str
    value: float
    dimensions: Dict[str, str]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric': self.metric,
            'value': self.value,
            'dimensions': self.dimensions
        }


@dataclass
class RelationshipEdge:
    """Edge in document relationship graph."""
    source: str
    target: str
    relationship_type: str
    weight: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DuckDBAnalytics:
    """Production analytics engine using DuckDB."""
    
    def __init__(self, 
                 db_path: Optional[Path] = None,
                 memory_limit: str = "1GB"):
        """
        Initialize DuckDB analytics engine.
        
        Args:
            db_path: Path to database file (None for in-memory)
            memory_limit: Memory limit for DuckDB
        """
        self.db_path = db_path
        self.memory_limit = memory_limit
        self.conn = None
        
        if HAS_DUCKDB:
            self._initialize_db()
        else:
            logger.warning("DuckDB not available, analytics limited")
    
    def _initialize_db(self):
        """Initialize DuckDB connection and schema."""
        # Connect to DuckDB
        if self.db_path:
            self.conn = duckdb.connect(str(self.db_path))
        else:
            self.conn = duckdb.connect(':memory:')
        
        # Set configuration
        self.conn.execute(f"SET memory_limit='{self.memory_limit}'")
        self.conn.execute("SET threads TO 4")
        
        # Create schema
        self._create_schema()
        
        logger.info(f"DuckDB analytics initialized {'in-memory' if not self.db_path else f'at {self.db_path}'}")
    
    def _create_schema(self):
        """Create database schema."""
        # Documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                title VARCHAR,
                path VARCHAR,
                type VARCHAR,
                project VARCHAR,
                created TIMESTAMP,
                modified TIMESTAMP,
                size INTEGER,
                word_count INTEGER,
                metadata JSON
            )
        """)
        
        # Events table for tracking all activities
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR PRIMARY KEY,
                event_type VARCHAR,
                timestamp TIMESTAMP,
                document_id VARCHAR,
                user_id VARCHAR,
                session_id VARCHAR,
                metadata JSON,
                INDEX idx_timestamp (timestamp),
                INDEX idx_document (document_id),
                INDEX idx_type (event_type)
            )
        """)
        
        # Search queries table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS search_queries (
                query_id VARCHAR PRIMARY KEY,
                query_text VARCHAR,
                timestamp TIMESTAMP,
                result_count INTEGER,
                clicked_position INTEGER,
                latency_ms DOUBLE,
                strategies_used JSON,
                user_satisfaction DOUBLE,
                INDEX idx_query_time (timestamp)
            )
        """)
        
        # Document relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS document_relationships (
                source_id VARCHAR,
                target_id VARCHAR,
                relationship_type VARCHAR,
                weight DOUBLE,
                created TIMESTAMP,
                metadata JSON,
                PRIMARY KEY (source_id, target_id, relationship_type),
                INDEX idx_source (source_id),
                INDEX idx_target (target_id)
            )
        """)
        
        # Metrics time series table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP,
                metric_name VARCHAR,
                value DOUBLE,
                dimensions JSON,
                INDEX idx_metric_time (metric_name, timestamp)
            )
        """)
        
        # User sessions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds INTEGER,
                event_count INTEGER,
                documents_accessed JSON,
                metadata JSON
            )
        """)
        
        # Patterns table for storing detected patterns
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id VARCHAR PRIMARY KEY,
                pattern_type VARCHAR,
                detected_at TIMESTAMP,
                confidence DOUBLE,
                affected_documents JSON,
                metadata JSON,
                INDEX idx_pattern_time (detected_at)
            )
        """)
    
    async def track_event(self, event: AnalyticsEvent):
        """
        Track an analytics event.
        
        Args:
            event: Event to track
        """
        if not HAS_DUCKDB:
            return
        
        try:
            self.conn.execute("""
                INSERT INTO events (event_id, event_type, timestamp, document_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                event.event_id,
                event.event_type,
                event.timestamp,
                event.document_id,
                json.dumps(event.metadata)
            ])
            
        except Exception as e:
            logger.error(f"Failed to track event: {e}")
    
    async def track_search(self, 
                          query: str,
                          results: List[SearchCandidate],
                          latency_ms: float,
                          strategies: List[str]):
        """
        Track a search query and its performance.
        
        Args:
            query: Search query text
            results: Search results
            latency_ms: Query latency
            strategies: Strategies used
        """
        if not HAS_DUCKDB:
            return
        
        try:
            query_id = hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:16]
            
            self.conn.execute("""
                INSERT INTO search_queries 
                (query_id, query_text, timestamp, result_count, latency_ms, strategies_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                query_id,
                query,
                datetime.now(),
                len(results),
                latency_ms,
                json.dumps(strategies)
            ])
            
        except Exception as e:
            logger.error(f"Failed to track search: {e}")
    
    async def update_document_stats(self, 
                                   doc_id: str,
                                   metadata: Dict[str, Any]):
        """
        Update document statistics.
        
        Args:
            doc_id: Document ID
            metadata: Document metadata
        """
        if not HAS_DUCKDB:
            return
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, path, type, project, created, modified, size, word_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                doc_id,
                metadata.get('title'),
                metadata.get('path'),
                metadata.get('type'),
                metadata.get('project'),
                metadata.get('created'),
                metadata.get('modified'),
                metadata.get('size'),
                metadata.get('word_count'),
                json.dumps(metadata.get('extra', {}))
            ])
            
        except Exception as e:
            logger.error(f"Failed to update document stats: {e}")
    
    async def add_relationship(self, edge: RelationshipEdge):
        """
        Add a document relationship.
        
        Args:
            edge: Relationship edge
        """
        if not HAS_DUCKDB:
            return
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO document_relationships
                (source_id, target_id, relationship_type, weight, created, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                edge.source,
                edge.target,
                edge.relationship_type,
                edge.weight,
                datetime.now(),
                json.dumps(edge.metadata)
            ])
            
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
    
    async def record_metric(self, metric: TimeSeriesData):
        """
        Record a time series metric.
        
        Args:
            metric: Metric data point
        """
        if not HAS_DUCKDB:
            return
        
        try:
            self.conn.execute("""
                INSERT INTO metrics (timestamp, metric_name, value, dimensions)
                VALUES (?, ?, ?, ?)
            """, [
                metric.timestamp,
                metric.metric,
                metric.value,
                json.dumps(metric.dimensions)
            ])
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    async def get_document_activity(self, 
                                   days: int = 30) -> pd.DataFrame:
        """
        Get document activity over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DataFrame with activity data
        """
        if not HAS_DUCKDB:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    DATE_TRUNC('day', timestamp) as day,
                    COUNT(*) as event_count,
                    COUNT(DISTINCT document_id) as unique_documents,
                    COUNT(DISTINCT session_id) as sessions
                FROM events
                WHERE timestamp > NOW() - INTERVAL ? DAY
                GROUP BY day
                ORDER BY day
            """
            
            df = self.conn.execute(query, [days]).fetchdf()
            return df
            
        except Exception as e:
            logger.error(f"Failed to get document activity: {e}")
            return pd.DataFrame()
    
    async def get_popular_documents(self, 
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular documents by access frequency.
        
        Args:
            limit: Number of documents to return
            
        Returns:
            List of popular documents with stats
        """
        if not HAS_DUCKDB:
            return []
        
        try:
            query = """
                SELECT 
                    d.id,
                    d.title,
                    d.type,
                    COUNT(e.event_id) as access_count,
                    MAX(e.timestamp) as last_accessed,
                    AVG(CASE WHEN e.event_type = 'search_click' 
                        THEN CAST(e.metadata->>'position' AS INTEGER) 
                        ELSE NULL END) as avg_click_position
                FROM documents d
                LEFT JOIN events e ON d.id = e.document_id
                WHERE e.timestamp > NOW() - INTERVAL 30 DAY
                GROUP BY d.id, d.title, d.type
                ORDER BY access_count DESC
                LIMIT ?
            """
            
            results = self.conn.execute(query, [limit]).fetchall()
            
            return [
                {
                    'id': row[0],
                    'title': row[1],
                    'type': row[2],
                    'access_count': row[3],
                    'last_accessed': row[4],
                    'avg_click_position': row[5]
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get popular documents: {e}")
            return []
    
    async def get_search_performance(self) -> Dict[str, Any]:
        """
        Get search performance metrics.
        
        Returns:
            Search performance statistics
        """
        if not HAS_DUCKDB:
            return {}
        
        try:
            # Overall stats
            overall = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(latency_ms) as avg_latency,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
                    AVG(result_count) as avg_results,
                    AVG(CASE WHEN clicked_position IS NOT NULL THEN 1 ELSE 0 END) as click_through_rate
                FROM search_queries
                WHERE timestamp > NOW() - INTERVAL 7 DAY
            """).fetchone()
            
            # Strategy performance
            strategy_perf = self.conn.execute("""
                SELECT 
                    JSON_EXTRACT_STRING(strategies_used, '$[0]') as strategy,
                    COUNT(*) as query_count,
                    AVG(latency_ms) as avg_latency
                FROM search_queries
                WHERE timestamp > NOW() - INTERVAL 7 DAY
                GROUP BY strategy
            """).fetchall()
            
            return {
                'overall': {
                    'total_queries': overall[0],
                    'avg_latency_ms': overall[1],
                    'p50_latency_ms': overall[2],
                    'p95_latency_ms': overall[3],
                    'p99_latency_ms': overall[4],
                    'avg_results': overall[5],
                    'click_through_rate': overall[6]
                },
                'by_strategy': [
                    {
                        'strategy': row[0],
                        'query_count': row[1],
                        'avg_latency_ms': row[2]
                    }
                    for row in strategy_perf
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get search performance: {e}")
            return {}
    
    async def get_document_graph(self, 
                                min_weight: float = 0.5) -> Dict[str, Any]:
        """
        Get document relationship graph.
        
        Args:
            min_weight: Minimum relationship weight
            
        Returns:
            Graph data with nodes and edges
        """
        if not HAS_DUCKDB:
            return {'nodes': [], 'edges': []}
        
        try:
            # Get relationships
            edges = self.conn.execute("""
                SELECT 
                    source_id,
                    target_id,
                    relationship_type,
                    weight,
                    metadata
                FROM document_relationships
                WHERE weight >= ?
            """, [min_weight]).fetchall()
            
            # Get unique nodes
            nodes_query = """
                SELECT DISTINCT id, title, type
                FROM documents
                WHERE id IN (
                    SELECT source_id FROM document_relationships WHERE weight >= ?
                    UNION
                    SELECT target_id FROM document_relationships WHERE weight >= ?
                )
            """
            nodes = self.conn.execute(nodes_query, [min_weight, min_weight]).fetchall()
            
            return {
                'nodes': [
                    {
                        'id': node[0],
                        'label': node[1],
                        'type': node[2]
                    }
                    for node in nodes
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'type': edge[2],
                        'weight': edge[3],
                        'metadata': json.loads(edge[4]) if edge[4] else {}
                    }
                    for edge in edges
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get document graph: {e}")
            return {'nodes': [], 'edges': []}
    
    async def predict_next_document(self, 
                                   current_doc: str,
                                   limit: int = 5) -> List[Dict[str, Any]]:
        """
        Predict next likely documents based on patterns.
        
        Args:
            current_doc: Current document ID
            limit: Number of predictions
            
        Returns:
            Predicted next documents with confidence
        """
        if not HAS_DUCKDB:
            return []
        
        try:
            # Find documents accessed after current doc in sessions
            query = """
                WITH session_sequences AS (
                    SELECT 
                        session_id,
                        document_id,
                        LAG(document_id) OVER (PARTITION BY session_id ORDER BY timestamp) as prev_doc,
                        LEAD(document_id) OVER (PARTITION BY session_id ORDER BY timestamp) as next_doc
                    FROM events
                    WHERE document_id IS NOT NULL
                )
                SELECT 
                    next_doc as predicted_doc,
                    COUNT(*) as frequency,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as confidence
                FROM session_sequences
                WHERE prev_doc = ?
                    AND next_doc IS NOT NULL
                GROUP BY next_doc
                ORDER BY frequency DESC
                LIMIT ?
            """
            
            results = self.conn.execute(query, [current_doc, limit]).fetchall()
            
            predictions = []
            for row in results:
                # Get document details
                doc_info = self.conn.execute(
                    "SELECT title, type FROM documents WHERE id = ?",
                    [row[0]]
                ).fetchone()
                
                if doc_info:
                    predictions.append({
                        'document_id': row[0],
                        'title': doc_info[0],
                        'type': doc_info[1],
                        'confidence': row[2],
                        'frequency': row[1]
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict next document: {e}")
            return []
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in usage patterns.
        
        Returns:
            List of detected anomalies
        """
        if not HAS_DUCKDB:
            return []
        
        try:
            anomalies = []
            
            # Detect unusual activity spikes
            activity_query = """
                WITH daily_stats AS (
                    SELECT 
                        DATE_TRUNC('day', timestamp) as day,
                        COUNT(*) as event_count
                    FROM events
                    WHERE timestamp > NOW() - INTERVAL 30 DAY
                    GROUP BY day
                ),
                stats AS (
                    SELECT 
                        AVG(event_count) as mean_count,
                        STDDEV(event_count) as std_count
                    FROM daily_stats
                )
                SELECT 
                    d.day,
                    d.event_count,
                    s.mean_count,
                    s.std_count,
                    (d.event_count - s.mean_count) / s.std_count as z_score
                FROM daily_stats d, stats s
                WHERE ABS((d.event_count - s.mean_count) / s.std_count) > 2
                ORDER BY d.day DESC
            """
            
            spikes = self.conn.execute(activity_query).fetchall()
            
            for spike in spikes:
                anomalies.append({
                    'type': 'activity_spike',
                    'date': spike[0],
                    'value': spike[1],
                    'z_score': spike[4],
                    'severity': 'high' if abs(spike[4]) > 3 else 'medium'
                })
            
            # Detect unusual search patterns
            search_query = """
                WITH search_stats AS (
                    SELECT 
                        AVG(latency_ms) as mean_latency,
                        STDDEV(latency_ms) as std_latency
                    FROM search_queries
                    WHERE timestamp > NOW() - INTERVAL 7 DAY
                )
                SELECT 
                    query_text,
                    latency_ms,
                    timestamp,
                    (latency_ms - mean_latency) / std_latency as z_score
                FROM search_queries, search_stats
                WHERE timestamp > NOW() - INTERVAL 1 DAY
                    AND ABS((latency_ms - mean_latency) / std_latency) > 2
                ORDER BY timestamp DESC
                LIMIT 10
            """
            
            slow_searches = self.conn.execute(search_query).fetchall()
            
            for search in slow_searches:
                anomalies.append({
                    'type': 'slow_search',
                    'query': search[0],
                    'latency_ms': search[1],
                    'timestamp': search[2],
                    'z_score': search[3],
                    'severity': 'high' if search[1] > 1000 else 'medium'
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []
    
    async def generate_insights(self) -> Dict[str, Any]:
        """
        Generate analytical insights.
        
        Returns:
            Dictionary of insights
        """
        if not HAS_DUCKDB:
            return {}
        
        try:
            insights = {}
            
            # Peak usage times
            peak_times = self.conn.execute("""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as event_count
                FROM events
                WHERE timestamp > NOW() - INTERVAL 7 DAY
                GROUP BY hour
                ORDER BY event_count DESC
                LIMIT 3
            """).fetchall()
            
            insights['peak_hours'] = [
                {'hour': row[0], 'activity': row[1]}
                for row in peak_times
            ]
            
            # Most connected documents
            connected = self.conn.execute("""
                SELECT 
                    source_id,
                    COUNT(DISTINCT target_id) as connection_count,
                    AVG(weight) as avg_weight
                FROM document_relationships
                GROUP BY source_id
                ORDER BY connection_count DESC
                LIMIT 5
            """).fetchall()
            
            insights['most_connected'] = [
                {
                    'document': row[0],
                    'connections': row[1],
                    'avg_weight': row[2]
                }
                for row in connected
            ]
            
            # Growth trends
            growth = self.conn.execute("""
                WITH weekly_counts AS (
                    SELECT 
                        DATE_TRUNC('week', created) as week,
                        COUNT(*) as new_documents
                    FROM documents
                    WHERE created > NOW() - INTERVAL 8 WEEK
                    GROUP BY week
                    ORDER BY week
                )
                SELECT 
                    week,
                    new_documents,
                    LAG(new_documents) OVER (ORDER BY week) as prev_week,
                    (new_documents - LAG(new_documents) OVER (ORDER BY week)) * 100.0 / 
                        NULLIF(LAG(new_documents) OVER (ORDER BY week), 0) as growth_rate
                FROM weekly_counts
            """).fetchall()
            
            insights['growth_trend'] = [
                {
                    'week': row[0],
                    'new_documents': row[1],
                    'growth_rate': row[3]
                }
                for row in growth if row[3] is not None
            ]
            
            # Query patterns
            patterns = self.conn.execute("""
                SELECT 
                    CASE 
                        WHEN LENGTH(query_text) < 10 THEN 'short'
                        WHEN LENGTH(query_text) < 30 THEN 'medium'
                        ELSE 'long'
                    END as query_type,
                    COUNT(*) as count,
                    AVG(result_count) as avg_results,
                    AVG(latency_ms) as avg_latency
                FROM search_queries
                WHERE timestamp > NOW() - INTERVAL 7 DAY
                GROUP BY query_type
            """).fetchall()
            
            insights['query_patterns'] = [
                {
                    'type': row[0],
                    'count': row[1],
                    'avg_results': row[2],
                    'avg_latency': row[3]
                }
                for row in patterns
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {}
    
    async def optimize_indexes(self):
        """Optimize database indexes for better performance."""
        if not HAS_DUCKDB:
            return
        
        try:
            # Analyze tables
            tables = ['documents', 'events', 'search_queries', 
                     'document_relationships', 'metrics']
            
            for table in tables:
                self.conn.execute(f"ANALYZE {table}")
            
            logger.info("Analytics indexes optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize indexes: {e}")
    
    async def export_analytics(self, 
                              output_path: Path,
                              format: str = 'parquet') -> bool:
        """
        Export analytics data.
        
        Args:
            output_path: Output directory
            format: Export format (parquet, csv, json)
            
        Returns:
            Success status
        """
        if not HAS_DUCKDB:
            return False
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            tables = ['documents', 'events', 'search_queries', 
                     'document_relationships', 'metrics', 'patterns']
            
            for table in tables:
                if format == 'parquet':
                    self.conn.execute(f"""
                        COPY {table} TO '{output_path}/{table}.parquet' 
                        (FORMAT PARQUET)
                    """)
                elif format == 'csv':
                    self.conn.execute(f"""
                        COPY {table} TO '{output_path}/{table}.csv' 
                        (FORMAT CSV, HEADER)
                    """)
                elif format == 'json':
                    df = self.conn.execute(f"SELECT * FROM {table}").fetchdf()
                    df.to_json(output_path / f"{table}.json", orient='records')
            
            logger.info(f"Analytics exported to {output_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export analytics: {e}")
            return False