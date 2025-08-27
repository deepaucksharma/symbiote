"""
Synthesis worker for background pattern detection and theme extraction.
Runs periodically to identify connections and generate insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from .compat import duckdb, logger

from .bus import EventBus
from .algorithms import ThemeSynthesizer


class SynthesisWorker:
    """Background worker for pattern synthesis and insight generation."""
    
    def __init__(
        self,
        vault_path,
        event_bus: EventBus,
        db_path: Optional[str] = None,
        interval_seconds: int = 300  # Run every 5 minutes
    ):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.db_path = db_path or str(vault_path / ".sym" / "analytics.db")
        self.interval_seconds = interval_seconds
        
        self.running = False
        self.task = None
        
        # Track synthesis state
        self.last_run = None
        self.patterns = {
            "themes": [],
            "clusters": [],
            "trends": [],
            "connections": []
        }
        
        # Statistics
        self.stats = {
            "runs": 0,
            "patterns_found": 0,
            "last_duration_ms": 0
        }
    
    async def start(self):
        """Start the synthesis worker."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        logger.info(f"Synthesis worker started (interval: {self.interval_seconds}s)")
    
    async def stop(self):
        """Stop the synthesis worker."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Synthesis worker stopped")
    
    async def _run_loop(self):
        """Main synthesis loop."""
        while self.running:
            try:
                # Run synthesis
                await self.run_synthesis()
                
                # Wait for next interval
                await asyncio.sleep(self.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                await asyncio.sleep(30)  # Brief pause on error
    
    async def run_synthesis(self):
        """Run a single synthesis pass."""
        start = asyncio.get_event_loop().time()
        logger.debug("Starting synthesis pass...")
        
        try:
            # Connect to analytics DB
            conn = duckdb.connect(self.db_path)
            
            # Get recent activity window (last 7 days)
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
            
            # Extract themes from recent captures
            themes = await self._extract_themes(conn, cutoff)
            
            # Detect activity clusters
            clusters = await self._detect_clusters(conn, cutoff)
            
            # Identify trends
            trends = await self._identify_trends(conn, cutoff)
            
            # Find connections between items
            connections = await self._find_connections(conn, cutoff)
            
            # Update patterns
            self.patterns = {
                "themes": themes,
                "clusters": clusters,
                "trends": trends,
                "connections": connections
            }
            
            # Store synthesis results
            await self._store_synthesis(conn)
            
            # Emit event with new patterns
            if themes or clusters or trends or connections:
                await self.event_bus.emit("synthesis:complete", {
                    "timestamp": datetime.utcnow().isoformat(),
                    "patterns": self.patterns
                })
                self.stats["patterns_found"] += len(themes) + len(clusters) + len(trends) + len(connections)
            
            conn.close()
            
            # Update stats
            self.stats["runs"] += 1
            self.stats["last_duration_ms"] = (asyncio.get_event_loop().time() - start) * 1000
            self.last_run = datetime.utcnow()
            
            logger.debug(f"Synthesis complete in {self.stats['last_duration_ms']:.1f}ms")
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
    
    async def _extract_themes(self, conn: duckdb.DuckDBPyConnection, cutoff: str) -> List[Dict]:
        """Extract dominant themes from recent activity."""
        themes = []
        
        try:
            # Get tag frequencies
            result = conn.execute("""
                SELECT 
                    tag,
                    COUNT(*) as frequency,
                    COUNT(DISTINCT DATE(created_at)) as active_days
                FROM capture_events,
                     UNNEST(string_split(tags, ',')) as t(tag)
                WHERE created_at >= ?
                  AND tag != ''
                GROUP BY tag
                HAVING COUNT(*) >= 3
                ORDER BY frequency DESC
                LIMIT 10
            """, [cutoff]).fetchall()
            
            for tag, freq, days in result:
                themes.append({
                    "type": "tag_theme",
                    "value": tag.strip(),
                    "frequency": freq,
                    "active_days": days,
                    "score": freq * (days / 7.0)  # Weight by consistency
                })
            
            # Get project activity themes
            result = conn.execute("""
                SELECT 
                    context->>'project' as project,
                    COUNT(*) as events,
                    COUNT(DISTINCT type) as event_types,
                    MAX(created_at) as last_activity
                FROM capture_events
                WHERE created_at >= ?
                  AND context->>'project' IS NOT NULL
                GROUP BY project
                HAVING COUNT(*) >= 5
                ORDER BY events DESC
                LIMIT 5
            """, [cutoff]).fetchall()
            
            for project, events, types, last_activity in result:
                themes.append({
                    "type": "project_focus",
                    "value": project,
                    "events": events,
                    "diversity": types,
                    "recency": last_activity,
                    "score": events * (types / 3.0)  # Reward diverse activity
                })
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
        
        return themes
    
    async def _detect_clusters(self, conn: duckdb.DuckDBPyConnection, cutoff: str) -> List[Dict]:
        """Detect temporal activity clusters."""
        clusters = []
        
        try:
            # Find bursts of activity
            result = conn.execute("""
                WITH hourly_activity AS (
                    SELECT 
                        DATE_TRUNC('hour', created_at) as hour,
                        COUNT(*) as events,
                        COUNT(DISTINCT type) as types,
                        ARRAY_AGG(DISTINCT context->>'project') as projects
                    FROM capture_events
                    WHERE created_at >= ?
                    GROUP BY hour
                    HAVING COUNT(*) >= 5
                )
                SELECT 
                    hour,
                    events,
                    types,
                    projects
                FROM hourly_activity
                ORDER BY events DESC
                LIMIT 10
            """, [cutoff]).fetchall()
            
            for hour, events, types, projects in result:
                # Filter out null projects
                active_projects = [p for p in projects if p]
                
                clusters.append({
                    "type": "activity_burst",
                    "timestamp": hour.isoformat(),
                    "events": events,
                    "event_types": types,
                    "projects": active_projects[:3],  # Top 3 projects
                    "intensity": events / types if types > 0 else events
                })
            
        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
        
        return clusters
    
    async def _identify_trends(self, conn: duckdb.DuckDBPyConnection, cutoff: str) -> List[Dict]:
        """Identify trending topics and patterns."""
        trends = []
        
        try:
            # Find rising topics (comparing last 3 days vs previous 4)
            result = conn.execute("""
                WITH recent AS (
                    SELECT 
                        LOWER(text) as topic,
                        COUNT(*) as recent_count
                    FROM capture_events
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 3 DAY)
                    GROUP BY topic
                ),
                previous AS (
                    SELECT 
                        LOWER(text) as topic,
                        COUNT(*) as prev_count
                    FROM capture_events
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                      AND created_at < DATE_SUB(NOW(), INTERVAL 3 DAY)
                    GROUP BY topic
                )
                SELECT 
                    r.topic,
                    r.recent_count,
                    COALESCE(p.prev_count, 0) as prev_count,
                    (r.recent_count - COALESCE(p.prev_count, 0)) / 
                        GREATEST(COALESCE(p.prev_count, 1), 1.0) as growth_rate
                FROM recent r
                LEFT JOIN previous p ON r.topic = p.topic
                WHERE r.recent_count >= 3
                  AND (r.recent_count > COALESCE(p.prev_count, 0) * 1.5 OR p.prev_count IS NULL)
                ORDER BY growth_rate DESC
                LIMIT 5
            """).fetchall()
            
            for topic, recent, previous, growth in result:
                # Extract key terms from topic
                words = topic.split()[:5]  # First 5 words
                
                trends.append({
                    "type": "rising_topic",
                    "keywords": words,
                    "recent_mentions": recent,
                    "previous_mentions": previous,
                    "growth_rate": growth,
                    "trend": "rising" if growth > 0 else "new"
                })
            
        except Exception as e:
            logger.error(f"Trend identification failed: {e}")
        
        return trends
    
    async def _find_connections(self, conn: duckdb.DuckDBPyConnection, cutoff: str) -> List[Dict]:
        """Find connections between recent items."""
        connections = []
        
        try:
            # Get recent items for link analysis
            result = conn.execute("""
                SELECT 
                    id,
                    text,
                    context->>'project' as project,
                    tags,
                    created_at
                FROM capture_events
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT 100
            """, [cutoff]).fetchall()
            
            items = []
            for row in result:
                items.append({
                    "id": row[0],
                    "title": row[1][:50] if row[1] else "",
                    "project": row[2],
                    "tags": row[3].split(",") if row[3] else [],
                    "modified": row[4].timestamp() if row[4] else 0
                })
            
            # Use ThemeSynthesizer to suggest links
            suggested_links = ThemeSynthesizer.suggest_links(items, threshold=0.4)
            
            # Convert to connection format
            for src_id, dst_id, score in suggested_links[:10]:
                # Find the items
                src_item = next((i for i in items if i["id"] == src_id), None)
                dst_item = next((i for i in items if i["id"] == dst_id), None)
                
                if src_item and dst_item:
                    connections.append({
                        "type": "suggested_link",
                        "source": {
                            "id": src_id,
                            "title": src_item["title"]
                        },
                        "target": {
                            "id": dst_id,
                            "title": dst_item["title"]
                        },
                        "score": score,
                        "reason": self._explain_connection(src_item, dst_item)
                    })
            
        except Exception as e:
            logger.error(f"Connection finding failed: {e}")
        
        return connections
    
    def _explain_connection(self, src: Dict, dst: Dict) -> str:
        """Generate explanation for why two items are connected."""
        reasons = []
        
        if src.get("project") == dst.get("project") and src.get("project"):
            reasons.append(f"same project ({src['project']})")
        
        common_tags = set(src.get("tags", [])) & set(dst.get("tags", []))
        if common_tags:
            reasons.append(f"shared tags ({', '.join(list(common_tags)[:2])})")
        
        # Check temporal proximity
        time_diff = abs(src.get("modified", 0) - dst.get("modified", 0))
        if time_diff < 3600:  # Within 1 hour
            reasons.append("created around same time")
        
        return " & ".join(reasons) if reasons else "semantic similarity"
    
    async def _store_synthesis(self, conn: duckdb.DuckDBPyConnection):
        """Store synthesis results in database."""
        try:
            # Create synthesis table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_results (
                    id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP,
                    patterns JSON,
                    stats JSON
                )
            """)
            
            # Store current synthesis
            conn.execute("""
                INSERT INTO synthesis_results (id, created_at, patterns, stats)
                VALUES (?, ?, ?, ?)
            """, [
                f"syn_{datetime.utcnow().timestamp()}",
                datetime.utcnow(),
                self.patterns,
                self.stats
            ])
            
            # Keep only last 100 synthesis results
            conn.execute("""
                DELETE FROM synthesis_results
                WHERE id NOT IN (
                    SELECT id FROM synthesis_results
                    ORDER BY created_at DESC
                    LIMIT 100
                )
            """)
            
        except Exception as e:
            logger.error(f"Failed to store synthesis: {e}")
    
    async def get_latest_patterns(self) -> Dict[str, Any]:
        """Get the latest synthesized patterns."""
        return {
            "timestamp": self.last_run.isoformat() if self.last_run else None,
            "patterns": self.patterns,
            "stats": self.stats
        }
    
    async def force_synthesis(self) -> Dict[str, Any]:
        """Force an immediate synthesis run."""
        await self.run_synthesis()
        return await self.get_latest_patterns()