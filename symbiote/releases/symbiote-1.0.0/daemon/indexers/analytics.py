"""DuckDB analytics indexer for receipts, links, and structured queries."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from ..compat import duckdb, logger

from ..config import Config
from ..bus import Event, get_event_bus


class AnalyticsIndexer:
    """
    Manages DuckDB tables for:
    - Notes and tasks metadata
    - Receipts (explainability)
    - Links (suggested vs confirmed)
    - System events and audit logs
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.vault_path / ".sym" / "analytics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._event_bus = get_event_bus()
    
    async def initialize(self) -> None:
        """Initialize DuckDB connection and create tables."""
        self.conn = duckdb.connect(str(self.db_path))
        await self._create_tables()
        
        # Subscribe to events
        self._event_bus.subscribe("capture.written", self._handle_capture_event)
        self._event_bus.subscribe("indexer.update", self._handle_update_event)
        
        logger.info(f"Analytics indexer initialized at {self.db_path}")
    
    async def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        
        # Notes table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                path TEXT,
                title TEXT,
                project TEXT,
                tags TEXT,              -- JSON array
                ts_captured TIMESTAMP,
                ts_modified TIMESTAMP
            )
        """)
        
        # Tasks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                path TEXT,
                title TEXT,
                status TEXT,
                project TEXT,
                energy TEXT,
                effort_min INTEGER,
                due DATE,
                tags TEXT,              -- JSON array
                ts_captured TIMESTAMP,
                ts_modified TIMESTAMP
            )
        """)
        
        # System events (optional desktop context)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events_system (
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                app TEXT,
                active_window TEXT,
                pid INTEGER,
                switches INTEGER
            )
        """)
        
        # Suggestions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                text TEXT,
                kind TEXT,              -- next_action|clarify|review|link_suggestion|plan
                context TEXT,           -- JSON blob
                receipts_id TEXT,
                accepted BOOLEAN,
                accepted_at TIMESTAMP,
                follow_through BOOLEAN,
                follow_through_at TIMESTAMP
            )
        """)
        
        # Receipts table (for explainability)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                suggestion_text TEXT,
                sources TEXT,           -- JSON array
                heuristics TEXT,        -- JSON array
                confidence TEXT,        -- low|medium|high
                outbound_preview TEXT,  -- JSON (payload for cloud if any)
                version INTEGER
            )
        """)
        
        # Confirmed links
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS links_confirmed (
                src_id TEXT,
                dst_id TEXT,
                strength DOUBLE,        -- [0..1]
                created_at TIMESTAMP,
                PRIMARY KEY (src_id, dst_id)
            )
        """)
        
        # Suggested links
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS links_suggested (
                src_id TEXT,
                dst_id TEXT,
                score DOUBLE,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                accepted_count INTEGER DEFAULT 0,
                rejected_count INTEGER DEFAULT 0,
                PRIMARY KEY (src_id, dst_id)
            )
        """)
        
        # Audit log for outbound calls
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_outbound (
                id TEXT PRIMARY KEY,
                ts TIMESTAMP,
                destination TEXT,       -- "cloud:provider:model"
                token_count INTEGER,
                categories TEXT,        -- JSON tags
                redactions TEXT         -- JSON summary
            )
        """)
        
        # Create useful view for recent context
        self.conn.execute("""
            CREATE OR REPLACE VIEW context_recent AS
            SELECT id, path, title, ts_modified, 'note' as type
            FROM notes
            UNION ALL
            SELECT id, path, title, ts_modified, 'task' as type
            FROM tasks
            ORDER BY ts_modified DESC
            LIMIT 25
        """)
        
        self.conn.commit()
    
    async def _handle_capture_event(self, event: Event) -> None:
        """Handle capture events to update metadata."""
        try:
            data = event.data
            entry_type = data.get("type", "note")
            
            # We'll get full metadata from file indexing
            # This is just a placeholder for immediate recording
            if entry_type == "task":
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks (id, path, ts_captured, ts_modified, status)
                    VALUES (?, ?, ?, ?, 'inbox')
                    """,
                    [
                        data["id"],
                        data["path"],
                        data["timestamp"],
                        data["timestamp"]
                    ]
                )
            else:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO notes (id, path, ts_captured, ts_modified)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        data["id"],
                        data["path"],
                        data["timestamp"],
                        data["timestamp"]
                    ]
                )
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to handle capture event: {e}")
    
    async def _handle_update_event(self, event: Event) -> None:
        """Handle file update events from watchdog."""
        # This would parse the file and update full metadata
        pass
    
    async def insert_note(self, note_data: Dict[str, Any]) -> None:
        """Insert or update a note."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO notes 
            (id, path, title, project, tags, ts_captured, ts_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                note_data["id"],
                note_data["path"],
                note_data.get("title"),
                note_data.get("project"),
                json.dumps(note_data.get("tags", [])),
                note_data.get("ts_captured"),
                note_data.get("ts_modified", datetime.utcnow())
            ]
        )
        self.conn.commit()
    
    async def insert_task(self, task_data: Dict[str, Any]) -> None:
        """Insert or update a task."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO tasks
            (id, path, title, status, project, energy, effort_min, due, tags, 
             ts_captured, ts_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task_data["id"],
                task_data["path"],
                task_data.get("title"),
                task_data.get("status", "inbox"),
                task_data.get("project"),
                task_data.get("energy", "unknown"),
                task_data.get("effort_min", 15),
                task_data.get("due"),
                json.dumps(task_data.get("tags", [])),
                task_data.get("ts_captured"),
                task_data.get("ts_modified", datetime.utcnow())
            ]
        )
        self.conn.commit()
    
    async def create_receipt(
        self,
        suggestion_text: str,
        sources: List[Dict[str, Any]],
        heuristics: List[str],
        confidence: str = "medium",
        outbound_preview: Optional[List[str]] = None
    ) -> str:
        """
        Create a receipt for explainability.
        Returns receipt ID.
        """
        import ulid
        receipt_id = f"rcp_{ulid.ULID()}"
        
        self.conn.execute(
            """
            INSERT INTO receipts
            (id, created_at, suggestion_text, sources, heuristics, confidence, 
             outbound_preview, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            [
                receipt_id,
                datetime.utcnow(),
                suggestion_text[:200],  # Limit length
                json.dumps(sources),
                json.dumps(heuristics),
                confidence,
                json.dumps(outbound_preview or []),
            ]
        )
        self.conn.commit()
        
        logger.debug(f"Created receipt: {receipt_id}")
        return receipt_id
    
    async def suggest_link(
        self,
        src_id: str,
        dst_id: str,
        score: float,
        reason: Optional[str] = None
    ) -> None:
        """Add or update a suggested link."""
        now = datetime.utcnow()
        
        # Check if suggestion exists
        existing = self.conn.execute(
            "SELECT * FROM links_suggested WHERE src_id = ? AND dst_id = ?",
            [src_id, dst_id]
        ).fetchone()
        
        if existing:
            # Update existing suggestion
            self.conn.execute(
                """
                UPDATE links_suggested
                SET score = ?, last_seen = ?
                WHERE src_id = ? AND dst_id = ?
                """,
                [score, now, src_id, dst_id]
            )
        else:
            # Create new suggestion
            self.conn.execute(
                """
                INSERT INTO links_suggested
                (src_id, dst_id, score, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                [src_id, dst_id, score, now, now]
            )
        
        self.conn.commit()
    
    async def promote_link(self, src_id: str, dst_id: str, strength: float = 1.0) -> None:
        """Promote a suggested link to confirmed."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO links_confirmed
            (src_id, dst_id, strength, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [src_id, dst_id, strength, datetime.utcnow()]
        )
        
        # Update suggestion stats
        self.conn.execute(
            """
            UPDATE links_suggested
            SET accepted_count = accepted_count + 1
            WHERE src_id = ? AND dst_id = ?
            """,
            [src_id, dst_id]
        )
        
        self.conn.commit()
    
    async def reject_link(self, src_id: str, dst_id: str) -> None:
        """Record link rejection."""
        self.conn.execute(
            """
            UPDATE links_suggested
            SET rejected_count = rejected_count + 1
            WHERE src_id = ? AND dst_id = ?
            """,
            [src_id, dst_id]
        )
        self.conn.commit()
    
    async def get_recent_context(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get recent notes and tasks."""
        results = self.conn.execute(
            "SELECT * FROM context_recent LIMIT ?",
            [limit]
        ).fetchall()
        
        return [
            {
                "id": r[0],
                "path": r[1],
                "title": r[2],
                "ts_modified": r[3],
                "type": r[4]
            }
            for r in results
        ]
    
    async def search_by_project(self, project: str) -> List[Dict[str, Any]]:
        """Search notes and tasks by project."""
        notes = self.conn.execute(
            "SELECT * FROM notes WHERE project = ?",
            [project]
        ).fetchall()
        
        tasks = self.conn.execute(
            "SELECT * FROM tasks WHERE project = ?",
            [project]
        ).fetchall()
        
        results = []
        
        for n in notes:
            results.append({
                "id": n[0],
                "path": n[1],
                "title": n[2],
                "type": "note",
                "project": n[3],
                "tags": json.loads(n[4]) if n[4] else []
            })
        
        for t in tasks:
            results.append({
                "id": t[0],
                "path": t[1],
                "title": t[2],
                "type": "task",
                "status": t[3],
                "project": t[4],
                "tags": json.loads(t[8]) if t[8] else []
            })
        
        return results
    
    async def log_outbound(
        self,
        destination: str,
        token_count: int,
        categories: List[str],
        redactions: Dict[str, Any]
    ) -> None:
        """Log an outbound API call for audit."""
        import ulid
        
        self.conn.execute(
            """
            INSERT INTO audit_outbound
            (id, ts, destination, token_count, categories, redactions)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                str(ulid.ULID()),
                datetime.utcnow(),
                destination,
                token_count,
                json.dumps(categories),
                json.dumps(redactions)
            ]
        )
        self.conn.commit()
    
    async def get_link_suggestions(
        self,
        min_score: float = 0.7,
        limit: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Get top link suggestions above threshold."""
        results = self.conn.execute(
            """
            SELECT src_id, dst_id, score
            FROM links_suggested
            WHERE score >= ?
                AND rejected_count < 3
                AND accepted_count = 0
            ORDER BY score DESC
            LIMIT ?
            """,
            [min_score, limit]
        ).fetchall()
        
        return [(r[0], r[1], r[2]) for r in results]
    
    async def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
        logger.info("Analytics indexer closed")