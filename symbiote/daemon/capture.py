"""WAL-based capture service for lossless, low-latency writes."""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
import aiofiles
import ulid
import frontmatter
from loguru import logger

from .bus import Event, get_event_bus
from .config import Config


CaptureType = Literal["task", "note", "question"]
SourceType = Literal["text", "voice", "clipboard"]


@dataclass
class CaptureEntry:
    """Represents a single capture entry."""
    id: str
    type: CaptureType
    text: str
    source: SourceType = "text"
    context: Optional[str] = None
    captured_at: datetime = None
    
    def __post_init__(self):
        if self.captured_at is None:
            self.captured_at = datetime.utcnow()
        if not self.id:
            self.id = str(ulid.ULID())


class CaptureService:
    """
    Handles capture operations with WAL for durability.
    
    Write path:
    1. Append WAL record (newline-delimited JSON)
    2. Flush + fsync for durability
    3. Materialize to journal and type-specific notes
    4. Emit event for indexers
    
    Performance target: p99 ≤ 200ms from API to WAL fsync
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.vault_path = config.vault_path
        self.wal_path = self.vault_path / ".sym" / "wal"
        self.wal_path.mkdir(parents=True, exist_ok=True)
        
        self._wal_lock = asyncio.Lock()
        self._event_bus = get_event_bus()
        self._wal_file = None
        self._current_wal_date = None
        
    async def initialize(self) -> None:
        """Initialize the capture service and replay WAL if needed."""
        await self._replay_wal()
        logger.info("Capture service initialized")
    
    async def capture(
        self,
        text: str,
        type: CaptureType = "note",
        source: SourceType = "text",
        context: Optional[str] = None
    ) -> CaptureEntry:
        """
        Capture a thought/task/note with WAL durability.
        
        Returns the capture entry with ID for immediate response.
        """
        entry = CaptureEntry(
            id=str(ulid.ULID()),
            type=type,
            text=text,
            source=source,
            context=context
        )
        
        # Write to WAL first (critical path)
        start_time = asyncio.get_event_loop().time()
        await self._write_wal(entry)
        wal_time = asyncio.get_event_loop().time() - start_time
        
        if wal_time > 0.2:  # 200ms threshold
            logger.warning(f"WAL write took {wal_time*1000:.1f}ms")
        
        # Materialize to vault (async, non-blocking)
        asyncio.create_task(self._materialize_entry(entry))
        
        # Emit event for indexers
        await self._event_bus.emit(Event(
            type="capture.written",
            data={
                "id": entry.id,
                "type": entry.type,
                "path": self._get_entry_path(entry),
                "timestamp": entry.captured_at.isoformat()
            },
            source="capture_service"
        ))
        
        logger.debug(f"Captured {entry.type}: {entry.id}")
        return entry
    
    async def _write_wal(self, entry: CaptureEntry) -> None:
        """Write entry to WAL with fsync for durability."""
        async with self._wal_lock:
            wal_file = await self._get_wal_file()
            
            # Create WAL record
            wal_record = {
                "ts": entry.captured_at.isoformat() + "Z",
                "op": "append",
                "type": entry.type,
                "id": entry.id,
                "text": entry.text,
                "source": entry.source,
                "context": entry.context
            }
            
            # Write as newline-delimited JSON
            await wal_file.write(json.dumps(wal_record) + "\n")
            await wal_file.flush()
            
            # fsync for durability
            os.fsync(wal_file.fileno())
    
    async def _get_wal_file(self):
        """Get or create today's WAL file."""
        today = date.today()
        
        if self._current_wal_date != today or self._wal_file is None:
            # Close previous WAL file if open
            if self._wal_file:
                await self._wal_file.close()
            
            # Open new WAL file
            wal_filename = f"{today.isoformat()}.log"
            wal_path = self.wal_path / wal_filename
            self._wal_file = await aiofiles.open(wal_path, 'a')
            self._current_wal_date = today
            
            logger.debug(f"Opened WAL file: {wal_path}")
        
        return self._wal_file
    
    async def _materialize_entry(self, entry: CaptureEntry) -> None:
        """Materialize entry to vault files."""
        try:
            # Write to journal
            await self._append_to_journal(entry)
            
            # Write type-specific note
            if entry.type == "task":
                await self._create_task_note(entry)
            elif entry.type in ["note", "question"]:
                await self._create_note(entry)
            
            logger.debug(f"Materialized entry: {entry.id}")
            
        except Exception as e:
            logger.error(f"Failed to materialize entry {entry.id}: {e}")
    
    async def _append_to_journal(self, entry: CaptureEntry) -> None:
        """Append entry to today's journal."""
        dt = entry.captured_at
        journal_dir = self.vault_path / "journal" / f"{dt.year:04d}" / f"{dt.month:02d}"
        journal_dir.mkdir(parents=True, exist_ok=True)
        
        journal_path = journal_dir / f"{dt.day:02d}.md"
        
        # Create journal header if new file
        if not journal_path.exists():
            header = f"# Journal - {dt.strftime('%Y-%m-%d')}\n\n"
            async with aiofiles.open(journal_path, 'w') as f:
                await f.write(header)
        
        # Append entry
        timestamp = dt.strftime("%H:%M:%S")
        entry_text = f"\n## {timestamp} - {entry.type.title()}\n"
        entry_text += f"{entry.text}\n"
        if entry.context:
            entry_text += f"*Context: {entry.context}*\n"
        
        async with aiofiles.open(journal_path, 'a') as f:
            await f.write(entry_text)
    
    async def _create_task_note(self, entry: CaptureEntry) -> None:
        """Create a task-specific note."""
        task_path = self.vault_path / "tasks" / f"task-{entry.id}.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract title (first line or first 50 chars)
        lines = entry.text.strip().split('\n')
        title = lines[0][:50] if lines else entry.text[:50]
        
        # Create frontmatter
        post = frontmatter.Post(
            content=entry.text,
            metadata={
                "id": entry.id,
                "type": "task",
                "captured": entry.captured_at.isoformat() + "Z",
                "status": "inbox",
                "title": title,
                "project": None,
                "energy": "unknown",
                "effort_min": 15,  # default estimate
                "due": None,
                "tags": self._extract_tags(entry.text),
                "evidence": {
                    "source": entry.source,
                    "context": entry.context,
                    "heuristics": []
                },
                "receipts_id": None
            }
        )
        
        async with aiofiles.open(task_path, 'w') as f:
            await f.write(frontmatter.dumps(post))
    
    async def _create_note(self, entry: CaptureEntry) -> None:
        """Create a note."""
        # Generate slug from first few words
        words = entry.text.split()[:3]
        slug = "-".join(w.lower() for w in words if w.isalnum())[:30]
        
        note_path = self.vault_path / "notes" / f"{slug}-{entry.id}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract title
        lines = entry.text.strip().split('\n')
        title = lines[0][:50] if lines else entry.text[:50]
        
        post = frontmatter.Post(
            content=entry.text,
            metadata={
                "id": entry.id,
                "type": "note",
                "captured": entry.captured_at.isoformat() + "Z",
                "title": title,
                "project": None,
                "tags": self._extract_tags(entry.text),
                "links": []
            }
        )
        
        async with aiofiles.open(note_path, 'w') as f:
            await f.write(frontmatter.dumps(post))
    
    def _extract_tags(self, text: str) -> list:
        """Extract hashtags from text."""
        import re
        tags = re.findall(r'#(\w+)', text)
        return list(set(tags))
    
    def _get_entry_path(self, entry: CaptureEntry) -> str:
        """Get the expected path for an entry."""
        if entry.type == "task":
            return f"tasks/task-{entry.id}.md"
        else:
            # Simplified path generation
            return f"notes/{entry.id}.md"
    
    async def _replay_wal(self) -> None:
        """Replay WAL entries on startup for recovery."""
        wal_files = sorted(self.wal_path.glob("*.log"))
        
        for wal_file in wal_files:
            try:
                async with aiofiles.open(wal_file, 'r') as f:
                    async for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            record = json.loads(line)
                            # Check if entry already materialized
                            entry_id = record.get("id")
                            if entry_id and not self._entry_exists(entry_id):
                                # Recreate and materialize entry
                                entry = CaptureEntry(
                                    id=entry_id,
                                    type=record.get("type", "note"),
                                    text=record.get("text", ""),
                                    source=record.get("source", "text"),
                                    context=record.get("context"),
                                    captured_at=datetime.fromisoformat(
                                        record["ts"].rstrip("Z")
                                    )
                                )
                                await self._materialize_entry(entry)
                                logger.info(f"Replayed WAL entry: {entry_id}")
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid WAL record: {e}")
                            continue
            
            except Exception as e:
                logger.error(f"Failed to replay WAL file {wal_file}: {e}")
    
    def _entry_exists(self, entry_id: str) -> bool:
        """Check if an entry already exists in the vault."""
        # Check tasks
        task_path = self.vault_path / "tasks" / f"task-{entry_id}.md"
        if task_path.exists():
            return True
        
        # Check notes (simplified check)
        for note_path in (self.vault_path / "notes").glob(f"*-{entry_id}.md"):
            if note_path.exists():
                return True
        
        return False
    
    async def close(self) -> None:
        """Close the capture service."""
        if self._wal_file:
            await self._wal_file.close()
        logger.info("Capture service closed")