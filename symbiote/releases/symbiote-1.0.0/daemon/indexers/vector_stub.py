"""
Stub implementation of Vector search indexer.
This is a minimal stub to allow daemon startup without heavy dependencies.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from ..algorithms import SearchCandidate
from ..bus import EventBus, Event


class VectorIndexer:
    """Stub Vector search indexer (no actual functionality)."""
    
    def __init__(self, vault_path: Path, event_bus: EventBus, model_name: str = "stub"):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.index_path = vault_path / ".sym" / "vector_index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Track stats
        self.stats = {
            "documents_indexed": 0,
            "last_update": None,
            "index_size_mb": 0,
            "model": "stub"
        }
        
        logger.warning("Using stub VectorIndexer - no actual vector search functionality")
    
    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Stub: pretend to index document."""
        logger.debug(f"Stub: would index document {doc.get('id', 'unknown')}")
        self.stats["documents_indexed"] += 1
        self.stats["last_update"] = datetime.utcnow()
        return True
    
    async def update_document(self, doc_id: str, doc: Dict[str, Any]) -> bool:
        """Stub: pretend to update document."""
        logger.debug(f"Stub: would update document {doc_id}")
        return True
    
    async def delete_document(self, doc_id: str) -> bool:
        """Stub: pretend to delete document."""
        logger.debug(f"Stub: would delete document {doc_id}")
        return True
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        max_results: int = 20,
        timeout_ms: int = 150
    ) -> List[SearchCandidate]:
        """Stub: return empty search results."""
        logger.debug(f"Stub: would search for '{query}' (returning empty)")
        return []
    
    async def find_similar(
        self,
        doc_id: str,
        max_results: int = 10
    ) -> List[SearchCandidate]:
        """Stub: return empty similarity results."""
        logger.debug(f"Stub: would find similar to {doc_id} (returning empty)")
        return []
    
    async def reindex_vault(self) -> int:
        """Stub: pretend to reindex vault."""
        logger.info("Stub: would reindex vault (returning 0)")
        return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get stub indexer statistics."""
        return {
            "type": "vector_stub",
            "documents": self.stats["documents_indexed"],
            "index_size_mb": 0,
            "last_update": self.stats["last_update"].isoformat() if self.stats["last_update"] else None,
            "model": "stub",
            "embedding_dim": 384
        }
    
    async def close(self):
        """Close the stub indexer."""
        logger.info("Stub vector indexer closed")