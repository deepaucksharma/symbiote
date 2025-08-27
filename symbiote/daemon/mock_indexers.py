"""Mock implementations for indexers that can be used without external dependencies."""

import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from .models import SearchCandidate


class MockFTSIndexer:
    """Mock FTS indexer that stores documents in memory."""
    
    def __init__(self, vault_path: Path, event_bus=None):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.documents = {}  # id -> document
        self.word_index = {}  # word -> set of doc_ids
        self.stats = {
            "documents_indexed": 0,
            "last_commit": None,
            "index_size_mb": 0
        }
    
    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Index a document in memory."""
        try:
            doc_id = doc.get("id", "")
            content = doc.get("content", doc.get("text", ""))
            title = doc.get("title", "")
            
            # Store document
            self.documents[doc_id] = doc
            
            # Simple word indexing
            words = (title + " " + content).lower().split()
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = set()
                self.word_index[word].add(doc_id)
            
            self.stats["documents_indexed"] += 1
            self.stats["last_commit"] = datetime.utcnow()
            
            logger.debug(f"Mock FTS indexed: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mock index document: {e}")
            return False
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        max_results: int = 20,
        timeout_ms: int = 100
    ) -> List[SearchCandidate]:
        """Search documents by matching words."""
        try:
            start = asyncio.get_event_loop().time()
            
            query_words = query.lower().split()
            matching_docs = set()
            
            # Find docs containing query words
            for word in query_words:
                if word in self.word_index:
                    matching_docs.update(self.word_index[word])
            
            candidates = []
            for doc_id in matching_docs:
                doc = self.documents.get(doc_id)
                if not doc:
                    continue
                
                # Project filter
                if project_hint and doc.get("project") != project_hint:
                    continue
                
                # Simple scoring based on word matches
                content = (doc.get("title", "") + " " + doc.get("content", doc.get("text", ""))).lower()
                score = sum(1 for word in query_words if word in content) / len(query_words)
                
                # Create snippet
                full_text = doc.get("content", doc.get("text", ""))
                snippet = full_text[:200] + "..." if len(full_text) > 200 else full_text
                
                candidate = SearchCandidate(
                    id=doc_id,
                    title=doc.get("title", doc_id),
                    path=doc.get("path", f"{doc_id}.md"),
                    snippet=snippet,
                    base_score=score,
                    source="fts",
                    project=doc.get("project"),
                    modified=doc.get("modified") if isinstance(doc.get("modified"), datetime) else datetime.utcnow()
                )
                candidates.append(candidate)
            
            # Sort by score and limit
            candidates.sort(key=lambda x: x.base_score, reverse=True)
            result = candidates[:max_results]
            
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Mock FTS search took {elapsed_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Mock FTS search failed: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get mock indexer statistics."""
        return {
            "type": "fts",
            "documents": self.stats["documents_indexed"],
            "index_size_mb": len(self.documents) * 0.01,  # Rough estimate
            "last_commit": self.stats["last_commit"].isoformat() if self.stats["last_commit"] else None
        }
    
    async def close(self):
        """Close the mock indexer."""
        logger.info("Mock FTS indexer closed")


class MockVectorIndexer:
    """Mock vector indexer with random similarity scores."""
    
    def __init__(self, vault_path: Path, event_bus=None):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.documents = {}
        self.stats = {
            "documents_indexed": 0,
            "last_update": None,
            "index_size_mb": 0,
            "model": "mock-embeddings"
        }
    
    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Mock index a document."""
        try:
            doc_id = doc.get("id", "")
            self.documents[doc_id] = doc
            self.stats["documents_indexed"] += 1
            self.stats["last_update"] = datetime.utcnow()
            
            logger.debug(f"Mock vector indexed: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mock vector index: {e}")
            return False
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        max_results: int = 20,
        timeout_ms: int = 150
    ) -> List[SearchCandidate]:
        """Mock vector search with random scores."""
        try:
            start = asyncio.get_event_loop().time()
            
            candidates = []
            for doc_id, doc in self.documents.items():
                # Project filter
                if project_hint and doc.get("project") != project_hint:
                    continue
                
                # Random similarity score (biased toward higher scores for demo)
                score = random.uniform(0.3, 0.95)
                
                # Boost score if query appears in title or content
                content = (doc.get("title", "") + " " + doc.get("content", doc.get("text", ""))).lower()
                if query.lower() in content:
                    score = min(0.98, score + 0.2)
                
                snippet = doc.get("content", doc.get("text", ""))[:200]
                
                candidate = SearchCandidate(
                    id=doc_id,
                    title=doc.get("title", doc_id),
                    path=doc.get("path", f"{doc_id}.md"),
                    snippet=snippet,
                    base_score=score,
                    source="vector",
                    project=doc.get("project"),
                    modified=doc.get("modified") if isinstance(doc.get("modified"), datetime) else datetime.utcnow()
                )
                candidates.append(candidate)
            
            # Sort by score and limit
            candidates.sort(key=lambda x: x.base_score, reverse=True)
            result = candidates[:max_results]
            
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Mock vector search took {elapsed_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Mock vector search failed: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get mock statistics."""
        return {
            "type": "vector",
            "documents": self.stats["documents_indexed"],
            "index_size_mb": len(self.documents) * 0.02,
            "last_update": self.stats["last_update"].isoformat() if self.stats["last_update"] else None,
            "model": self.stats["model"]
        }
    
    async def close(self):
        """Close the mock indexer."""
        logger.info("Mock vector indexer closed")


class MockAnalyticsIndexer:
    """Mock analytics indexer using simple in-memory storage."""
    
    def __init__(self, config):
        self.config = config
        self.notes = {}
        self.tasks = {}
        self.receipts = {}
        self.links_suggested = {}
        self.links_confirmed = {}
        self.suggestions = {}
        
    async def initialize(self) -> None:
        """Initialize mock analytics."""
        logger.info("Mock analytics indexer initialized")
    
    async def create_receipt(
        self,
        suggestion_text: str,
        sources: List[Dict[str, Any]],
        heuristics: List[str],
        confidence: str = "medium",
        outbound_preview: Optional[List[str]] = None
    ) -> str:
        """Create a mock receipt."""
        import uuid
        receipt_id = f"rcp_{uuid.uuid4().hex[:8]}"
        
        self.receipts[receipt_id] = {
            "id": receipt_id,
            "created_at": datetime.utcnow(),
            "suggestion_text": suggestion_text,
            "sources": sources,
            "heuristics": heuristics,
            "confidence": confidence,
            "outbound_preview": outbound_preview or []
        }
        
        logger.debug(f"Created mock receipt: {receipt_id}")
        return receipt_id
    
    async def get_recent_context(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get mock recent context."""
        # Generate some mock data
        mock_items = []
        
        for i in range(min(limit, 10)):
            mock_items.append({
                "id": f"item_{i}",
                "title": f"Example item {i}",
                "path": f"notes/item_{i}.md",
                "type": "note" if i % 2 == 0 else "task",
                "project": f"project_{i % 3}" if i < 6 else None,
                "tags": [f"tag_{i % 4}"],
                "modified": datetime.utcnow() - timedelta(hours=i),
                "status": "inbox" if i % 3 == 0 else "active"
            })
        
        return mock_items
    
    async def suggest_link(self, src_id: str, dst_id: str, score: float, reason: Optional[str] = None) -> None:
        """Mock link suggestion."""
        key = f"{src_id}:{dst_id}"
        self.links_suggested[key] = {
            "src_id": src_id,
            "dst_id": dst_id,
            "score": score,
            "reason": reason,
            "created_at": datetime.utcnow()
        }
        logger.debug(f"Mock suggested link: {src_id} -> {dst_id} ({score:.2f})")
    
    async def get_link_suggestions(self, min_score: float = 0.7, limit: int = 10) -> List[tuple]:
        """Get mock link suggestions."""
        suggestions = []
        for link in self.links_suggested.values():
            if link["score"] >= min_score:
                suggestions.append((link["src_id"], link["dst_id"], link["score"]))
        
        return sorted(suggestions, key=lambda x: x[2], reverse=True)[:limit]
    
    async def promote_link(self, src_id: str, dst_id: str, strength: float = 1.0) -> None:
        """Mock promote link."""
        key = f"{src_id}:{dst_id}"
        self.links_confirmed[key] = {
            "src_id": src_id,
            "dst_id": dst_id,
            "strength": strength,
            "created_at": datetime.utcnow()
        }
        logger.debug(f"Mock promoted link: {src_id} -> {dst_id}")
    
    async def reject_link(self, src_id: str, dst_id: str) -> None:
        """Mock reject link."""
        logger.debug(f"Mock rejected link: {src_id} -> {dst_id}")
    
    async def close(self) -> None:
        """Close mock analytics."""
        logger.info("Mock analytics indexer closed")


class SimpleTextWAL:
    """Simple WAL implementation using text files."""
    
    def __init__(self, wal_path: Path):
        self.wal_path = wal_path
        self.wal_path.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.current_date = None
    
    async def append(self, entry: Dict[str, Any]) -> None:
        """Append entry to WAL."""
        today = datetime.utcnow().date()
        
        if self.current_date != today:
            if self.current_file:
                self.current_file.close()
            
            wal_file = self.wal_path / f"{today.isoformat()}.log"
            self.current_file = open(wal_file, 'a')
            self.current_date = today
        
        # Write as JSON line
        line = json.dumps(entry) + "\n"
        self.current_file.write(line)
        self.current_file.flush()
    
    async def close(self):
        """Close WAL file."""
        if self.current_file:
            self.current_file.close()