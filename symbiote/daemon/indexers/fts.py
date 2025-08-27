"""
Full-Text Search indexer using Tantivy.
Provides sub-100ms search over markdown content with real-time updates.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..compat import tantivy, logger

from ..capture import CaptureEntry
from ..algorithms import SearchCandidate
from ..bus import EventBus, Event


class FTSIndexer:
    """Full-text search indexer using Tantivy."""
    
    def __init__(self, vault_path: Path, event_bus: EventBus):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.index_path = vault_path / ".sym" / "fts_index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Define schema
        self.schema_builder = tantivy.SchemaBuilder()
        self.schema_builder.add_text_field("id", stored=True)
        self.schema_builder.add_text_field("title", stored=True)
        self.schema_builder.add_text_field("content", stored=True, tokenizer_name="en_stem")
        self.schema_builder.add_text_field("path", stored=True)
        self.schema_builder.add_text_field("project", stored=True, tokenizer_name="raw")
        self.schema_builder.add_text_field("tags", stored=True)
        self.schema_builder.add_date_field("modified", stored=True)
        self.schema = self.schema_builder.build()
        
        # Initialize or open index
        if not (self.index_path / "meta.json").exists():
            self.index = tantivy.Index(self.schema, path=str(self.index_path))
        else:
            self.index = tantivy.Index.open(str(self.index_path))
        
        self.writer = self.index.writer(heap_size=50_000_000)  # 50MB heap
        self.searcher = None
        self._refresh_searcher()
        
        # Track indexing stats
        self.stats = {
            "documents_indexed": 0,
            "last_commit": None,
            "index_size_mb": 0
        }
        
        # Subscribe to events
        self._subscribe_events()
    
    def _subscribe_events(self):
        """Subscribe to relevant events."""
        self.event_bus.subscribe("capture:complete", self._on_capture)
        self.event_bus.subscribe("document:updated", self._on_document_update)
        self.event_bus.subscribe("document:deleted", self._on_document_delete)
    
    async def _on_capture(self, event: Event):
        """Handle new capture event."""
        entry = event.data
        if isinstance(entry, dict):
            await self.index_document(entry)
    
    async def _on_document_update(self, event: Event):
        """Handle document update."""
        doc = event.data
        await self.update_document(doc["id"], doc)
    
    async def _on_document_delete(self, event: Event):
        """Handle document deletion."""
        doc_id = event.data
        await self.delete_document(doc_id)
    
    def _refresh_searcher(self):
        """Refresh the searcher to see latest changes."""
        self.searcher = self.index.searcher()
    
    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Index a single document."""
        try:
            # Build Tantivy document
            tantivy_doc = tantivy.Document()
            
            # Required fields
            tantivy_doc.add_text("id", doc.get("id", ""))
            tantivy_doc.add_text("title", doc.get("title", ""))
            tantivy_doc.add_text("content", doc.get("content", doc.get("text", "")))
            tantivy_doc.add_text("path", doc.get("path", ""))
            
            # Optional fields
            if "project" in doc:
                tantivy_doc.add_text("project", doc["project"])
            
            if "tags" in doc:
                tags_str = " ".join(doc["tags"]) if isinstance(doc["tags"], list) else doc["tags"]
                tantivy_doc.add_text("tags", tags_str)
            
            if "modified" in doc:
                if isinstance(doc["modified"], str):
                    modified_dt = datetime.fromisoformat(doc["modified"])
                elif isinstance(doc["modified"], (int, float)):
                    modified_dt = datetime.fromtimestamp(doc["modified"])
                else:
                    modified_dt = doc["modified"]
                tantivy_doc.add_date("modified", modified_dt)
            
            # Add to index
            self.writer.add_document(tantivy_doc)
            
            # Commit periodically (every 10 docs for real-time feel)
            self.stats["documents_indexed"] += 1
            if self.stats["documents_indexed"] % 10 == 0:
                await self.commit()
            
            logger.debug(f"Indexed document: {doc.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return False
    
    async def update_document(self, doc_id: str, doc: Dict[str, Any]) -> bool:
        """Update an existing document."""
        # Tantivy doesn't have direct update, so delete + re-add
        await self.delete_document(doc_id)
        return await self.index_document(doc)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        try:
            # Delete by term
            self.writer.delete_term("id", doc_id)
            
            # Commit deletion
            await self.commit()
            
            logger.debug(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def commit(self):
        """Commit pending changes."""
        try:
            self.writer.commit()
            self._refresh_searcher()
            self.stats["last_commit"] = datetime.utcnow()
            
            # Calculate index size
            index_size = sum(
                f.stat().st_size for f in self.index_path.glob("*")
                if f.is_file()
            )
            self.stats["index_size_mb"] = index_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Failed to commit index: {e}")
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        max_results: int = 20,
        timeout_ms: int = 100
    ) -> List[SearchCandidate]:
        """
        Search the FTS index.
        
        Args:
            query: Search query
            project_hint: Optional project filter
            max_results: Maximum results to return
            timeout_ms: Timeout in milliseconds
        
        Returns:
            List of SearchCandidate objects
        """
        try:
            start = asyncio.get_event_loop().time()
            
            # Build query
            query_parser = tantivy.QueryParser.for_index(
                self.index,
                ["title", "content", "tags"]
            )
            
            # Add project filter if provided
            if project_hint:
                full_query = f'({query}) AND project:"{project_hint}"'
            else:
                full_query = query
            
            parsed_query = query_parser.parse_query(full_query)
            
            # Execute search
            search_result = self.searcher.search(
                parsed_query,
                limit=max_results
            )
            
            # Convert to SearchCandidates
            candidates = []
            for score, doc_addr in search_result.hits:
                doc = self.searcher.doc(doc_addr)
                
                # Extract fields
                doc_dict = {}
                for field_name, field_values in doc.items():
                    if field_values:
                        doc_dict[field_name] = field_values[0]
                
                # Create snippet (first 200 chars of content)
                content = doc_dict.get("content", "")
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                # Parse modified date
                modified = None
                if "modified" in doc_dict:
                    try:
                        modified = datetime.fromisoformat(doc_dict["modified"])
                    except:
                        modified = datetime.utcnow()
                
                candidate = SearchCandidate(
                    id=doc_dict.get("id", ""),
                    title=doc_dict.get("title", ""),
                    path=doc_dict.get("path", ""),
                    snippet=snippet,
                    base_score=min(score / 10.0, 1.0),  # Normalize score
                    source="fts",
                    project=doc_dict.get("project"),
                    modified=modified
                )
                candidates.append(candidate)
            
            # Check timeout
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"FTS search took {elapsed_ms:.1f}ms (timeout: {timeout_ms}ms)")
            
            return candidates
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
    
    async def reindex_vault(self) -> int:
        """
        Reindex entire vault from markdown files.
        
        Returns:
            Number of documents indexed
        """
        logger.info("Starting FTS reindex of vault...")
        
        # Clear existing index
        self.writer = self.index.writer(heap_size=50_000_000)
        
        indexed = 0
        
        # Index all markdown files
        for md_file in self.vault_path.glob("**/*.md"):
            if md_file.name.startswith("."):
                continue
            
            try:
                # Read file
                content = md_file.read_text(encoding="utf-8")
                
                # Extract frontmatter if present
                doc = {
                    "id": md_file.stem,
                    "title": md_file.stem.replace("-", " ").title(),
                    "content": content,
                    "path": str(md_file.relative_to(self.vault_path)),
                    "modified": datetime.fromtimestamp(md_file.stat().st_mtime)
                }
                
                # Try to extract project from path
                parts = md_file.relative_to(self.vault_path).parts
                if len(parts) > 1:
                    doc["project"] = parts[0]
                
                # Index document
                if await self.index_document(doc):
                    indexed += 1
                
            except Exception as e:
                logger.error(f"Failed to index {md_file}: {e}")
        
        # Final commit
        await self.commit()
        
        logger.info(f"FTS reindex complete: {indexed} documents")
        return indexed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "type": "fts",
            "documents": self.stats["documents_indexed"],
            "index_size_mb": self.stats["index_size_mb"],
            "last_commit": self.stats["last_commit"].isoformat() if self.stats["last_commit"] else None,
            "searcher_generation": self.searcher.num_docs() if self.searcher else 0
        }
    
    async def close(self):
        """Close the indexer."""
        try:
            # Final commit
            await self.commit()
            
            # Close writer
            self.writer = None
            
            logger.info("FTS indexer closed")
            
        except Exception as e:
            logger.error(f"Error closing FTS indexer: {e}")