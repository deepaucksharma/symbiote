"""
Vector search indexer using LanceDB and sentence transformers.
Provides semantic search capabilities with local embeddings.
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..compat import lancedb, SentenceTransformer, logger

from ..algorithms import SearchCandidate
from ..bus import EventBus, Event


class VectorIndexer:
    """Vector search indexer using LanceDB."""
    
    def __init__(self, vault_path: Path, event_bus: EventBus, model_name: str = "all-MiniLM-L6-v2"):
        self.vault_path = vault_path
        self.event_bus = event_bus
        self.index_path = vault_path / ".sym" / "vector_index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model (small, fast, local)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.index_path))
        
        # Create or open table
        self._init_table()
        
        # Track stats
        self.stats = {
            "documents_indexed": 0,
            "last_update": None,
            "index_size_mb": 0,
            "model": model_name
        }
        
        # Subscribe to events
        self._subscribe_events()
    
    def _init_table(self):
        """Initialize the vector table."""
        table_name = "documents"
        
        # Check if table exists
        existing_tables = self.db.table_names()
        
        if table_name not in existing_tables:
            # Create new table with schema
            schema = {
                "id": str,
                "title": str,
                "content": str,
                "path": str,
                "project": Optional[str],
                "tags": Optional[str],
                "modified": float,
                "vector": [float] * self.embedding_dim
            }
            
            # Create empty table
            self.table = self.db.create_table(
                table_name,
                data=[],
                schema=schema
            )
        else:
            self.table = self.db.open_table(table_name)
    
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
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Truncate very long text (model has max input length)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Index a single document."""
        try:
            # Prepare text for embedding
            text_parts = []
            if doc.get("title"):
                text_parts.append(doc["title"])
            if doc.get("content") or doc.get("text"):
                text_parts.append(doc.get("content", doc.get("text", "")))
            if doc.get("tags"):
                tags = doc["tags"]
                if isinstance(tags, list):
                    text_parts.append(" ".join(tags))
                else:
                    text_parts.append(tags)
            
            combined_text = " ".join(text_parts)
            
            # Generate embedding
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._embed_text, combined_text
            )
            
            # Prepare record
            record = {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "content": doc.get("content", doc.get("text", "")),
                "path": doc.get("path", ""),
                "project": doc.get("project"),
                "tags": " ".join(doc["tags"]) if isinstance(doc.get("tags"), list) else doc.get("tags"),
                "modified": doc.get("modified", datetime.utcnow().timestamp()),
                "vector": embedding.tolist()
            }
            
            # Add to table
            self.table.add([record])
            
            self.stats["documents_indexed"] += 1
            self.stats["last_update"] = datetime.utcnow()
            
            logger.debug(f"Vector indexed document: {doc.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vector index document: {e}")
            return False
    
    async def update_document(self, doc_id: str, doc: Dict[str, Any]) -> bool:
        """Update an existing document."""
        # LanceDB doesn't have direct update, so delete + re-add
        await self.delete_document(doc_id)
        return await self.index_document(doc)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        try:
            # Delete by id
            self.table.delete(f'id = "{doc_id}"')
            
            logger.debug(f"Deleted document from vector index: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete from vector index: {e}")
            return False
    
    async def search(
        self,
        query: str,
        project_hint: Optional[str] = None,
        max_results: int = 20,
        timeout_ms: int = 150
    ) -> List[SearchCandidate]:
        """
        Search the vector index using semantic similarity.
        
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
            
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._embed_text, query
            )
            
            # Build search
            search_builder = self.table.search(query_embedding.tolist())
            
            # Add project filter if provided
            if project_hint:
                search_builder = search_builder.where(f'project = "{project_hint}"')
            
            # Execute search
            results = search_builder.limit(max_results).to_pandas()
            
            # Convert to SearchCandidates
            candidates = []
            for _, row in results.iterrows():
                # Calculate similarity score (distance to similarity)
                # LanceDB returns L2 distance, convert to similarity
                distance = row.get("_distance", 0)
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                # Create snippet
                content = row.get("content", "")
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                # Parse modified timestamp
                modified = None
                if "modified" in row:
                    try:
                        modified = datetime.fromtimestamp(row["modified"])
                    except:
                        modified = datetime.utcnow()
                
                candidate = SearchCandidate(
                    id=row.get("id", ""),
                    title=row.get("title", ""),
                    path=row.get("path", ""),
                    snippet=snippet,
                    base_score=min(similarity, 1.0),
                    source="vector",
                    project=row.get("project"),
                    modified=modified
                )
                candidates.append(candidate)
            
            # Check timeout
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Vector search took {elapsed_ms:.1f}ms (timeout: {timeout_ms}ms)")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def find_similar(
        self,
        doc_id: str,
        max_results: int = 10
    ) -> List[SearchCandidate]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document ID to find similar to
            max_results: Maximum results to return
        
        Returns:
            List of similar documents
        """
        try:
            # Get the document
            result = self.table.search().where(f'id = "{doc_id}"').limit(1).to_pandas()
            
            if result.empty:
                return []
            
            # Get its vector
            doc_vector = result.iloc[0]["vector"]
            
            # Search for similar (excluding self)
            results = (
                self.table.search(doc_vector)
                .where(f'id != "{doc_id}"')
                .limit(max_results)
                .to_pandas()
            )
            
            # Convert to candidates
            candidates = []
            for _, row in results.iterrows():
                distance = row.get("_distance", 0)
                similarity = 1.0 / (1.0 + distance)
                
                candidate = SearchCandidate(
                    id=row.get("id", ""),
                    title=row.get("title", ""),
                    path=row.get("path", ""),
                    snippet=row.get("content", "")[:200],
                    base_score=similarity,
                    source="vector",
                    project=row.get("project"),
                    modified=datetime.fromtimestamp(row.get("modified", 0))
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Find similar failed: {e}")
            return []
    
    async def reindex_vault(self) -> int:
        """
        Reindex entire vault from markdown files.
        
        Returns:
            Number of documents indexed
        """
        logger.info("Starting vector reindex of vault...")
        
        # Clear existing data
        self.table.delete("1=1")  # Delete all
        
        indexed = 0
        batch = []
        batch_size = 50
        
        # Index all markdown files
        for md_file in self.vault_path.glob("**/*.md"):
            if md_file.name.startswith("."):
                continue
            
            try:
                # Read file
                content = md_file.read_text(encoding="utf-8")
                
                # Prepare document
                doc = {
                    "id": md_file.stem,
                    "title": md_file.stem.replace("-", " ").title(),
                    "content": content,
                    "path": str(md_file.relative_to(self.vault_path)),
                    "modified": md_file.stat().st_mtime
                }
                
                # Try to extract project from path
                parts = md_file.relative_to(self.vault_path).parts
                if len(parts) > 1:
                    doc["project"] = parts[0]
                
                # Generate embedding
                text = f"{doc['title']} {content[:500]}"
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self._embed_text, text
                )
                
                # Add to batch
                record = {
                    **doc,
                    "vector": embedding.tolist()
                }
                batch.append(record)
                
                # Process batch
                if len(batch) >= batch_size:
                    self.table.add(batch)
                    indexed += len(batch)
                    batch = []
                    logger.info(f"Vector indexed {indexed} documents...")
                
            except Exception as e:
                logger.error(f"Failed to vector index {md_file}: {e}")
        
        # Process remaining batch
        if batch:
            self.table.add(batch)
            indexed += len(batch)
        
        self.stats["last_update"] = datetime.utcnow()
        
        logger.info(f"Vector reindex complete: {indexed} documents")
        return indexed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        # Calculate index size
        index_size = sum(
            f.stat().st_size for f in self.index_path.glob("**/*")
            if f.is_file()
        )
        
        return {
            "type": "vector",
            "documents": self.stats["documents_indexed"],
            "index_size_mb": index_size / (1024 * 1024),
            "last_update": self.stats["last_update"].isoformat() if self.stats["last_update"] else None,
            "model": self.stats["model"],
            "embedding_dim": self.embedding_dim
        }
    
    async def close(self):
        """Close the indexer."""
        try:
            # LanceDB handles cleanup automatically
            logger.info("Vector indexer closed")
            
        except Exception as e:
            logger.error(f"Error closing vector indexer: {e}")