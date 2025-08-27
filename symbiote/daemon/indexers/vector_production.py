"""Production Vector Search Indexer using Sentence Transformers and FAISS.

This module provides real semantic search capabilities using:
- Sentence transformers for generating semantic embeddings
- FAISS for efficient similarity search
- Proper chunking and document processing
- Incremental indexing with persistence
"""

import asyncio
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False

from loguru import logger
from ...daemon.bus import EventBus, Event
from ...daemon.algorithms import SearchCandidate


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        """Create from dictionary."""
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        return cls(**data)


class ChunkingStrategy:
    """Smart document chunking with overlap and context preservation."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 128,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks while preserving sentence boundaries.
        """
        chunks = []
        
        # Split into sentences (simple approach, can be improved with NLTK)
        sentences = text.replace('\n\n', '\n').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=-1,  # Will be updated later
                    metadata=metadata
                ))
                
                # Keep overlap for next chunk
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    sent_size = len(sent.split())
                    if overlap_size + sent_size <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += sent_size
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk if it meets minimum size
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                total_chunks=-1,
                metadata=metadata
            ))
        
        # Update total chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks


class VectorIndexProduction:
    """Production-ready vector search using FAISS and sentence transformers."""
    
    def __init__(self,
                 vault_path: Path,
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: Optional[Path] = None,
                 dimension: int = 384,
                 use_gpu: bool = False):
        """
        Initialize vector index with real embeddings.
        
        Args:
            vault_path: Path to the vault directory
            model_name: Sentence transformer model to use
            index_path: Path to save/load index
            dimension: Embedding dimension (must match model)
            use_gpu: Whether to use GPU for embeddings and search
        """
        self.vault_path = Path(vault_path)
        self.index_path = index_path or self.vault_path / ".symbiote" / "vector.idx"
        self.metadata_path = self.index_path.with_suffix('.meta')
        self.dimension = dimension
        self.use_gpu = use_gpu and HAS_VECTOR_DEPS
        
        # Initialize model and index
        self.model = None
        self.index = None
        self.chunk_metadata: Dict[int, DocumentChunk] = {}
        self.doc_to_chunks: Dict[str, List[int]] = {}
        self.chunker = ChunkingStrategy()
        
        if HAS_VECTOR_DEPS:
            self._initialize()
        else:
            logger.warning("Vector dependencies not available, using fallback")
    
    def _initialize(self):
        """Initialize model and index."""
        # Load embedding model
        device = 'cuda' if self.use_gpu else 'cpu'
        self.model = SentenceTransformer(self.model_name, device=device)
        
        # Initialize or load FAISS index
        if self.index_path.exists():
            self._load_index()
        else:
            # Create new index with inner product (cosine similarity after normalization)
            self.index = faiss.IndexFlatIP(self.dimension)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _load_index(self):
        """Load existing index from disk."""
        try:
            # Load FAISS index
            cpu_index = faiss.read_index(str(self.index_path))
            
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.chunk_metadata = {
                        idx: DocumentChunk.from_dict(chunk_dict)
                        for idx, chunk_dict in saved_data['chunks'].items()
                    }
                    self.doc_to_chunks = saved_data['doc_mapping']
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _save_index(self):
        """Save index to disk."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index (convert to CPU first if on GPU)
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(self.index_path))
            else:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            saved_data = {
                'chunks': {
                    idx: chunk.to_dict()
                    for idx, chunk in self.chunk_metadata.items()
                },
                'doc_mapping': self.doc_to_chunks
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(saved_data, f)
            
            logger.debug(f"Saved index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer."""
        if not self.model:
            # Fallback to random for testing
            return np.random.randn(self.dimension).astype(np.float32)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
    
    async def index_document(self, 
                           doc_id: str,
                           content: str,
                           metadata: Dict[str, Any]) -> int:
        """
        Index a document by chunking and embedding it.
        
        Returns:
            Number of chunks indexed
        """
        if not HAS_VECTOR_DEPS:
            return 0
        
        try:
            # Remove existing chunks for this document
            await self.remove_document(doc_id)
            
            # Chunk the document
            chunks = self.chunker.chunk_text(content, doc_id, metadata)
            
            if not chunks:
                return 0
            
            # Generate embeddings for all chunks
            embeddings = []
            chunk_indices = []
            
            for chunk in chunks:
                embedding = self._generate_embedding(chunk.content)
                chunk.embedding = embedding
                embeddings.append(embedding)
                
                # Store chunk metadata
                idx = self.index.ntotal + len(embeddings) - 1
                self.chunk_metadata[idx] = chunk
                chunk_indices.append(idx)
            
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            
            # Update document to chunks mapping
            self.doc_to_chunks[doc_id] = chunk_indices
            
            # Save periodically
            if self.index.ntotal % 100 == 0:
                self._save_index()
            
            logger.debug(f"Indexed {len(chunks)} chunks for document {doc_id}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return 0
    
    async def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and all its chunks from the index.
        
        Note: FAISS doesn't support deletion, so we mark as deleted
        and periodically rebuild the index.
        """
        if doc_id in self.doc_to_chunks:
            # Mark chunks as deleted (set to zero vector)
            for idx in self.doc_to_chunks[doc_id]:
                if idx in self.chunk_metadata:
                    del self.chunk_metadata[idx]
            
            del self.doc_to_chunks[doc_id]
            
            # TODO: Periodically rebuild index to actually remove vectors
            return True
        
        return False
    
    async def search(self,
                    query: str,
                    limit: int = 10,
                    threshold: float = 0.5) -> List[SearchCandidate]:
        """
        Perform semantic search using real embeddings.
        
        Args:
            query: Search query
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search candidates with similarity scores
        """
        if not HAS_VECTOR_DEPS or self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            query_vector = query_embedding.reshape(1, -1)
            
            # Search in FAISS
            k = min(limit * 3, self.index.ntotal)  # Search more to account for grouping
            distances, indices = self.index.search(query_vector, k)
            
            # Group results by document
            doc_scores: Dict[str, List[Tuple[float, DocumentChunk]]] = {}
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx not in self.chunk_metadata:
                    continue
                
                chunk = self.chunk_metadata[idx]
                score = float(dist)  # Cosine similarity (after normalization)
                
                if score >= threshold:
                    if chunk.doc_id not in doc_scores:
                        doc_scores[chunk.doc_id] = []
                    doc_scores[chunk.doc_id].append((score, chunk))
            
            # Create search candidates from top chunks per document
            candidates = []
            
            for doc_id, chunk_results in doc_scores.items():
                # Sort chunks by score and take best
                chunk_results.sort(key=lambda x: x[0], reverse=True)
                best_score, best_chunk = chunk_results[0]
                
                # Create snippet from best matching chunk
                snippet = best_chunk.content[:200]
                if len(best_chunk.content) > 200:
                    snippet += "..."
                
                candidate = SearchCandidate(
                    id=doc_id,
                    title=best_chunk.metadata.get('title', doc_id),
                    path=best_chunk.metadata.get('path', ''),
                    snippet=snippet,
                    base_score=best_score,
                    source='vector',
                    project=best_chunk.metadata.get('project'),
                    tags=best_chunk.metadata.get('tags', []),
                    modified=best_chunk.metadata.get('modified')
                )
                
                candidates.append(candidate)
            
            # Sort by score and return top results
            candidates.sort(key=lambda x: x.base_score, reverse=True)
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def find_similar(self,
                          doc_id: str,
                          limit: int = 5) -> List[SearchCandidate]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document ID to find similar documents for
            limit: Maximum results to return
            
        Returns:
            List of similar documents
        """
        if doc_id not in self.doc_to_chunks:
            return []
        
        # Get embeddings for all chunks of this document
        chunk_indices = self.doc_to_chunks[doc_id]
        if not chunk_indices:
            return []
        
        # Use the first chunk's embedding as representative
        # (could be improved by averaging all chunks)
        first_chunk = self.chunk_metadata.get(chunk_indices[0])
        if not first_chunk or first_chunk.embedding is None:
            return []
        
        # Search using the document's embedding
        query_vector = first_chunk.embedding.reshape(1, -1)
        k = min(limit * 3, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Group and filter results (excluding self)
        doc_scores: Dict[str, float] = {}
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx not in self.chunk_metadata:
                continue
            
            chunk = self.chunk_metadata[idx]
            if chunk.doc_id == doc_id:
                continue  # Skip self
            
            score = float(dist)
            if chunk.doc_id not in doc_scores or score > doc_scores[chunk.doc_id]:
                doc_scores[chunk.doc_id] = score
        
        # Create candidates
        candidates = []
        for similar_doc_id, score in doc_scores.items():
            # Get metadata from first chunk
            similar_chunks = self.doc_to_chunks.get(similar_doc_id, [])
            if similar_chunks:
                chunk = self.chunk_metadata.get(similar_chunks[0])
                if chunk:
                    candidate = SearchCandidate(
                        id=similar_doc_id,
                        title=chunk.metadata.get('title', similar_doc_id),
                        path=chunk.metadata.get('path', ''),
                        snippet=chunk.content[:200],
                        base_score=score,
                        source='vector',
                        project=chunk.metadata.get('project'),
                        tags=chunk.metadata.get('tags', []),
                        modified=chunk.metadata.get('modified')
                    )
                    candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.base_score, reverse=True)
        return candidates[:limit]
    
    async def reindex_vault(self) -> int:
        """
        Reindex all documents in the vault.
        
        Returns:
            Number of documents indexed
        """
        # Clear existing index
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.use_gpu and HAS_VECTOR_DEPS:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.chunk_metadata.clear()
        self.doc_to_chunks.clear()
        
        # Index all markdown files in vault
        indexed_count = 0
        
        for md_file in self.vault_path.rglob("*.md"):
            try:
                # Read content
                content = md_file.read_text(encoding='utf-8')
                
                # Extract basic metadata
                doc_id = md_file.stem
                metadata = {
                    'title': doc_id.replace('_', ' ').title(),
                    'path': str(md_file.relative_to(self.vault_path)),
                    'modified': datetime.fromtimestamp(md_file.stat().st_mtime)
                }
                
                # Extract project from path if in project folder
                parts = md_file.relative_to(self.vault_path).parts
                if len(parts) > 1 and parts[0] == 'projects':
                    metadata['project'] = parts[1]
                
                # Index document
                chunks = await self.index_document(doc_id, content, metadata)
                if chunks > 0:
                    indexed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to index {md_file}: {e}")
        
        # Save final index
        self._save_index()
        
        logger.info(f"Reindexed {indexed_count} documents with {self.index.ntotal} chunks")
        return indexed_count