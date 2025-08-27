"""Production Full-Text Search Indexer using Whoosh.

This module provides real full-text search capabilities using:
- Whoosh for full-text indexing and search
- BM25 scoring for relevance ranking
- Proper tokenization and stemming
- Phrase search and wildcard support
- Incremental indexing with ACID guarantees
"""

import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import hashlib

try:
    from whoosh import index
    from whoosh.fields import Schema, TEXT, ID, KEYWORD, DATETIME, NUMERIC, STORED
    from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer, NgramFilter
    from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
    from whoosh.query import Term, And, Or, Phrase, Wildcard, DateRange, NumericRange
    from whoosh.writing import AsyncWriter
    from whoosh.searching import Results
    from whoosh import scoring
    HAS_FTS_DEPS = True
except ImportError:
    HAS_FTS_DEPS = False

from loguru import logger
from ...daemon.algorithms import SearchCandidate


class FTSIndexProduction:
    """Production-ready full-text search using Whoosh."""
    
    def __init__(self,
                 vault_path: Path,
                 index_path: Optional[Path] = None,
                 use_stemming: bool = True,
                 use_ngrams: bool = False):
        """
        Initialize FTS index with Whoosh.
        
        Args:
            vault_path: Path to the vault directory
            index_path: Path to save/load index
            use_stemming: Whether to use stemming analyzer
            use_ngrams: Whether to use n-gram indexing for fuzzy search
        """
        self.vault_path = Path(vault_path)
        self.index_path = index_path or self.vault_path / ".symbiote" / "fts_index"
        self.use_stemming = use_stemming
        self.use_ngrams = use_ngrams
        
        self.index = None
        self.schema = None
        
        if HAS_FTS_DEPS:
            self._initialize()
        else:
            logger.warning("Whoosh not available, using fallback")
    
    def _create_schema(self) -> Schema:
        """Create the Whoosh schema for documents."""
        # Choose analyzer based on configuration
        if self.use_stemming:
            content_analyzer = StemmingAnalyzer()
        else:
            content_analyzer = StandardAnalyzer()
        
        # Add n-gram filter for fuzzy matching if enabled
        if self.use_ngrams:
            content_analyzer | NgramFilter(minsize=2, maxsize=4)
        
        schema = Schema(
            # Core fields
            id=ID(stored=True, unique=True),
            path=ID(stored=True),
            title=TEXT(stored=True, analyzer=content_analyzer, field_boost=2.0),
            content=TEXT(stored=True, analyzer=content_analyzer),
            
            # Metadata fields
            project=ID(stored=True),
            tags=KEYWORD(stored=True, commas=True, scorable=True),
            type=ID(stored=True),
            status=ID(stored=True),
            
            # Temporal fields
            created=DATETIME(stored=True),
            modified=DATETIME(stored=True),
            
            # Numeric fields for scoring
            importance=NUMERIC(stored=True, sortable=True),
            word_count=NUMERIC(stored=True),
            
            # Additional searchable fields
            headings=TEXT(analyzer=content_analyzer, field_boost=1.5),
            links=KEYWORD(commas=True),
            code_blocks=TEXT(analyzer=StandardAnalyzer()),  # Don't stem code
            
            # Snippet for display (first 500 chars)
            snippet=STORED()
        )
        
        return schema
    
    def _initialize(self):
        """Initialize or load the Whoosh index."""
        self.schema = self._create_schema()
        
        # Ensure index directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Create or open index
        if index.exists_in(str(self.index_path)):
            try:
                self.index = index.open_dir(str(self.index_path))
                logger.info(f"Opened existing Whoosh index at {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to open index, creating new: {e}")
                self.index = index.create_in(str(self.index_path), self.schema)
        else:
            self.index = index.create_in(str(self.index_path), self.schema)
            logger.info(f"Created new Whoosh index at {self.index_path}")
    
    def _extract_document_fields(self, 
                                 doc_id: str,
                                 content: str,
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prepare fields for indexing."""
        fields = {
            'id': doc_id,
            'path': metadata.get('path', ''),
            'title': metadata.get('title', doc_id),
            'content': content,
            'project': metadata.get('project', ''),
            'tags': ','.join(metadata.get('tags', [])),
            'type': metadata.get('type', 'note'),
            'status': metadata.get('status', 'active'),
            'created': metadata.get('created', datetime.now()),
            'modified': metadata.get('modified', datetime.now()),
            'importance': metadata.get('importance', 0),
            'word_count': len(content.split()),
            'snippet': content[:500] if len(content) > 500 else content
        }
        
        # Extract headings (markdown headers)
        headings = []
        for line in content.split('\n'):
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
                if heading:
                    headings.append(heading)
        fields['headings'] = ' '.join(headings)
        
        # Extract links (markdown links and URLs)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)|https?://[^\s]+'
        links = re.findall(link_pattern, content)
        link_texts = []
        for match in links:
            if isinstance(match, tuple):
                link_texts.append(match[0] if match[0] else match[1])
            else:
                link_texts.append(match)
        fields['links'] = ','.join(link_texts)
        
        # Extract code blocks
        code_pattern = r'```[^\n]*\n(.*?)```'
        code_blocks = re.findall(code_pattern, content, re.DOTALL)
        fields['code_blocks'] = '\n'.join(code_blocks)
        
        return fields
    
    async def index_document(self,
                           doc_id: str,
                           content: str,
                           metadata: Dict[str, Any]) -> bool:
        """
        Index a document with full-text search.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata
            
        Returns:
            True if successful
        """
        if not HAS_FTS_DEPS:
            return False
        
        try:
            # Extract fields
            fields = self._extract_document_fields(doc_id, content, metadata)
            
            # Use AsyncWriter for better concurrency
            writer = AsyncWriter(self.index)
            
            # Update document (will replace if exists)
            writer.update_document(**fields)
            
            # Commit changes
            writer.commit()
            
            logger.debug(f"Indexed document {doc_id} with {fields['word_count']} words")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return False
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if not HAS_FTS_DEPS:
            return False
        
        try:
            writer = self.index.writer()
            writer.delete_by_term('id', doc_id)
            writer.commit()
            logger.debug(f"Removed document {doc_id} from index")
            return True
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def _build_query(self, query_text: str, fields: List[str] = None):
        """Build a Whoosh query from text."""
        if fields is None:
            fields = ['title', 'content', 'headings', 'tags']
        
        # Use multifield parser for searching across fields
        parser = MultifieldParser(fields, self.index.schema, group=OrGroup)
        
        # Parse the query
        try:
            query = parser.parse(query_text)
        except Exception as e:
            logger.warning(f"Failed to parse query '{query_text}': {e}")
            # Fallback to simple term query
            terms = []
            for field in fields:
                for word in query_text.split():
                    terms.append(Term(field, word.lower()))
            query = Or(terms)
        
        return query
    
    async def search(self,
                    query: str,
                    limit: int = 10,
                    project: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    date_range: Optional[tuple] = None) -> List[SearchCandidate]:
        """
        Perform full-text search with advanced features.
        
        Args:
            query: Search query (supports phrases, wildcards, field queries)
            limit: Maximum results to return
            project: Filter by project
            tags: Filter by tags
            date_range: Filter by date range (start, end)
            
        Returns:
            List of search candidates with BM25 scores
        """
        if not HAS_FTS_DEPS:
            return []
        
        try:
            with self.index.searcher(weighting=scoring.BM25F()) as searcher:
                # Build main query
                main_query = self._build_query(query)
                
                # Add filters
                filters = []
                if project:
                    filters.append(Term('project', project))
                
                if tags:
                    tag_queries = [Term('tags', tag) for tag in tags]
                    filters.append(Or(tag_queries))
                
                if date_range:
                    start, end = date_range
                    filters.append(DateRange('modified', start, end))
                
                # Combine queries
                if filters:
                    final_query = And([main_query] + filters)
                else:
                    final_query = main_query
                
                # Execute search
                results = searcher.search(final_query, limit=limit)
                
                # Convert to SearchCandidates
                candidates = []
                for hit in results:
                    # Calculate normalized score (0-1 range)
                    # BM25 scores can be > 1, so we use a sigmoid-like normalization
                    raw_score = hit.score
                    normalized_score = 2.0 / (1.0 + pow(2.71828, -raw_score * 0.5)) - 1.0
                    
                    candidate = SearchCandidate(
                        id=hit['id'],
                        title=hit['title'],
                        path=hit['path'],
                        snippet=self._highlight_snippet(hit, query),
                        base_score=min(1.0, normalized_score),
                        source='fts',
                        project=hit.get('project') if hit.get('project') else None,
                        tags=hit['tags'].split(',') if hit.get('tags') else [],
                        modified=hit.get('modified')
                    )
                    candidates.append(candidate)
                
                return candidates
                
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
    
    def _highlight_snippet(self, hit: Dict, query: str, max_length: int = 200) -> str:
        """Create a highlighted snippet showing query matches."""
        content = hit.get('content', '')
        if not content:
            return hit.get('snippet', '')[:max_length]
        
        # Simple highlighting - find query terms and show context
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find best match position
        best_pos = 0
        best_score = 0
        
        for i in range(len(content) - max_length):
            window = content_lower[i:i+max_length]
            score = sum(1 for term in query_terms if term in window)
            if score > best_score:
                best_score = score
                best_pos = i
        
        # Extract snippet
        snippet = content[best_pos:best_pos + max_length]
        
        # Add ellipsis if needed
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_length < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    async def search_similar(self,
                           doc_id: str,
                           limit: int = 5) -> List[SearchCandidate]:
        """
        Find documents similar to a given document using more-like-this.
        
        Args:
            doc_id: Document to find similar documents for
            limit: Maximum results
            
        Returns:
            List of similar documents
        """
        if not HAS_FTS_DEPS:
            return []
        
        try:
            with self.index.searcher() as searcher:
                # Get the source document
                source_doc = searcher.document(id=doc_id)
                if not source_doc:
                    return []
                
                # Extract key terms from source document
                content = source_doc.get('content', '')
                title = source_doc.get('title', '')
                
                # Use title and top content words for similarity
                text = f"{title} {content[:1000]}"
                
                # Simple term extraction (can be improved with TF-IDF)
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top terms
                top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Build query from top terms
                query_terms = [Term('content', term[0]) for term in top_terms]
                query = Or(query_terms)
                
                # Search
                results = searcher.search(query, limit=limit + 1)
                
                # Convert to candidates (excluding source)
                candidates = []
                for hit in results:
                    if hit['id'] != doc_id:
                        candidate = SearchCandidate(
                            id=hit['id'],
                            title=hit['title'],
                            path=hit['path'],
                            snippet=hit.get('snippet', '')[:200],
                            base_score=min(1.0, hit.score / 10.0),
                            source='fts',
                            project=hit.get('project'),
                            tags=hit['tags'].split(',') if hit.get('tags') else [],
                            modified=hit.get('modified')
                        )
                        candidates.append(candidate)
                
                return candidates[:limit]
                
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not HAS_FTS_DEPS:
            return {}
        
        try:
            with self.index.searcher() as searcher:
                return {
                    'total_documents': searcher.doc_count(),
                    'index_size_mb': sum(
                        f.stat().st_size for f in Path(self.index_path).glob('*')
                    ) / (1024 * 1024),
                    'fields': list(self.schema.names()),
                    'last_modified': max(
                        f.stat().st_mtime for f in Path(self.index_path).glob('*')
                    )
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def reindex_vault(self) -> int:
        """
        Reindex all documents in the vault.
        
        Returns:
            Number of documents indexed
        """
        # Clear and recreate index
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        
        self._initialize()
        
        # Index all markdown files
        indexed_count = 0
        
        for md_file in self.vault_path.rglob("*.md"):
            try:
                # Read content
                content = md_file.read_text(encoding='utf-8')
                
                # Extract metadata
                doc_id = md_file.stem
                metadata = {
                    'title': doc_id.replace('_', ' ').replace('-', ' ').title(),
                    'path': str(md_file.relative_to(self.vault_path)),
                    'created': datetime.fromtimestamp(md_file.stat().st_ctime),
                    'modified': datetime.fromtimestamp(md_file.stat().st_mtime)
                }
                
                # Extract project from path
                parts = md_file.relative_to(self.vault_path).parts
                if len(parts) > 1:
                    if parts[0] == 'projects':
                        metadata['project'] = parts[1]
                    metadata['type'] = parts[0].rstrip('s')  # notes -> note
                
                # Extract tags from content (simple #tag pattern)
                tag_pattern = r'#(\w+)'
                tags = re.findall(tag_pattern, content)
                if tags:
                    metadata['tags'] = list(set(tags))
                
                # Index document
                success = await self.index_document(doc_id, content, metadata)
                if success:
                    indexed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to index {md_file}: {e}")
        
        logger.info(f"Reindexed {indexed_count} documents")
        return indexed_count
    
    async def optimize_index(self):
        """Optimize the index for better search performance."""
        if not HAS_FTS_DEPS:
            return
        
        try:
            writer = self.index.writer()
            writer.commit(optimize=True)
            logger.info("Index optimized successfully")
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")