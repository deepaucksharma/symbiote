"""
Mock implementations for optional dependencies to enable testing without heavy dependencies.
These mocks provide the same interfaces as the real packages but with simplified functionality.
"""

import asyncio
import json
import random
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path


# Mock pydantic
class BaseModel:
    """Mock pydantic BaseModel for configuration."""
    
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
    
    def model_dump(self) -> Dict[str, Any]:
        """Return dict of model data."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def field_validator(cls, field_name: str):
        """Mock field validator decorator."""
        def decorator(func):
            return func
        return decorator


def Field(**kwargs):
    """Mock pydantic Field function."""
    return kwargs.get('default_factory', lambda: None)()


# Mock aiofiles
class MockAsyncFile:
    """Mock async file object."""
    
    def __init__(self, path: Path, mode: str):
        self.path = path
        self.mode = mode
        self._closed = False
        
    async def write(self, data: str) -> int:
        """Mock write operation."""
        if 'w' in self.mode or 'a' in self.mode:
            with open(self.path, self.mode.replace('a', 'a').replace('w', 'w')) as f:
                return f.write(data)
        return len(data)
    
    async def read(self) -> str:
        """Mock read operation."""
        try:
            with open(self.path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
    
    async def flush(self):
        """Mock flush operation."""
        pass
    
    async def close(self):
        """Mock close operation."""
        self._closed = True
    
    def fileno(self) -> int:
        """Mock file descriptor."""
        return 1
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class MockAioFiles:
    """Mock aiofiles module."""
    
    @staticmethod
    def open(path: Union[str, Path], mode: str = 'r'):
        """Mock aiofiles.open()."""
        return MockAsyncFile(Path(path), mode)


# Mock frontmatter
class Post:
    """Mock frontmatter Post object."""
    
    def __init__(self, content: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}


def loads(text: str) -> Post:
    """Mock frontmatter.loads()."""
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            try:
                import yaml
                metadata = yaml.safe_load(parts[1]) or {}
                content = parts[2].strip()
            except:
                metadata = {}
                content = text
        else:
            metadata = {}
            content = text
    else:
        metadata = {}
        content = text
    
    return Post(content=content, metadata=metadata)


def dumps(post: Post) -> str:
    """Mock frontmatter.dumps()."""
    if post.metadata:
        import yaml
        yaml_front = yaml.dump(post.metadata, default_flow_style=False)
        return f"---\n{yaml_front}---\n{post.content}"
    return post.content


# Mock ULID
class ULID:
    """Mock ULID class."""
    
    def __init__(self):
        self._value = f"01{random.randint(100000000000000000000000, 999999999999999999999999):024d}"
    
    def __str__(self):
        return self._value


# Mock tantivy
class MockTantivyIndex:
    """Mock Tantivy index."""
    
    def __init__(self, schema, path: Optional[Path] = None):
        self.schema = schema
        self.path = path
        self._docs = []
    
    def add_document(self, doc: Dict[str, Any]):
        """Add document to mock index."""
        self._docs.append(doc)
    
    def commit(self):
        """Mock commit operation."""
        pass
    
    def searcher(self):
        """Return mock searcher."""
        return MockTantivySearcher(self._docs)
    
    def reload_searchers(self):
        """Mock reload operation."""
        pass


class MockTantivySearcher:
    """Mock Tantivy searcher."""
    
    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs
    
    def search(self, query, limit: int = 10):
        """Mock search operation."""
        # Simple text matching
        results = []
        query_text = str(query).lower()
        
        for i, doc in enumerate(self._docs):
            score = 0.5  # Default score
            for field, value in doc.items():
                if query_text in str(value).lower():
                    score = 0.8
                    break
            
            if score > 0.5:
                results.append((score, i))
        
        # Sort by score and limit
        results.sort(reverse=True)
        return results[:limit]


class MockTantivyQuery:
    """Mock Tantivy query."""
    
    def __init__(self, text: str):
        self.text = text
    
    def __str__(self):
        return self.text


class MockTantivySchema:
    """Mock Tantivy schema."""
    
    def __init__(self):
        self.fields = {}
    
    def add_text_field(self, name: str, **kwargs):
        """Add text field to schema."""
        self.fields[name] = "text"
    
    def add_id_field(self, name: str, **kwargs):
        """Add ID field to schema."""
        self.fields[name] = "id"


def Schema():
    """Create mock Tantivy schema."""
    return MockTantivySchema()


def Index(schema, path: Optional[Path] = None):
    """Create mock Tantivy index."""
    return MockTantivyIndex(schema, path)


class QueryParser:
    """Mock Tantivy query parser."""
    
    def __init__(self, schema, fields: List[str]):
        self.schema = schema
        self.fields = fields
    
    def parse_query(self, query_str: str):
        """Parse query string."""
        return MockTantivyQuery(query_str)


# Mock lancedb
class MockLanceDBTable:
    """Mock LanceDB table."""
    
    def __init__(self, name: str):
        self.name = name
        self._data = []
    
    def add(self, data: List[Dict[str, Any]]):
        """Add data to table."""
        self._data.extend(data)
    
    def search(self, query_vector: List[float], limit: int = 10):
        """Mock vector search."""
        # Return random subset of data
        results = random.sample(self._data, min(len(self._data), limit))
        return MockLanceDBResults(results)


class MockLanceDBResults:
    """Mock LanceDB search results."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self._data = data
    
    def to_list(self):
        """Convert to list."""
        return self._data


class MockLanceDB:
    """Mock LanceDB connection."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.tables = {}
    
    def create_table(self, name: str, data: List[Dict[str, Any]] = None):
        """Create table."""
        table = MockLanceDBTable(name)
        if data:
            table.add(data)
        self.tables[name] = table
        return table
    
    def open_table(self, name: str):
        """Open existing table."""
        if name not in self.tables:
            self.tables[name] = MockLanceDBTable(name)
        return self.tables[name]


def connect(uri: Union[str, Path]):
    """Connect to mock LanceDB."""
    return MockLanceDB(uri)


# Mock sentence-transformers
class SentenceTransformer:
    """Mock SentenceTransformer model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = 384  # Common dimension
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Generate mock embeddings."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        for sentence in sentences:
            # Generate consistent mock embedding based on sentence hash
            seed = hash(sentence) % (2**32)
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
            embeddings.append(embedding)
        
        return embeddings if len(sentences) > 1 else embeddings[0]


# Mock duckdb
class MockDuckDBConnection:
    """Mock DuckDB connection."""
    
    def __init__(self):
        self._tables = {}
    
    def execute(self, query: str, params: Optional[List] = None):
        """Execute mock SQL query."""
        return MockDuckDBResult([])
    
    def fetchall(self):
        """Fetch all results."""
        return []
    
    def close(self):
        """Close connection."""
        pass


class MockDuckDBResult:
    """Mock DuckDB result."""
    
    def __init__(self, rows: List[tuple]):
        self.rows = rows
    
    def fetchall(self):
        """Fetch all rows."""
        return self.rows


def connect(database: str = ":memory:"):
    """Create mock DuckDB connection."""
    return MockDuckDBConnection()


# Mock loguru (simple version)
class MockLogger:
    """Mock loguru logger."""
    
    def info(self, message: str, *args, **kwargs):
        print(f"INFO: {message}")
    
    def debug(self, message: str, *args, **kwargs):
        print(f"DEBUG: {message}")
    
    def warning(self, message: str, *args, **kwargs):
        print(f"WARNING: {message}")
    
    def error(self, message: str, *args, **kwargs):
        print(f"ERROR: {message}")


# Create mock modules namespace
class MockModules:
    """Container for all mock modules."""
    
    # pydantic mocks
    class pydantic:
        BaseModel = BaseModel
        Field = Field
        field_validator = BaseModel.field_validator
    
    # aiofiles mock
    aiofiles = MockAioFiles()
    
    # frontmatter mock
    class frontmatter:
        Post = Post
        loads = loads
        dumps = dumps
    
    # ulid mock
    class ulid:
        ULID = ULID
    
    # tantivy mock
    class tantivy:
        Schema = Schema
        Index = Index
        QueryParser = QueryParser
    
    # lancedb mock
    class lancedb:
        connect = connect
    
    # sentence_transformers mock
    class sentence_transformers:
        SentenceTransformer = SentenceTransformer
    
    # duckdb mock
    class duckdb:
        connect = connect
    
    # loguru mock
    class loguru:
        logger = MockLogger()


# Export the mock modules
mock_modules = MockModules()