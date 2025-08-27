"""
Compatibility module that tries to import real dependencies first, 
then falls back to mock implementations.
"""

import sys
import warnings
from pathlib import Path

# Track which dependencies are mocked
_mocked_deps = set()

def is_mocked(dep_name: str) -> bool:
    """Check if a dependency is being mocked."""
    return dep_name in _mocked_deps

def get_mocked_dependencies() -> set:
    """Get set of dependencies that are being mocked."""
    return _mocked_deps.copy()

# Try to import real dependencies, fall back to mocks
try:
    from pydantic import BaseModel, Field, field_validator
    Config = None  # Will use real pydantic config
except ImportError:
    _mocked_deps.add('pydantic')
    warnings.warn("pydantic not available, using simplified config", ImportWarning)
    from .mock_dependencies import mock_modules
    BaseModel = mock_modules.pydantic.BaseModel
    Field = mock_modules.pydantic.Field
    field_validator = mock_modules.pydantic.field_validator
    
    # Use config stub
    from .config_stub import ConfigStub as Config

try:
    import aiofiles
except ImportError:
    _mocked_deps.add('aiofiles')
    warnings.warn("aiofiles not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    aiofiles = mock_modules.aiofiles

try:
    import frontmatter
except ImportError:
    _mocked_deps.add('frontmatter')
    warnings.warn("frontmatter not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    frontmatter = mock_modules.frontmatter

try:
    import ulid
except ImportError:
    _mocked_deps.add('ulid')
    warnings.warn("ulid not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    ulid = mock_modules.ulid

try:
    import tantivy
except ImportError:
    _mocked_deps.add('tantivy')
    warnings.warn("tantivy not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    tantivy = mock_modules.tantivy

try:
    import lancedb
except ImportError:
    _mocked_deps.add('lancedb')
    warnings.warn("lancedb not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    lancedb = mock_modules.lancedb

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _mocked_deps.add('sentence_transformers')
    warnings.warn("sentence_transformers not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    SentenceTransformer = mock_modules.sentence_transformers.SentenceTransformer

try:
    import duckdb
except ImportError:
    _mocked_deps.add('duckdb')
    warnings.warn("duckdb not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    duckdb = mock_modules.duckdb

try:
    from loguru import logger
except ImportError:
    _mocked_deps.add('loguru')
    warnings.warn("loguru not available, using mock", ImportWarning)
    from .mock_dependencies import mock_modules
    logger = mock_modules.loguru.logger

try:
    import psutil
except ImportError:
    _mocked_deps.add('psutil')
    warnings.warn("psutil not available, resource monitoring disabled", ImportWarning)
    psutil = None

try:
    import aiohttp
except ImportError:
    _mocked_deps.add('aiohttp')
    warnings.warn("aiohttp not available, HTTP server disabled", ImportWarning)
    aiohttp = None

# Print summary of what's mocked
if _mocked_deps:
    print(f"Symbiote running with mocked dependencies: {', '.join(sorted(_mocked_deps))}")
    print("Some features may have limited functionality.")
else:
    print("Symbiote running with all real dependencies.")