"""Data models for Symbiote daemon."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SearchCandidate:
    """A search result candidate with utility scoring."""
    id: str
    title: str
    path: str
    snippet: str
    base_score: float  # Raw score from source
    source: str  # fts|vector|recents
    project: Optional[str] = None
    tags: List[str] = None
    modified: Optional[datetime] = None


@dataclass 
class CaptureEntry:
    """Represents a single capture entry."""
    id: str
    type: str
    text: str
    source: str = "text"
    context: Optional[str] = None
    captured_at: Optional[datetime] = None