"""Core algorithms for search, suggestion, and synthesis."""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


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
    
    def calculate_utility(
        self,
        query_project: Optional[str] = None,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate utility score based on Part 3 specification.
        
        utility = max(
            FTS_score * 1.00 + project_match*0.10 + recency_boost*0.05,
            Vector_score * 0.95 + project_match*0.10 + recency_boost*0.05,
            Recents_score * 0.80 + project_match*0.20
        )
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Base weight by source
        source_weights = {
            "fts": 1.00,
            "vector": 0.95,
            "recents": 0.80
        }
        weighted_score = self.base_score * source_weights.get(self.source, 1.0)
        
        # Project match bonus
        project_match = 0.2 if (query_project and self.project == query_project) else 0.0
        if self.source == "recents":
            project_match *= 2  # Recents get higher project weight
        else:
            project_match *= 0.5  # Others get standard weight
        
        # Recency boost (exponential decay)
        recency_boost = 0.0
        if self.modified:
            hours_ago = (current_time - self.modified).total_seconds() / 3600
            if hours_ago < 72:  # Within 3 days
                # exp(-Δt / τ) where τ = 24 hours
                recency_boost = 0.2 * math.exp(-hours_ago / 24)
        
        utility = weighted_score + project_match + recency_boost * 0.25
        return min(1.0, utility)  # Cap at 1.0


class SearchRacer:
    """
    Implements the racing search algorithm from Part 3.
    Returns first useful results ASAP.
    """
    
    USEFULNESS_THRESHOLD = 0.55
    
    @staticmethod
    def is_useful(candidates: List[SearchCandidate], query_project: Optional[str] = None) -> bool:
        """Check if any candidate meets the usefulness threshold."""
        if not candidates:
            return False
        
        for candidate in candidates[:3]:  # Check top 3
            utility = candidate.calculate_utility(query_project)
            if utility >= SearchRacer.USEFULNESS_THRESHOLD:
                return True
        
        return False
    
    @staticmethod
    def merge_results(
        existing: List[SearchCandidate],
        new: List[SearchCandidate],
        max_results: int = 10
    ) -> List[SearchCandidate]:
        """Merge and deduplicate results, maintaining utility order."""
        seen_ids = {c.id for c in existing}
        
        # Add new results
        for candidate in new:
            if candidate.id not in seen_ids:
                existing.append(candidate)
                seen_ids.add(candidate.id)
        
        # Re-sort by utility
        existing.sort(
            key=lambda c: c.calculate_utility(),
            reverse=True
        )
        
        return existing[:max_results]
    
    @staticmethod
    def rerank_combined(
        fts_results: List[SearchCandidate],
        vector_results: List[SearchCandidate],
        max_results: int = 10
    ) -> List[SearchCandidate]:
        """
        Optional re-ranking when both FTS and Vector finish quickly.
        Combines and re-ranks by utility score.
        """
        combined = []
        seen_ids = set()
        
        for candidate in fts_results + vector_results:
            if candidate.id not in seen_ids:
                combined.append(candidate)
                seen_ids.add(candidate.id)
        
        # Sort by utility
        combined.sort(
            key=lambda c: c.calculate_utility(),
            reverse=True
        )
        
        return combined[:max_results]


@dataclass
class SuggestionCandidate:
    """A suggestion candidate with scoring."""
    text: str
    kind: str  # next_action|clarify|review|link_suggestion
    sources: List[Dict[str, Any]]
    heuristics: List[str]
    score: float = 0.0
    
    def calculate_score(
        self,
        relevance_to_query: float,
        recency_weight: float,
        feasibility: float,
        seen_recently: bool = False
    ) -> float:
        """
        Calculate suggestion score per Part 3:
        
        sugg_score =
          0.35 * relevance_to_query
        + 0.25 * recency_of_sources
        + 0.20 * feasibility_given_free_minutes
        + 0.20 * repetition_penalty_inverse
        """
        repetition_penalty = 0.3 if seen_recently else 1.0
        
        self.score = (
            0.35 * relevance_to_query +
            0.25 * recency_weight +
            0.20 * feasibility +
            0.20 * repetition_penalty
        )
        
        return self.score


class SuggestionGenerator:
    """Generate actionable suggestions with receipts."""
    
    SUGGESTION_THRESHOLD = 0.65
    MAX_SUGGESTION_LENGTH = 120
    
    @staticmethod
    def generate_heuristic_candidates(
        context_items: List[Dict[str, Any]],
        free_minutes: int = 30,
        project: Optional[str] = None
    ) -> List[SuggestionCandidate]:
        """Generate suggestion candidates based on heuristics."""
        candidates = []
        
        # Check for recently edited docs in project
        if project and context_items:
            cutoff_time = datetime.utcnow() - timedelta(days=2)
            recent_docs = []
            for item in context_items:
                if item.get("project") == project:
                    modified = item.get("modified")
                    if isinstance(modified, datetime) and modified > cutoff_time:
                        recent_docs.append(item)
                    elif isinstance(modified, (int, float)) and modified > cutoff_time.timestamp():
                        recent_docs.append(item)
            if recent_docs:
                candidates.append(SuggestionCandidate(
                    text=f"Continue work on {recent_docs[0].get('title', 'document')} ({min(25, free_minutes)} min)",
                    kind="next_action",
                    sources=[{"id": recent_docs[0]["id"], "reason": "recent_edit"}],
                    heuristics=["recent_project_activity", f"{free_minutes}m_free_window"]
                ))
        
        # Check for inbox items if short time window
        inbox_items = [
            item for item in context_items
            if item.get("status") == "inbox"
        ]
        if inbox_items and free_minutes < 20:
            candidates.append(SuggestionCandidate(
                text=f"Clarify {min(5, len(inbox_items))} inbox items ({free_minutes} min)",
                kind="clarify",
                sources=[{"id": item["id"], "reason": "inbox"} for item in inbox_items[:5]],
                heuristics=["inbox_processing", f"{free_minutes}m_free_window"]
            ))
        
        # Check for overdue tasks
        overdue = [
            item for item in context_items
            if item.get("due") and item["due"] < datetime.utcnow().isoformat()
        ]
        if overdue:
            candidates.append(SuggestionCandidate(
                text=f"Complete overdue: {overdue[0].get('title', 'task')} ({min(20, free_minutes)} min)",
                kind="next_action",
                sources=[{"id": overdue[0]["id"], "reason": "overdue"}],
                heuristics=["overdue_task", f"{free_minutes}m_free_window"]
            ))
        
        return candidates
    
    @staticmethod
    def select_best_suggestion(
        candidates: List[SuggestionCandidate],
        query: Optional[str] = None
    ) -> Optional[SuggestionCandidate]:
        """Select the best suggestion above threshold."""
        if not candidates:
            return None
        
        # Score each candidate
        for candidate in candidates:
            # Simple relevance scoring
            relevance = 0.7  # Default
            if query:
                # Check if query terms appear in suggestion
                query_terms = query.lower().split()
                suggestion_terms = candidate.text.lower().split()
                overlap = len(set(query_terms) & set(suggestion_terms))
                relevance = min(1.0, 0.5 + overlap * 0.2)
            
            # Recency (simplified - based on heuristics)
            recency = 0.8 if "recent" in " ".join(candidate.heuristics) else 0.5
            
            # Feasibility (based on time match)
            feasibility = 0.9 if "free_window" in " ".join(candidate.heuristics) else 0.6
            
            candidate.calculate_score(relevance, recency, feasibility)
        
        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        best = candidates[0]
        if best.score >= SuggestionGenerator.SUGGESTION_THRESHOLD:
            # Truncate text if needed
            if len(best.text) > SuggestionGenerator.MAX_SUGGESTION_LENGTH:
                best.text = best.text[:117] + "..."
            return best
        
        return None
    
    @staticmethod
    def determine_confidence(score: float) -> str:
        """Determine confidence level from score."""
        if score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        else:
            return "low"


class ThemeSynthesizer:
    """Extract themes and patterns for daily/weekly synthesis."""
    
    MAX_THEMES = 3
    MAX_LINKS = 7
    MAX_THEME_LENGTH = 16  # words
    
    @staticmethod
    def extract_themes(
        recent_items: List[Dict[str, Any]],
        time_window_hours: int = 72
    ) -> List[str]:
        """
        Extract themes from recent activity using TF-IDF.
        Returns list of theme labels.
        """
        # Simplified theme extraction
        # In production, would use proper TF-IDF
        
        # Count tag frequencies
        tag_counts = {}
        project_counts = {}
        
        for item in recent_items:
            for tag in item.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            project = item.get("project")
            if project:
                project_counts[project] = project_counts.get(project, 0) + 1
        
        themes = []
        
        # Top tags as themes
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3:  # Minimum frequency
                themes.append(f"Focus on {tag}")
                if len(themes) >= ThemeSynthesizer.MAX_THEMES:
                    break
        
        # Add project themes if room
        if len(themes) < ThemeSynthesizer.MAX_THEMES:
            for project, count in sorted(project_counts.items(), key=lambda x: x[1], reverse=True):
                if count >= 5:
                    themes.append(f"Progress on {project}")
                    if len(themes) >= ThemeSynthesizer.MAX_THEMES:
                        break
        
        return themes[:ThemeSynthesizer.MAX_THEMES]
    
    @staticmethod
    def suggest_links(
        items: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Suggest links between items based on co-occurrence and similarity.
        Returns list of (src_id, dst_id, score) tuples.
        """
        suggestions = []
        
        # Simple co-occurrence based on shared tags/project
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                score = 0.0
                
                # Same project
                if item1.get("project") and item1["project"] == item2.get("project"):
                    score += 0.3
                
                # Shared tags
                tags1 = set(item1.get("tags", []))
                tags2 = set(item2.get("tags", []))
                if tags1 and tags2:
                    overlap = len(tags1 & tags2)
                    if overlap > 0:
                        score += 0.2 * min(1.0, overlap / min(len(tags1), len(tags2)))
                
                # Temporal proximity (modified within 3 days)
                if item1.get("modified") and item2.get("modified"):
                    mod1 = item1["modified"]
                    mod2 = item2["modified"]
                    
                    # Convert to timestamps if needed
                    if isinstance(mod1, datetime):
                        mod1 = mod1.timestamp()
                    if isinstance(mod2, datetime):
                        mod2 = mod2.timestamp()
                    
                    time_diff = abs(mod1 - mod2)
                    if time_diff < 3 * 24 * 3600:  # 3 days in seconds
                        score += 0.2
                
                # Add semantic similarity score (would use embeddings in production)
                # For now, simple title similarity
                if item1.get("title") and item2.get("title"):
                    title1_words = set(item1["title"].lower().split())
                    title2_words = set(item2["title"].lower().split())
                    if title1_words and title2_words:
                        jaccard = len(title1_words & title2_words) / len(title1_words | title2_words)
                        score += 0.3 * jaccard
                
                if score >= threshold:
                    suggestions.append((item1["id"], item2["id"], score))
        
        # Sort by score and limit
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:ThemeSynthesizer.MAX_LINKS]