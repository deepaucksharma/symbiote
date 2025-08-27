"""Unit tests for core algorithms - search racing, suggestions, synthesis."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from symbiote.daemon.algorithms import (
    SearchCandidate, SearchRacer, SuggestionCandidate,
    SuggestionGenerator, ThemeSynthesizer
)


class TestSearchCandidate:
    """Test search candidate utility scoring."""
    
    def test_utility_calculation_fts(self):
        """Test FTS utility calculation."""
        candidate = SearchCandidate(
            id="test1",
            title="Q3 Strategy",
            path="notes/q3.md",
            snippet="Strategy outline",
            base_score=0.8,
            source="fts",
            project="planning",
            modified=datetime.utcnow() - timedelta(hours=12)
        )
        
        # FTS gets 1.0 weight
        utility = candidate.calculate_utility(query_project="planning")
        
        # Should include base score + project match + recency
        assert utility > 0.8  # Base score
        assert utility < 1.0  # Capped at 1.0
    
    def test_utility_calculation_vector(self):
        """Test vector search utility calculation."""
        candidate = SearchCandidate(
            id="test2",
            title="Related Document",
            path="notes/related.md",
            snippet="Similar content",
            base_score=0.75,
            source="vector",
            project=None,
            modified=datetime.utcnow() - timedelta(days=5)
        )
        
        # Vector gets 0.95 weight
        utility = candidate.calculate_utility()
        expected = 0.75 * 0.95  # No project match, old recency
        assert abs(utility - expected) < 0.1
    
    def test_utility_calculation_recents(self):
        """Test recents utility calculation."""
        candidate = SearchCandidate(
            id="test3",
            title="Recent Note",
            path="notes/recent.md",
            snippet="Just created",
            base_score=0.5,
            source="recents",
            project="current",
            modified=datetime.utcnow() - timedelta(minutes=30)
        )
        
        utility = candidate.calculate_utility(query_project="current")
        
        # Recents get 0.8 weight but 2x project bonus
        assert utility > 0.5  # Should be boosted
    
    def test_recency_boost_decay(self):
        """Test recency boost exponential decay."""
        now = datetime.utcnow()
        
        # Very recent (1 hour ago)
        recent = SearchCandidate(
            id="r1", title="Recent", path="r.md", snippet="",
            base_score=0.5, source="fts",
            modified=now - timedelta(hours=1)
        )
        
        # Old (4 days ago)
        old = SearchCandidate(
            id="o1", title="Old", path="o.md", snippet="",
            base_score=0.5, source="fts",
            modified=now - timedelta(days=4)
        )
        
        recent_utility = recent.calculate_utility()
        old_utility = old.calculate_utility()
        
        # Recent should have higher utility due to recency boost
        assert recent_utility > old_utility


class TestSearchRacer:
    """Test search racing algorithm."""
    
    def test_is_useful_threshold(self):
        """Test usefulness threshold detection."""
        # Below threshold
        low_candidates = [
            SearchCandidate("1", "Low", "l.md", "", 0.4, "fts")
        ]
        assert not SearchRacer.is_useful(low_candidates)
        
        # Above threshold
        high_candidates = [
            SearchCandidate("2", "High", "h.md", "", 0.6, "fts")
        ]
        assert SearchRacer.is_useful(high_candidates)
    
    def test_is_useful_with_project_match(self):
        """Test usefulness with project match."""
        candidates = [
            SearchCandidate(
                "1", "Match", "m.md", "", 0.45, "fts",
                project="target"
            )
        ]
        
        # Below threshold without project
        assert not SearchRacer.is_useful(candidates)
        
        # Should pass with project match due to boost
        assert SearchRacer.is_useful(candidates, query_project="target")
    
    def test_merge_results_deduplication(self):
        """Test result merging with deduplication."""
        existing = [
            SearchCandidate("1", "First", "1.md", "", 0.8, "fts"),
            SearchCandidate("2", "Second", "2.md", "", 0.7, "vector")
        ]
        
        new = [
            SearchCandidate("2", "Second", "2.md", "", 0.75, "recents"),  # Duplicate
            SearchCandidate("3", "Third", "3.md", "", 0.6, "recents")
        ]
        
        merged = SearchRacer.merge_results(existing, new)
        
        # Should deduplicate
        ids = [r.id for r in merged]
        assert len(ids) == len(set(ids))
        assert "3" in ids
    
    def test_rerank_combined(self):
        """Test combined reranking of FTS and vector results."""
        fts_results = [
            SearchCandidate("1", "FTS1", "1.md", "", 0.9, "fts"),
            SearchCandidate("2", "FTS2", "2.md", "", 0.7, "fts")
        ]
        
        vector_results = [
            SearchCandidate("3", "Vec1", "3.md", "", 0.85, "vector"),
            SearchCandidate("2", "FTS2", "2.md", "", 0.8, "vector")  # Overlap
        ]
        
        reranked = SearchRacer.rerank_combined(fts_results, vector_results, max_results=3)
        
        # Should combine and rerank by utility
        assert len(reranked) == 3
        assert reranked[0].base_score >= reranked[1].base_score


class TestSuggestionGenerator:
    """Test suggestion generation."""
    
    def test_generate_project_suggestion(self):
        """Test suggestion for recent project activity."""
        context_items = [
            {
                "id": "1",
                "title": "API Design",
                "project": "api-v2",
                "modified": (datetime.utcnow() - timedelta(hours=1)).timestamp()
            }
        ]
        
        candidates = SuggestionGenerator.generate_heuristic_candidates(
            context_items,
            free_minutes=30,
            project="api-v2"
        )
        
        assert len(candidates) > 0
        assert any("Continue work" in c.text for c in candidates)
    
    def test_generate_inbox_suggestion(self):
        """Test suggestion for inbox processing."""
        context_items = [
            {"id": f"t{i}", "status": "inbox"} 
            for i in range(5)
        ]
        
        candidates = SuggestionGenerator.generate_heuristic_candidates(
            context_items,
            free_minutes=15
        )
        
        assert any("inbox" in c.text.lower() for c in candidates)
    
    def test_generate_overdue_suggestion(self):
        """Test suggestion for overdue tasks."""
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        context_items = [
            {
                "id": "overdue1",
                "title": "Important Task",
                "due": yesterday
            }
        ]
        
        candidates = SuggestionGenerator.generate_heuristic_candidates(
            context_items,
            free_minutes=20
        )
        
        assert any("overdue" in c.text.lower() for c in candidates)
    
    def test_select_best_suggestion(self):
        """Test best suggestion selection."""
        candidates = [
            SuggestionCandidate(
                "Low quality", "next_action", [], ["test"], score=0.4
            ),
            SuggestionCandidate(
                "High quality", "next_action", [], ["test"], score=0.8
            )
        ]
        
        # Manually set scores
        candidates[0].score = 0.4
        candidates[1].score = 0.8
        
        best = SuggestionGenerator.select_best_suggestion(candidates)
        
        assert best is not None
        assert best.text == "High quality"
    
    def test_suggestion_length_limit(self):
        """Test suggestion text length is limited."""
        long_text = "x" * 200
        candidates = [
            SuggestionCandidate(long_text, "next_action", [], [])
        ]
        candidates[0].score = 0.9
        
        best = SuggestionGenerator.select_best_suggestion(candidates)
        
        assert best is not None
        assert len(best.text) <= 120
        assert best.text.endswith("...")
    
    def test_confidence_levels(self):
        """Test confidence determination."""
        assert SuggestionGenerator.determine_confidence(0.9) == "high"
        assert SuggestionGenerator.determine_confidence(0.75) == "medium"
        assert SuggestionGenerator.determine_confidence(0.6) == "low"


class TestThemeSynthesizer:
    """Test theme extraction and link suggestion."""
    
    def test_extract_themes_from_tags(self):
        """Test theme extraction from tag frequencies."""
        recent_items = [
            {"tags": ["api", "design"]},
            {"tags": ["api", "testing"]},
            {"tags": ["api", "documentation"]},
            {"tags": ["performance"]},
            {"tags": ["performance", "optimization"]}
        ]
        
        themes = ThemeSynthesizer.extract_themes(recent_items)
        
        assert len(themes) <= ThemeSynthesizer.MAX_THEMES
        assert any("api" in theme.lower() for theme in themes)
    
    def test_extract_themes_from_projects(self):
        """Test theme extraction from project activity."""
        recent_items = [
            {"project": "WebRTC"} for _ in range(6)
        ] + [
            {"project": "Dashboard"} for _ in range(4)
        ]
        
        themes = ThemeSynthesizer.extract_themes(recent_items)
        
        assert len(themes) > 0
        assert any("WebRTC" in theme for theme in themes)
    
    def test_suggest_links_cooccurrence(self):
        """Test link suggestion based on co-occurrence."""
        items = [
            {
                "id": "1",
                "title": "API Design",
                "project": "api-v2",
                "tags": ["api", "design"],
                "modified": datetime.utcnow().timestamp()
            },
            {
                "id": "2",
                "title": "API Testing",
                "project": "api-v2",
                "tags": ["api", "testing"],
                "modified": datetime.utcnow().timestamp()
            },
            {
                "id": "3",
                "title": "Unrelated",
                "project": "other",
                "tags": ["random"],
                "modified": (datetime.utcnow() - timedelta(days=10)).timestamp()
            }
        ]
        
        suggestions = ThemeSynthesizer.suggest_links(items, threshold=0.3)
        
        # Should suggest link between items 1 and 2 (same project + shared tags)
        assert len(suggestions) > 0
        
        # Check that high-scoring pair is suggested
        src, dst, score = suggestions[0]
        assert score > 0.3
    
    def test_suggest_links_threshold(self):
        """Test link suggestion threshold filtering."""
        items = [
            {"id": "1", "title": "A", "tags": []},
            {"id": "2", "title": "B", "tags": []},
        ]
        
        # High threshold should filter out weak links
        suggestions = ThemeSynthesizer.suggest_links(items, threshold=0.9)
        assert len(suggestions) == 0
    
    def test_max_links_limit(self):
        """Test maximum link suggestions limit."""
        # Create many items that would all link
        items = [
            {
                "id": str(i),
                "title": f"Item {i}",
                "project": "same",
                "tags": ["shared"],
                "modified": datetime.utcnow().timestamp()
            }
            for i in range(10)
        ]
        
        suggestions = ThemeSynthesizer.suggest_links(items, threshold=0.1)
        
        # Should be limited
        assert len(suggestions) <= ThemeSynthesizer.MAX_LINKS


@pytest.mark.parametrize("score,expected_confidence", [
    (0.95, "high"),
    (0.87, "high"),
    (0.78, "medium"),
    (0.72, "medium"),
    (0.65, "low"),
    (0.50, "low"),
])
def test_confidence_mapping(score, expected_confidence):
    """Test confidence level mapping."""
    confidence = SuggestionGenerator.determine_confidence(score)
    assert confidence == expected_confidence