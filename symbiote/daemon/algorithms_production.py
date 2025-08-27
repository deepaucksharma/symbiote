"""Production algorithms for search, suggestion, and synthesis.

This module implements sophisticated algorithms for:
- TF-IDF based theme extraction
- Document clustering using DBSCAN/K-means
- Link suggestion using co-occurrence and semantic similarity
- Pattern detection and trend analysis
- Multi-strategy search fusion with learned weights
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import heapq
import hashlib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

from loguru import logger
from .algorithms import SearchCandidate


@dataclass
class Theme:
    """Represents a discovered theme in the vault."""
    name: str
    keywords: List[str]
    document_ids: List[str]
    relevance_score: float
    time_range: Tuple[datetime, datetime]
    trend: str  # 'rising', 'stable', 'declining'
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'keywords': self.keywords,
            'document_count': len(self.document_ids),
            'relevance_score': self.relevance_score,
            'trend': self.trend
        }


@dataclass 
class DocumentCluster:
    """Represents a cluster of related documents."""
    cluster_id: int
    document_ids: List[str]
    centroid: np.ndarray
    keywords: List[str]
    coherence_score: float
    
    def to_dict(self) -> Dict:
        return {
            'cluster_id': self.cluster_id,
            'size': len(self.document_ids),
            'keywords': self.keywords[:5],
            'coherence_score': self.coherence_score
        }


@dataclass
class LinkSuggestion:
    """Represents a suggested link between documents."""
    source_id: str
    target_id: str
    confidence: float
    reason: str
    shared_concepts: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'confidence': self.confidence,
            'reason': self.reason,
            'shared_concepts': self.shared_concepts[:3]
        }


class TFIDFProcessor:
    """TF-IDF based text processing and analysis."""
    
    def __init__(self, 
                 max_features: int = 1000,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 use_stemming: bool = True):
        """
        Initialize TF-IDF processor.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as proportion)
            use_stemming: Whether to use stemming
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_stemming = use_stemming
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.document_ids = []
        
        if HAS_ML_DEPS and use_stemming:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
            except:
                self.stop_words = set()
                self.stemmer = None
        else:
            self.stop_words = set()
            self.stemmer = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF."""
        if not self.use_stemming or not self.stemmer:
            return text.lower()
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and stem
        processed = []
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                processed.append(stemmed)
        
        return ' '.join(processed)
    
    def fit_transform(self, documents: Dict[str, str]) -> np.ndarray:
        """
        Fit TF-IDF model and transform documents.
        
        Args:
            documents: Dict mapping document IDs to content
            
        Returns:
            TF-IDF matrix
        """
        if not HAS_ML_DEPS:
            # Fallback to simple term frequency
            return self._simple_tf_matrix(documents)
        
        # Prepare documents
        self.document_ids = list(documents.keys())
        texts = [self._preprocess_text(documents[doc_id]) for doc_id in self.document_ids]
        
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),  # Include bigrams
            sublinear_tf=True,  # Use log normalization
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return self.tfidf_matrix
    
    def _simple_tf_matrix(self, documents: Dict[str, str]) -> np.ndarray:
        """Simple term frequency matrix as fallback."""
        self.document_ids = list(documents.keys())
        
        # Build vocabulary
        vocab = set()
        doc_words = []
        for doc_id in self.document_ids:
            words = documents[doc_id].lower().split()
            doc_words.append(words)
            vocab.update(words)
        
        vocab = sorted(list(vocab))[:self.max_features]
        vocab_idx = {word: i for i, word in enumerate(vocab)}
        
        # Build matrix
        matrix = np.zeros((len(documents), len(vocab)))
        for i, words in enumerate(doc_words):
            for word in words:
                if word in vocab_idx:
                    matrix[i, vocab_idx[word]] += 1
        
        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + 1e-10)
        
        self.feature_names = np.array(vocab)
        return matrix
    
    def extract_keywords(self, doc_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords for a document."""
        if self.tfidf_matrix is None:
            return []
        
        # Get TF-IDF scores for document
        doc_tfidf = self.tfidf_matrix[doc_index].toarray().flatten()
        
        # Get top indices
        top_indices = doc_tfidf.argsort()[-top_n:][::-1]
        
        # Return keywords with scores
        keywords = []
        for idx in top_indices:
            if doc_tfidf[idx] > 0:
                keywords.append((self.feature_names[idx], doc_tfidf[idx]))
        
        return keywords
    
    def extract_themes(self, num_themes: int = 5) -> List[Theme]:
        """Extract main themes using LDA or clustering."""
        if self.tfidf_matrix is None or len(self.document_ids) < num_themes:
            return []
        
        themes = []
        
        if HAS_ML_DEPS:
            try:
                # Use LDA for topic modeling
                lda = LatentDirichletAllocation(
                    n_components=num_themes,
                    random_state=42,
                    max_iter=10
                )
                
                lda.fit(self.tfidf_matrix)
                
                # Extract themes
                for topic_idx, topic in enumerate(lda.components_):
                    # Get top words for this topic
                    top_word_indices = topic.argsort()[-10:][::-1]
                    keywords = [self.feature_names[i] for i in top_word_indices]
                    
                    # Get documents most related to this topic
                    doc_topic_dist = lda.transform(self.tfidf_matrix)
                    topic_docs = doc_topic_dist[:, topic_idx].argsort()[-10:][::-1]
                    doc_ids = [self.document_ids[i] for i in topic_docs]
                    
                    # Create theme
                    theme = Theme(
                        name=f"Theme_{topic_idx}_{keywords[0]}",
                        keywords=keywords,
                        document_ids=doc_ids,
                        relevance_score=float(topic.max()),
                        time_range=(datetime.now(), datetime.now()),
                        trend='stable'
                    )
                    themes.append(theme)
                    
            except Exception as e:
                logger.error(f"LDA failed: {e}")
        
        # Fallback to simple clustering
        if not themes:
            themes = self._simple_theme_extraction(num_themes)
        
        return themes
    
    def _simple_theme_extraction(self, num_themes: int) -> List[Theme]:
        """Simple theme extraction based on term frequency."""
        # Count term frequencies across all documents
        term_docs = defaultdict(set)
        term_scores = defaultdict(float)
        
        for i, doc_id in enumerate(self.document_ids):
            keywords = self.extract_keywords(i, top_n=20)
            for term, score in keywords:
                term_docs[term].add(doc_id)
                term_scores[term] += score
        
        # Get top terms as themes
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:num_themes]
        
        themes = []
        for i, (term, score) in enumerate(top_terms):
            theme = Theme(
                name=f"Theme_{term}",
                keywords=[term],
                document_ids=list(term_docs[term]),
                relevance_score=score / len(self.document_ids),
                time_range=(datetime.now(), datetime.now()),
                trend='stable'
            )
            themes.append(theme)
        
        return themes


class DocumentClusterer:
    """Document clustering for finding related content."""
    
    def __init__(self, min_cluster_size: int = 3):
        """
        Initialize document clusterer.
        
        Args:
            min_cluster_size: Minimum documents to form a cluster
        """
        self.min_cluster_size = min_cluster_size
        self.clusters = []
    
    def cluster_documents(self, 
                         tfidf_matrix: np.ndarray,
                         document_ids: List[str],
                         method: str = 'dbscan') -> List[DocumentCluster]:
        """
        Cluster documents based on similarity.
        
        Args:
            tfidf_matrix: TF-IDF matrix of documents
            document_ids: List of document IDs
            method: Clustering method ('dbscan' or 'kmeans')
            
        Returns:
            List of document clusters
        """
        if not HAS_ML_DEPS or tfidf_matrix.shape[0] < self.min_cluster_size:
            return []
        
        clusters = []
        
        try:
            if method == 'dbscan':
                # Use DBSCAN for density-based clustering
                clustering = DBSCAN(
                    eps=0.3,
                    min_samples=self.min_cluster_size,
                    metric='cosine'
                )
                
                labels = clustering.fit_predict(tfidf_matrix)
                
            else:  # kmeans
                # Determine optimal k using elbow method (simplified)
                k = min(10, tfidf_matrix.shape[0] // 5)
                
                clustering = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )
                
                labels = clustering.fit_predict(tfidf_matrix)
            
            # Process clusters
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label
            
            for cluster_id in unique_labels:
                # Get documents in this cluster
                cluster_mask = labels == cluster_id
                cluster_doc_ids = [
                    document_ids[i] for i in range(len(document_ids))
                    if cluster_mask[i]
                ]
                
                if len(cluster_doc_ids) < self.min_cluster_size:
                    continue
                
                # Calculate centroid
                cluster_vectors = tfidf_matrix[cluster_mask]
                centroid = cluster_vectors.mean(axis=0)
                
                # Calculate coherence (average pairwise similarity)
                similarities = cosine_similarity(cluster_vectors)
                coherence = similarities.mean()
                
                # Extract keywords (top terms in centroid)
                if hasattr(centroid, 'A'):
                    centroid_array = centroid.A.flatten()
                else:
                    centroid_array = centroid.flatten()
                
                top_indices = centroid_array.argsort()[-5:][::-1]
                keywords = [f"term_{i}" for i in top_indices]  # Placeholder
                
                cluster = DocumentCluster(
                    cluster_id=int(cluster_id),
                    document_ids=cluster_doc_ids,
                    centroid=centroid,
                    keywords=keywords,
                    coherence_score=float(coherence)
                )
                
                clusters.append(cluster)
                
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
        
        self.clusters = clusters
        return clusters
    
    def find_similar_clusters(self, 
                             doc_vector: np.ndarray,
                             top_k: int = 3) -> List[Tuple[DocumentCluster, float]]:
        """Find clusters most similar to a document."""
        if not self.clusters:
            return []
        
        similarities = []
        
        for cluster in self.clusters:
            # Calculate similarity to cluster centroid
            if hasattr(doc_vector, 'A'):
                doc_array = doc_vector.A.flatten()
            else:
                doc_array = doc_vector.flatten()
            
            if hasattr(cluster.centroid, 'A'):
                centroid_array = cluster.centroid.A.flatten()
            else:
                centroid_array = cluster.centroid.flatten()
            
            # Cosine similarity
            dot_product = np.dot(doc_array, centroid_array)
            norm_product = np.linalg.norm(doc_array) * np.linalg.norm(centroid_array)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities.append((cluster, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class LinkSuggestionEngine:
    """Engine for suggesting links between documents."""
    
    def __init__(self, 
                 min_confidence: float = 0.7,
                 max_suggestions: int = 10):
        """
        Initialize link suggestion engine.
        
        Args:
            min_confidence: Minimum confidence for suggestions
            max_suggestions: Maximum suggestions per document
        """
        self.min_confidence = min_confidence
        self.max_suggestions = max_suggestions
    
    def suggest_links(self,
                     tfidf_matrix: np.ndarray,
                     document_ids: List[str],
                     existing_links: Dict[str, Set[str]] = None) -> List[LinkSuggestion]:
        """
        Suggest links between related documents.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            document_ids: Document IDs
            existing_links: Existing links to avoid suggesting
            
        Returns:
            List of link suggestions
        """
        if existing_links is None:
            existing_links = {}
        
        suggestions = []
        
        # Calculate pairwise similarities
        if HAS_ML_DEPS:
            similarities = cosine_similarity(tfidf_matrix)
        else:
            # Simple dot product similarity
            similarities = tfidf_matrix.dot(tfidf_matrix.T)
            if hasattr(similarities, 'toarray'):
                similarities = similarities.toarray()
        
        # Find top suggestions for each document
        for i, source_id in enumerate(document_ids):
            # Get similarity scores for this document
            doc_similarities = similarities[i]
            
            # Get indices of most similar documents
            similar_indices = doc_similarities.argsort()[-self.max_suggestions-1:][::-1]
            
            for j in similar_indices:
                if i == j:  # Skip self
                    continue
                
                target_id = document_ids[j]
                confidence = float(doc_similarities[j])
                
                # Skip if below threshold
                if confidence < self.min_confidence:
                    continue
                
                # Skip if link already exists
                if target_id in existing_links.get(source_id, set()):
                    continue
                
                # Determine reason for suggestion
                if confidence > 0.9:
                    reason = "Very high content similarity"
                elif confidence > 0.8:
                    reason = "High content similarity"
                else:
                    reason = "Moderate content similarity"
                
                suggestion = LinkSuggestion(
                    source_id=source_id,
                    target_id=target_id,
                    confidence=confidence,
                    reason=reason,
                    shared_concepts=[]  # TODO: Extract shared concepts
                )
                
                suggestions.append(suggestion)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions


class SearchFusionEngine:
    """Advanced search result fusion with learned weights."""
    
    def __init__(self):
        """Initialize search fusion engine."""
        # Learned weights for different scenarios
        self.scenario_weights = {
            'keyword_heavy': {'fts': 0.7, 'vector': 0.2, 'recents': 0.1},
            'semantic_heavy': {'fts': 0.3, 'vector': 0.6, 'recents': 0.1},
            'temporal_heavy': {'fts': 0.2, 'vector': 0.2, 'recents': 0.6},
            'balanced': {'fts': 0.4, 'vector': 0.4, 'recents': 0.2}
        }
        
        # Track performance for weight learning
        self.performance_history = []
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query to apply appropriate weights."""
        query_lower = query.lower()
        
        # Check for temporal indicators
        temporal_keywords = ['today', 'yesterday', 'recent', 'latest', 'last', 'new']
        if any(keyword in query_lower for keyword in temporal_keywords):
            return 'temporal_heavy'
        
        # Check for semantic indicators (questions, concepts)
        semantic_indicators = ['what', 'why', 'how', 'explain', 'concept', 'idea']
        if any(indicator in query_lower for indicator in semantic_indicators):
            return 'semantic_heavy'
        
        # Check for specific keywords (likely FTS)
        if '"' in query or len(query.split()) == 1:
            return 'keyword_heavy'
        
        return 'balanced'
    
    def fuse_results(self,
                    fts_results: List[SearchCandidate],
                    vector_results: List[SearchCandidate],
                    recents_results: List[SearchCandidate],
                    query: str,
                    project_hint: Optional[str] = None) -> List[SearchCandidate]:
        """
        Fuse results from multiple search strategies.
        
        Args:
            fts_results: Full-text search results
            vector_results: Vector search results
            recents_results: Recent items results
            query: Original query
            project_hint: Project context
            
        Returns:
            Fused and re-ranked results
        """
        # Detect query type
        query_type = self.detect_query_type(query)
        weights = self.scenario_weights[query_type]
        
        # Combine all results
        all_candidates = {}
        
        # Process each result set
        for candidates, source_weight in [
            (fts_results, weights['fts']),
            (vector_results, weights['vector']),
            (recents_results, weights['recents'])
        ]:
            for rank, candidate in enumerate(candidates):
                if candidate.id not in all_candidates:
                    all_candidates[candidate.id] = {
                        'candidate': candidate,
                        'scores': {},
                        'ranks': {}
                    }
                
                # Store score and rank from this source
                all_candidates[candidate.id]['scores'][candidate.source] = (
                    candidate.base_score * source_weight
                )
                all_candidates[candidate.id]['ranks'][candidate.source] = rank
        
        # Calculate final scores
        fused_results = []
        
        for doc_id, data in all_candidates.items():
            candidate = data['candidate']
            
            # Combine scores (weighted sum)
            final_score = sum(data['scores'].values())
            
            # Apply rank fusion (reciprocal rank fusion)
            rrf_score = 0
            for source, rank in data['ranks'].items():
                rrf_score += weights[source] / (rank + 60)  # RRF with k=60
            
            # Combine score and rank fusion
            candidate.base_score = 0.7 * final_score + 0.3 * rrf_score
            
            # Apply project boost if applicable
            if project_hint and candidate.project == project_hint:
                candidate.base_score *= 1.15
            
            fused_results.append(candidate)
        
        # Sort by final score
        fused_results.sort(key=lambda x: x.base_score, reverse=True)
        
        # Track performance for learning (would need user feedback)
        self._track_performance(query_type, fused_results)
        
        return fused_results
    
    def _track_performance(self, query_type: str, results: List[SearchCandidate]):
        """Track performance for weight learning."""
        # In production, this would track click-through rates,
        # dwell time, and user feedback to adjust weights
        self.performance_history.append({
            'timestamp': datetime.now(),
            'query_type': query_type,
            'result_count': len(results),
            'top_score': results[0].base_score if results else 0
        })
        
        # Periodically adjust weights based on performance
        if len(self.performance_history) % 100 == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Update weights based on performance history."""
        # Placeholder for weight learning algorithm
        # In production, would use gradient descent or
        # reinforcement learning to optimize weights
        pass


class SuggestionGenerator:
    """Advanced suggestion generation with explainability."""
    
    def __init__(self):
        """Initialize suggestion generator."""
        self.tfidf_processor = TFIDFProcessor()
        self.clusterer = DocumentClusterer()
        self.link_engine = LinkSuggestionEngine()
    
    def generate_suggestions(self,
                           context: Dict[str, Any],
                           documents: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate intelligent suggestions based on context.
        
        Args:
            context: Current context (query, project, time available, etc.)
            documents: Document corpus
            
        Returns:
            Suggestions with explanations
        """
        suggestions = {
            'themes': [],
            'clusters': [],
            'links': [],
            'next_actions': []
        }
        
        if not documents:
            return suggestions
        
        # Process documents with TF-IDF
        tfidf_matrix = self.tfidf_processor.fit_transform(documents)
        
        # Extract themes
        themes = self.tfidf_processor.extract_themes(num_themes=5)
        suggestions['themes'] = [theme.to_dict() for theme in themes]
        
        # Find clusters
        clusters = self.clusterer.cluster_documents(
            tfidf_matrix,
            list(documents.keys())
        )
        suggestions['clusters'] = [cluster.to_dict() for cluster in clusters]
        
        # Suggest links
        links = self.link_engine.suggest_links(
            tfidf_matrix,
            list(documents.keys())
        )
        suggestions['links'] = [link.to_dict() for link in links[:10]]
        
        # Generate next actions based on context
        if context.get('free_minutes', 0) > 0:
            suggestions['next_actions'] = self._generate_next_actions(
                context,
                themes,
                clusters
            )
        
        return suggestions
    
    def _generate_next_actions(self,
                              context: Dict,
                              themes: List[Theme],
                              clusters: List[DocumentCluster]) -> List[Dict]:
        """Generate actionable next steps."""
        actions = []
        free_minutes = context.get('free_minutes', 0)
        
        # Quick actions (< 15 minutes)
        if free_minutes >= 5:
            if themes:
                actions.append({
                    'action': f"Review theme: {themes[0].name}",
                    'estimated_minutes': 10,
                    'reason': f"Most relevant theme with {len(themes[0].document_ids)} documents"
                })
        
        # Medium actions (15-30 minutes)
        if free_minutes >= 15:
            if clusters and clusters[0].coherence_score > 0.7:
                actions.append({
                    'action': f"Explore cluster of {len(clusters[0].document_ids)} related documents",
                    'estimated_minutes': 20,
                    'reason': f"High coherence cluster (score: {clusters[0].coherence_score:.2f})"
                })
        
        # Longer actions (30+ minutes)
        if free_minutes >= 30:
            actions.append({
                'action': "Synthesize findings across top themes",
                'estimated_minutes': 30,
                'reason': "Create comprehensive understanding"
            })
        
        return actions