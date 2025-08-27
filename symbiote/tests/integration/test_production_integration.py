"""Integration tests for production components with real dependencies.

These tests verify that all production components work together correctly
with real implementations instead of mocks.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# Import production components
from symbiote.daemon.production_integration import (
    ProductionIntegration,
    DependencyChecker,
    get_production_integration
)
from symbiote.daemon.indexers.fts_production import FTSIndexProduction
from symbiote.daemon.indexers.vector_production import VectorIndexProduction
from symbiote.daemon.indexers.analytics_production import DuckDBAnalytics, AnalyticsEvent
from symbiote.daemon.search_orchestrator import SearchOrchestrator, SearchRequest
from symbiote.daemon.synthesis_worker_production import SynthesisWorker
from symbiote.daemon.privacy_gates import ConsentManager, ConsentLevel
from symbiote.daemon.error_handling import ResilientExecutor
from symbiote.daemon.algorithms_production import (
    TFIDFProcessor,
    DocumentClusterer,
    LinkSuggestionEngine,
    SearchFusionEngine
)


@pytest.fixture
async def temp_vault():
    """Create temporary vault with test documents."""
    vault_dir = tempfile.mkdtemp(prefix="test_vault_")
    vault_path = Path(vault_dir)
    
    # Create test documents
    docs = [
        {
            'name': 'machine_learning.md',
            'content': """# Machine Learning Notes
            
Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. Deep learning uses 
neural networks with multiple layers. Supervised learning uses labeled data.

Tags: #ml #ai #deeplearning
"""
        },
        {
            'name': 'python_tips.md',
            'content': """# Python Programming Tips

Python is a high-level programming language. Use list comprehensions for cleaner code.
Virtual environments help manage dependencies. Type hints improve code clarity.

Tags: #python #programming #tips
"""
        },
        {
            'name': 'project_ideas.md',
            'content': """# Project Ideas

1. Build a recommendation system using machine learning
2. Create a web scraper with Python
3. Develop a chatbot using natural language processing
4. Implement a time series forecasting model

Tags: #projects #ideas #ml #python
"""
        },
        {
            'name': 'daily_notes.md',
            'content': """# Daily Notes

Today I learned about transformers in deep learning. They are powerful for NLP tasks.
Need to review Python decorators tomorrow. Consider using PyTorch for the next project.

Contact: john.doe@example.com, phone: 555-123-4567

Tags: #daily #learning
"""
        }
    ]
    
    for doc in docs:
        doc_path = vault_path / doc['name']
        doc_path.write_text(doc['content'])
    
    yield vault_path
    
    # Cleanup
    shutil.rmtree(vault_dir)


@pytest.fixture
async def production_integration(temp_vault):
    """Create production integration instance."""
    config = {
        'vault_path': str(temp_vault),
        'storage_path': str(temp_vault / '.storage'),
        'production_mode': True,
        'enable_search_cache': False  # Disable for testing
    }
    
    # Create config file
    config_path = temp_vault / 'test_config.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    integration = ProductionIntegration(config_path)
    await integration.initialize()
    
    yield integration
    
    await integration.shutdown()


class TestDependencyDetection:
    """Test dependency detection and fallback."""
    
    def test_dependency_checker(self):
        """Test that dependency checker correctly identifies available packages."""
        deps = DependencyChecker.check_all()
        
        assert 'vector_search' in deps
        assert 'fts' in deps
        assert 'ml' in deps
        assert 'privacy' in deps
        assert 'analytics' in deps
        
        # At least some dependencies should be available in test environment
        available_count = sum(1 for d in deps.values() if d.available)
        assert available_count > 0
    
    def test_production_mode_detection(self, production_integration):
        """Test that production mode is correctly detected."""
        status = production_integration.get_status()
        
        assert 'mode' in status
        assert status['mode'] in ['production', 'degraded']
        assert 'dependencies' in status
        assert 'configuration' in status


class TestVectorSearch:
    """Test production vector search functionality."""
    
    @pytest.mark.asyncio
    async def test_vector_indexing(self, production_integration):
        """Test document indexing with real embeddings."""
        vector_index = production_integration.vector_indexer
        
        # Skip if vector dependencies not available
        if not production_integration.dependency_status['vector_search'].available:
            pytest.skip("Vector search dependencies not available")
        
        # Index a document
        doc_id = "test_doc"
        content = "This is a test document about machine learning and artificial intelligence."
        metadata = {'title': 'Test Document', 'created': datetime.now()}
        
        chunks = await vector_index.index_document(doc_id, content, metadata)
        assert chunks > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, production_integration):
        """Test semantic similarity search."""
        vector_index = production_integration.vector_indexer
        
        if not production_integration.dependency_status['vector_search'].available:
            pytest.skip("Vector search dependencies not available")
        
        # Reindex vault
        await vector_index.reindex_vault()
        
        # Search for semantically similar content
        results = await vector_index.search("deep learning neural networks", limit=5)
        
        assert len(results) > 0
        assert results[0].source == 'vector'
        assert results[0].base_score > 0.3  # Should have some similarity
    
    @pytest.mark.asyncio
    async def test_find_similar_documents(self, production_integration):
        """Test finding similar documents."""
        vector_index = production_integration.vector_indexer
        
        if not production_integration.dependency_status['vector_search'].available:
            pytest.skip("Vector search dependencies not available")
        
        await vector_index.reindex_vault()
        
        # Find documents similar to machine_learning
        similar = await vector_index.find_similar("machine_learning", limit=3)
        
        assert len(similar) > 0
        # Project ideas should be similar due to ML content
        assert any('project' in s.id.lower() for s in similar)


class TestFullTextSearch:
    """Test production FTS functionality."""
    
    @pytest.mark.asyncio
    async def test_fts_indexing(self, production_integration):
        """Test FTS indexing with Whoosh."""
        fts_index = production_integration.fts_indexer
        
        if not production_integration.dependency_status['fts'].available:
            pytest.skip("FTS dependencies not available")
        
        # Index a document
        success = await fts_index.index_document(
            "test_doc",
            "Python programming with machine learning",
            {'title': 'Test', 'tags': ['python', 'ml']}
        )
        
        assert success
    
    @pytest.mark.asyncio
    async def test_phrase_search(self, production_integration):
        """Test phrase searching."""
        fts_index = production_integration.fts_indexer
        
        if not production_integration.dependency_status['fts'].available:
            pytest.skip("FTS dependencies not available")
        
        await fts_index.reindex_vault()
        
        # Search for exact phrase
        results = await fts_index.search('"machine learning"', limit=5)
        
        assert len(results) > 0
        assert 'machine learning' in results[0].snippet.lower()
    
    @pytest.mark.asyncio
    async def test_field_search(self, production_integration):
        """Test searching specific fields."""
        fts_index = production_integration.fts_indexer
        
        if not production_integration.dependency_status['fts'].available:
            pytest.skip("FTS dependencies not available")
        
        await fts_index.reindex_vault()
        
        # Search in tags
        results = await fts_index.search("python", tags=["programming"])
        
        assert len(results) > 0


class TestSearchOrchestration:
    """Test search orchestration with racing and fusion."""
    
    @pytest.mark.asyncio
    async def test_search_racing(self, production_integration):
        """Test parallel search strategy racing."""
        orchestrator = production_integration.search_orchestrator
        
        request = SearchRequest(
            query="machine learning",
            limit=5,
            timeout_ms=500
        )
        
        start = time.time()
        result = await orchestrator.search(request)
        elapsed = time.time() - start
        
        assert len(result.candidates) > 0
        assert elapsed < 1.0  # Should complete quickly
        assert len(result.strategies_used) > 0
        assert 'total' in result.latency_ms
    
    @pytest.mark.asyncio
    async def test_early_termination(self, production_integration):
        """Test early termination when quality threshold met."""
        orchestrator = production_integration.search_orchestrator
        
        request = SearchRequest(
            query="python",
            limit=5,
            quality_threshold=0.5,  # Low threshold for early termination
            timeout_ms=1000
        )
        
        result = await orchestrator.search(request)
        
        assert len(result.candidates) > 0
        # Should terminate early, not use all strategies
        assert result.latency_ms['total'] < 500
    
    @pytest.mark.asyncio
    async def test_result_fusion(self, production_integration):
        """Test fusion of results from multiple strategies."""
        orchestrator = production_integration.search_orchestrator
        
        request = SearchRequest(
            query="programming tips",
            limit=10
        )
        
        result = await orchestrator.search(request)
        
        assert len(result.candidates) > 0
        assert result.fusion_method in ['keyword_heavy', 'semantic_heavy', 
                                        'temporal_heavy', 'balanced']
        
        # Check that results are properly scored
        scores = [c.base_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)  # Should be sorted


class TestPrivacyGates:
    """Test privacy and security features."""
    
    @pytest.mark.asyncio
    async def test_pii_detection(self, production_integration):
        """Test PII detection in documents."""
        privacy_mgr = production_integration.privacy_manager
        
        text = "Contact John Doe at john.doe@example.com or 555-123-4567"
        
        pii_matches = privacy_mgr.pii_detector.detect(text)
        
        assert len(pii_matches) > 0
        
        # Should detect email and phone
        pii_types = [m.pii_type.value for m in pii_matches]
        assert 'email' in pii_types
        assert 'phone' in pii_types
    
    @pytest.mark.asyncio
    async def test_data_redaction(self, production_integration):
        """Test PII redaction."""
        privacy_mgr = production_integration.privacy_manager
        
        text = "Email me at test@example.com with SSN 123-45-6789"
        
        redacted = privacy_mgr.redactor.redact(
            text,
            consent_level=ConsentLevel.ALLOW_ANONYMOUS
        )
        
        assert "test@example.com" not in redacted
        assert "123-45-6789" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "[SSN_REDACTED]" in redacted
    
    @pytest.mark.asyncio
    async def test_consent_management(self, production_integration):
        """Test consent request and grant flow."""
        privacy_mgr = production_integration.privacy_manager
        
        # Request consent
        request = await privacy_mgr.request_consent(
            operation="cloud_sync",
            data="Sample data with email@example.com",
            purpose="Sync to cloud storage"
        )
        
        assert request.request_id
        assert len(request.pii_detected) > 0
        
        # Grant consent
        success = await privacy_mgr.grant_consent(
            request.request_id,
            ConsentLevel.ALLOW_EXCERPTS
        )
        
        assert success


class TestSynthesisWorker:
    """Test pattern detection and synthesis."""
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, production_integration):
        """Test pattern detection in documents."""
        synthesis = production_integration.synthesis_worker
        
        if not production_integration.dependency_status['ml'].available:
            pytest.skip("ML dependencies not available")
        
        # Run synthesis
        results = await synthesis.synthesize()
        
        assert 'patterns' in results
        assert 'insights' in results
        assert results['document_count'] > 0
    
    @pytest.mark.asyncio
    async def test_theme_extraction(self, production_integration):
        """Test theme extraction from documents."""
        if not production_integration.dependency_status['ml'].available:
            pytest.skip("ML dependencies not available")
        
        # Load documents
        vault_path = Path(production_integration.config['vault_path'])
        documents = {}
        
        for md_file in vault_path.glob("*.md"):
            documents[md_file.stem] = md_file.read_text()
        
        # Extract themes
        processor = TFIDFProcessor()
        processor.fit_transform(documents)
        themes = processor.extract_themes(num_themes=3)
        
        assert len(themes) > 0
        assert themes[0].keywords
        assert themes[0].relevance_score > 0
    
    @pytest.mark.asyncio
    async def test_document_clustering(self, production_integration):
        """Test document clustering."""
        if not production_integration.dependency_status['ml'].available:
            pytest.skip("ML dependencies not available")
        
        vault_path = Path(production_integration.config['vault_path'])
        documents = {}
        
        for md_file in vault_path.glob("*.md"):
            documents[md_file.stem] = md_file.read_text()
        
        # Cluster documents
        processor = TFIDFProcessor()
        tfidf_matrix = processor.fit_transform(documents)
        
        clusterer = DocumentClusterer()
        clusters = clusterer.cluster_documents(
            tfidf_matrix,
            list(documents.keys()),
            method='kmeans'
        )
        
        # Should find some clusters in test documents
        assert len(clusters) > 0
        if clusters:
            assert clusters[0].coherence_score > 0


class TestAnalyticsEngine:
    """Test DuckDB analytics functionality."""
    
    @pytest.mark.asyncio
    async def test_event_tracking(self, production_integration, temp_vault):
        """Test analytics event tracking."""
        if not production_integration.dependency_status['analytics'].available:
            pytest.skip("Analytics dependencies not available")
        
        analytics = DuckDBAnalytics(db_path=temp_vault / 'analytics.db')
        
        # Track events
        event = AnalyticsEvent(
            event_id="test_001",
            event_type="document_access",
            timestamp=datetime.now(),
            document_id="test_doc",
            metadata={'action': 'read'}
        )
        
        await analytics.track_event(event)
        
        # Track search
        from symbiote.daemon.algorithms import SearchCandidate
        
        results = [
            SearchCandidate(
                id="doc1",
                title="Test",
                path="test.md",
                snippet="test",
                base_score=0.8,
                source="fts"
            )
        ]
        
        await analytics.track_search("test query", results, 50.0, ["fts"])
        
        # Get statistics
        activity = await analytics.get_document_activity(days=1)
        assert not activity.empty
    
    @pytest.mark.asyncio
    async def test_search_performance_metrics(self, production_integration, temp_vault):
        """Test search performance tracking."""
        if not production_integration.dependency_status['analytics'].available:
            pytest.skip("Analytics dependencies not available")
        
        analytics = DuckDBAnalytics(db_path=temp_vault / 'analytics.db')
        
        # Track multiple searches
        from symbiote.daemon.algorithms import SearchCandidate
        
        for i in range(10):
            results = [
                SearchCandidate(
                    id=f"doc{j}",
                    title=f"Result {j}",
                    path=f"doc{j}.md",
                    snippet="content",
                    base_score=0.5 + j * 0.1,
                    source="fts"
                )
                for j in range(3)
            ]
            
            latency = 20.0 + i * 10  # Varying latencies
            await analytics.track_search(f"query {i}", results, latency, ["fts", "vector"])
        
        # Get performance metrics
        perf = await analytics.get_search_performance()
        
        assert 'overall' in perf
        assert perf['overall']['total_queries'] > 0
        assert perf['overall']['avg_latency_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, production_integration, temp_vault):
        """Test anomaly detection in usage patterns."""
        if not production_integration.dependency_status['analytics'].available:
            pytest.skip("Analytics dependencies not available")
        
        analytics = DuckDBAnalytics(db_path=temp_vault / 'analytics.db')
        
        # Create normal baseline
        for day in range(7):
            for _ in range(10):
                event = AnalyticsEvent(
                    event_id=f"normal_{day}_{_}",
                    event_type="access",
                    timestamp=datetime.now() - timedelta(days=day),
                    document_id=f"doc_{_}",
                    metadata={}
                )
                await analytics.track_event(event)
        
        # Create anomaly (spike)
        for _ in range(100):
            event = AnalyticsEvent(
                event_id=f"spike_{_}",
                event_type="access",
                timestamp=datetime.now(),
                document_id=f"doc_{_}",
                metadata={}
            )
            await analytics.track_event(event)
        
        # Detect anomalies
        anomalies = await analytics.detect_anomalies()
        
        # Should detect the spike
        assert len(anomalies) > 0
        assert any(a['type'] == 'activity_spike' for a in anomalies)


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, production_integration):
        """Test circuit breaker functionality."""
        executor = production_integration.error_handler
        
        # Create failing function
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Service unavailable")
            return "Success"
        
        # Should retry and eventually succeed
        result = await executor.execute(
            "test_service",
            failing_function,
            circuit_breaker=True,
            retry=True
        )
        
        assert result == "Success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, production_integration, temp_vault):
        """Test automatic error recovery."""
        executor = production_integration.error_handler
        
        # Test file recovery
        missing_file = temp_vault / "missing.txt"
        
        async def read_file():
            return missing_file.read_text()
        
        # Should fail first time
        with pytest.raises(FileNotFoundError):
            await executor.execute(
                "file_service",
                read_file,
                recover=False
            )
        
        # Create context for recovery
        async def read_with_recovery():
            try:
                return missing_file.read_text()
            except FileNotFoundError:
                missing_file.touch()
                return "Created"
        
        result = await executor.execute(
            "file_service",
            read_with_recovery,
            recover=True
        )
        
        assert result == "Created"
        assert missing_file.exists()
    
    def test_health_monitoring(self, production_integration):
        """Test system health monitoring."""
        executor = production_integration.error_handler
        
        health = executor.get_health_status()
        
        assert 'circuit_breakers' in health
        assert 'error_summary' in health
        
        # Should have clean health initially
        assert health['error_summary']['total_errors'] == 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_document_lifecycle(self, production_integration):
        """Test complete document lifecycle from creation to search."""
        vault_path = Path(production_integration.config['vault_path'])
        
        # 1. Create new document
        new_doc = vault_path / "test_new_doc.md"
        new_doc.write_text("""# Test Document

This is a test document about quantum computing and cryptography.
It contains important information about quantum algorithms.

Tags: #quantum #crypto #algorithms
""")
        
        # 2. Index document in all systems
        if production_integration.dependency_status['fts'].available:
            await production_integration.fts_indexer.index_document(
                "test_new_doc",
                new_doc.read_text(),
                {'title': 'Test Document', 'tags': ['quantum', 'crypto']}
            )
        
        if production_integration.dependency_status['vector_search'].available:
            await production_integration.vector_indexer.index_document(
                "test_new_doc",
                new_doc.read_text(),
                {'title': 'Test Document'}
            )
        
        # 3. Search for document
        request = SearchRequest(
            query="quantum cryptography",
            limit=5
        )
        
        result = await production_integration.search_orchestrator.search(request)
        
        assert len(result.candidates) > 0
        assert any('test_new_doc' in c.id for c in result.candidates)
        
        # 4. Run synthesis to detect patterns
        if production_integration.dependency_status['ml'].available:
            synthesis_results = await production_integration.synthesis_worker.synthesize()
            assert synthesis_results['document_count'] > 0
        
        # 5. Check privacy if sensitive content
        text_with_pii = "Contact: alice@quantum.com for quantum key distribution"
        pii_matches = production_integration.privacy_manager.pii_detector.detect(text_with_pii)
        assert len(pii_matches) > 0
        
        # 6. Track analytics if available
        if production_integration.dependency_status['analytics'].available:
            analytics = DuckDBAnalytics()
            event = AnalyticsEvent(
                event_id="test_lifecycle",
                event_type="document_created",
                timestamp=datetime.now(),
                document_id="test_new_doc",
                metadata={'workflow': 'test'}
            )
            await analytics.track_event(event)
    
    @pytest.mark.asyncio
    async def test_production_mode_graceful_degradation(self):
        """Test that system degrades gracefully with missing dependencies."""
        # Create config with minimal dependencies
        config = {
            'vault_path': './test_vault_degrade',
            'production_mode': True
        }
        
        temp_dir = Path('./test_vault_degrade')
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize with potentially missing dependencies
            integration = ProductionIntegration()
            
            # Should not crash even if dependencies missing
            status = integration.get_status()
            assert 'mode' in status
            
            # Core functionality should still work
            assert integration.event_bus is not None
            assert integration.error_handler is not None
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for production components."""
    
    @pytest.mark.asyncio
    async def test_search_latency(self, production_integration, benchmark):
        """Benchmark search latency."""
        orchestrator = production_integration.search_orchestrator
        
        request = SearchRequest(
            query="machine learning",
            limit=10
        )
        
        # Warm up
        await orchestrator.search(request)
        
        # Benchmark
        result = await benchmark(orchestrator.search, request)
        
        assert result.latency_ms['total'] < 300  # Should meet SLO
    
    @pytest.mark.asyncio
    async def test_indexing_throughput(self, production_integration, benchmark):
        """Benchmark document indexing throughput."""
        if not production_integration.dependency_status['fts'].available:
            pytest.skip("FTS not available")
        
        fts_index = production_integration.fts_indexer
        
        async def index_batch():
            for i in range(10):
                await fts_index.index_document(
                    f"bench_doc_{i}",
                    f"Content for document {i} with various keywords",
                    {'title': f'Doc {i}'}
                )
        
        await benchmark(index_batch)
    
    @pytest.mark.asyncio
    async def test_pattern_detection_performance(self, production_integration, benchmark):
        """Benchmark pattern detection performance."""
        if not production_integration.dependency_status['ml'].available:
            pytest.skip("ML not available")
        
        synthesis = production_integration.synthesis_worker
        
        # Benchmark synthesis
        result = await benchmark(synthesis.synthesize)
        
        assert result['elapsed_seconds'] < 10  # Should complete quickly