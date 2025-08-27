"""Production integration module that replaces all mock implementations.

This module provides the production-ready integration layer that:
- Detects available dependencies and uses real implementations
- Falls back gracefully when dependencies are missing
- Provides unified interface for all components
- Manages component lifecycle and health
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import importlib.util

from loguru import logger

# Import production components
from .indexers.fts_production import FTSIndexProduction
from .indexers.vector_production import VectorIndexProduction
from .algorithms_production import (
    TFIDFProcessor,
    DocumentClusterer,
    LinkSuggestionEngine,
    SearchFusionEngine,
    SuggestionGenerator
)
from .search_orchestrator import SearchOrchestrator
from .synthesis_worker_production import SynthesisWorker
from .privacy_gates import ConsentManager, PIIDetector, DataRedactor
from .error_handling import ResilientExecutor, ErrorAggregator


@dataclass
class DependencyStatus:
    """Status of a dependency."""
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'available': self.available,
            'version': self.version,
            'error': self.error
        }


class DependencyChecker:
    """Check and validate dependencies."""
    
    @staticmethod
    def check_all() -> Dict[str, DependencyStatus]:
        """Check all optional dependencies."""
        dependencies = {}
        
        # Check vector search dependencies
        try:
            import sentence_transformers
            import faiss
            dependencies['vector_search'] = DependencyStatus(
                name='vector_search',
                available=True,
                version=f"sentence_transformers={sentence_transformers.__version__}"
            )
        except ImportError as e:
            dependencies['vector_search'] = DependencyStatus(
                name='vector_search',
                available=False,
                error=str(e)
            )
        
        # Check FTS dependencies
        try:
            import whoosh
            dependencies['fts'] = DependencyStatus(
                name='fts',
                available=True,
                version=f"whoosh={whoosh.__version__}"
            )
        except ImportError as e:
            dependencies['fts'] = DependencyStatus(
                name='fts',
                available=False,
                error=str(e)
            )
        
        # Check ML dependencies
        try:
            import sklearn
            import pandas
            import numpy
            dependencies['ml'] = DependencyStatus(
                name='ml',
                available=True,
                version=f"sklearn={sklearn.__version__}"
            )
        except ImportError as e:
            dependencies['ml'] = DependencyStatus(
                name='ml',
                available=False,
                error=str(e)
            )
        
        # Check privacy dependencies
        try:
            import spacy
            import cryptography
            dependencies['privacy'] = DependencyStatus(
                name='privacy',
                available=True,
                version=f"spacy={spacy.__version__}"
            )
        except ImportError as e:
            dependencies['privacy'] = DependencyStatus(
                name='privacy',
                available=False,
                error=str(e)
            )
        
        # Check analytics dependencies
        try:
            import duckdb
            dependencies['analytics'] = DependencyStatus(
                name='analytics',
                available=True,
                version=f"duckdb={duckdb.__version__}"
            )
        except ImportError as e:
            dependencies['analytics'] = DependencyStatus(
                name='analytics',
                available=False,
                error=str(e)
            )
        
        return dependencies


class ProductionComponentFactory:
    """Factory for creating production components."""
    
    def __init__(self, config: Dict[str, Any], dependency_status: Dict[str, DependencyStatus]):
        """
        Initialize component factory.
        
        Args:
            config: Configuration dictionary
            dependency_status: Status of dependencies
        """
        self.config = config
        self.dependency_status = dependency_status
        self.components = {}
    
    def create_fts_indexer(self, vault_path: Path) -> Any:
        """Create FTS indexer (production or fallback)."""
        if self.dependency_status['fts'].available:
            logger.info("Using production FTS indexer (Whoosh)")
            return FTSIndexProduction(vault_path)
        else:
            logger.warning("Whoosh not available, using basic FTS")
            # Import fallback
            from .indexers.fts import FTSIndexer
            return FTSIndexer(vault_path)
    
    def create_vector_indexer(self, vault_path: Path) -> Any:
        """Create vector indexer (production or fallback)."""
        if self.dependency_status['vector_search'].available:
            logger.info("Using production vector indexer (FAISS + Sentence Transformers)")
            return VectorIndexProduction(
                vault_path,
                model_name=self.config.get('vector_model', 'all-MiniLM-L6-v2'),
                use_gpu=self.config.get('use_gpu', False)
            )
        else:
            logger.warning("Vector dependencies not available, using stub")
            # Import fallback
            from .indexers.vector_stub import VectorIndexer
            return VectorIndexer(vault_path)
    
    def create_search_orchestrator(self, vault_path: Path, event_bus: Any) -> SearchOrchestrator:
        """Create search orchestrator."""
        logger.info("Creating production search orchestrator")
        return SearchOrchestrator(
            vault_path=vault_path,
            event_bus=event_bus,
            enable_cache=self.config.get('enable_search_cache', True)
        )
    
    def create_synthesis_worker(self, vault_path: Path, event_bus: Any) -> SynthesisWorker:
        """Create synthesis worker."""
        if self.dependency_status['ml'].available:
            logger.info("Creating production synthesis worker with ML capabilities")
            return SynthesisWorker(
                vault_path=vault_path,
                event_bus=event_bus,
                interval_minutes=self.config.get('synthesis_interval', 5)
            )
        else:
            logger.warning("ML dependencies not available, synthesis limited")
            # Still create worker but with limited capabilities
            return SynthesisWorker(
                vault_path=vault_path,
                event_bus=event_bus,
                interval_minutes=self.config.get('synthesis_interval', 5)
            )
    
    def create_privacy_manager(self, storage_path: Path) -> ConsentManager:
        """Create privacy manager."""
        if self.dependency_status['privacy'].available:
            logger.info("Creating production privacy manager with NLP")
            return ConsentManager(storage_path)
        else:
            logger.warning("Privacy dependencies limited, using pattern matching only")
            return ConsentManager(storage_path)
    
    def create_suggestion_generator(self) -> SuggestionGenerator:
        """Create suggestion generator."""
        if self.dependency_status['ml'].available:
            logger.info("Creating production suggestion generator")
            return SuggestionGenerator()
        else:
            logger.warning("ML not available, suggestions limited")
            # Import fallback
            from .algorithms import SuggestionGenerator as BasicSuggestionGenerator
            return BasicSuggestionGenerator()
    
    def create_error_handler(self) -> ResilientExecutor:
        """Create error handler."""
        logger.info("Creating production error handler")
        return ResilientExecutor()


class ProductionIntegration:
    """Main production integration class."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize production integration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.dependency_status = DependencyChecker.check_all()
        self.factory = ProductionComponentFactory(self.config, self.dependency_status)
        
        # Core paths
        self.vault_path = Path(self.config.get('vault_path', './vault'))
        self.storage_path = Path(self.config.get('storage_path', './storage'))
        
        # Components (lazy initialization)
        self._event_bus = None
        self._fts_indexer = None
        self._vector_indexer = None
        self._search_orchestrator = None
        self._synthesis_worker = None
        self._privacy_manager = None
        self._suggestion_generator = None
        self._error_handler = None
        
        # Report status
        self._report_status()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration."""
        default_config = {
            'vault_path': './vault',
            'storage_path': './storage',
            'vector_model': 'all-MiniLM-L6-v2',
            'use_gpu': False,
            'enable_search_cache': True,
            'synthesis_interval': 5,
            'privacy_mode': 'strict',
            'error_recovery': True,
            'production_mode': True
        }
        
        if config_path and config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def _report_status(self):
        """Report dependency and configuration status."""
        logger.info("=" * 60)
        logger.info("PRODUCTION INTEGRATION STATUS")
        logger.info("=" * 60)
        
        # Report dependencies
        logger.info("Dependencies:")
        for name, status in self.dependency_status.items():
            if status.available:
                logger.info(f"  ✓ {name}: {status.version or 'available'}")
            else:
                logger.warning(f"  ✗ {name}: not available")
        
        # Report configuration
        logger.info("\nConfiguration:")
        logger.info(f"  Vault path: {self.vault_path}")
        logger.info(f"  Storage path: {self.storage_path}")
        logger.info(f"  Production mode: {self.config.get('production_mode', False)}")
        logger.info(f"  GPU enabled: {self.config.get('use_gpu', False)}")
        
        # Determine mode
        all_available = all(s.available for s in self.dependency_status.values())
        if all_available:
            logger.info("\n✅ FULL PRODUCTION MODE - All features available")
        else:
            missing = [n for n, s in self.dependency_status.items() if not s.available]
            logger.warning(f"\n⚠️  DEGRADED MODE - Missing: {', '.join(missing)}")
        
        logger.info("=" * 60)
    
    @property
    def event_bus(self):
        """Get or create event bus."""
        if self._event_bus is None:
            from .bus import EventBus
            self._event_bus = EventBus()
        return self._event_bus
    
    @property
    def fts_indexer(self):
        """Get or create FTS indexer."""
        if self._fts_indexer is None:
            self._fts_indexer = self.factory.create_fts_indexer(self.vault_path)
        return self._fts_indexer
    
    @property
    def vector_indexer(self):
        """Get or create vector indexer."""
        if self._vector_indexer is None:
            self._vector_indexer = self.factory.create_vector_indexer(self.vault_path)
        return self._vector_indexer
    
    @property
    def search_orchestrator(self):
        """Get or create search orchestrator."""
        if self._search_orchestrator is None:
            self._search_orchestrator = self.factory.create_search_orchestrator(
                self.vault_path,
                self.event_bus
            )
        return self._search_orchestrator
    
    @property
    def synthesis_worker(self):
        """Get or create synthesis worker."""
        if self._synthesis_worker is None:
            self._synthesis_worker = self.factory.create_synthesis_worker(
                self.vault_path,
                self.event_bus
            )
        return self._synthesis_worker
    
    @property
    def privacy_manager(self):
        """Get or create privacy manager."""
        if self._privacy_manager is None:
            self._privacy_manager = self.factory.create_privacy_manager(
                self.storage_path / 'privacy'
            )
        return self._privacy_manager
    
    @property
    def suggestion_generator(self):
        """Get or create suggestion generator."""
        if self._suggestion_generator is None:
            self._suggestion_generator = self.factory.create_suggestion_generator()
        return self._suggestion_generator
    
    @property
    def error_handler(self):
        """Get or create error handler."""
        if self._error_handler is None:
            self._error_handler = self.factory.create_error_handler()
        return self._error_handler
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing production components...")
        
        # Create directories
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize indexes
        if self.dependency_status['fts'].available:
            await self.error_handler.execute(
                'fts_init',
                self._initialize_fts,
                recover=True
            )
        
        if self.dependency_status['vector_search'].available:
            await self.error_handler.execute(
                'vector_init',
                self._initialize_vector,
                recover=True
            )
        
        # Start synthesis worker
        await self.synthesis_worker.start()
        
        logger.info("Production initialization complete")
    
    async def _initialize_fts(self):
        """Initialize FTS index."""
        doc_count = await self.fts_indexer.reindex_vault()
        logger.info(f"FTS index initialized with {doc_count} documents")
    
    async def _initialize_vector(self):
        """Initialize vector index."""
        doc_count = await self.vector_indexer.reindex_vault()
        logger.info(f"Vector index initialized with {doc_count} documents")
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down production components...")
        
        # Stop synthesis worker
        if self._synthesis_worker:
            await self.synthesis_worker.stop()
        
        # Save indexes
        if self._vector_indexer and hasattr(self._vector_indexer, '_save_index'):
            self._vector_indexer._save_index()
        
        logger.info("Production shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        status = {
            'mode': 'production' if all(s.available for s in self.dependency_status.values()) else 'degraded',
            'dependencies': {
                name: status.to_dict()
                for name, status in self.dependency_status.items()
            },
            'configuration': {
                'vault_path': str(self.vault_path),
                'storage_path': str(self.storage_path),
                'production_mode': self.config.get('production_mode', False)
            },
            'components': {}
        }
        
        # Component health
        if self._error_handler:
            status['health'] = self.error_handler.get_health_status()
        
        # Search statistics
        if self._search_orchestrator:
            asyncio.create_task(self._add_search_stats(status))
        
        # Synthesis status
        if self._synthesis_worker:
            status['synthesis'] = self.synthesis_worker.get_latest_insights()
        
        return status
    
    async def _add_search_stats(self, status: Dict):
        """Add search statistics to status."""
        stats = await self.search_orchestrator.get_statistics()
        status['components']['search'] = stats


# Global instance
_production_instance: Optional[ProductionIntegration] = None


def get_production_integration(config_path: Optional[Path] = None) -> ProductionIntegration:
    """Get or create production integration instance."""
    global _production_instance
    
    if _production_instance is None:
        _production_instance = ProductionIntegration(config_path)
    
    return _production_instance


async def run_production_mode(config_path: Optional[Path] = None):
    """Run in full production mode."""
    logger.info("Starting Symbiote in PRODUCTION MODE")
    
    # Set production environment
    os.environ['SYMBIOTE_MODE'] = 'production'
    
    # Initialize production integration
    integration = get_production_integration(config_path)
    
    try:
        # Initialize components
        await integration.initialize()
        
        # Log status
        status = integration.get_status()
        logger.info(f"Production status: {status['mode']}")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Periodic health check
            health = integration.error_handler.get_health_status()
            if health.get('error_summary', {}).get('recent_errors', 0) > 100:
                logger.warning("High error rate detected in production")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await integration.shutdown()


if __name__ == "__main__":
    # Run in production mode
    asyncio.run(run_production_mode())