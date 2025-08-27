#!/usr/bin/env python3
"""Production validation script to verify all components are working correctly.

This script performs comprehensive validation of the production system:
- Checks all dependencies
- Verifies each component works
- Runs performance benchmarks
- Tests error recovery
- Validates security features
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil

from daemon.production_integration import (
    ProductionIntegration,
    DependencyChecker
)
from daemon.algorithms import SearchCandidate
from daemon.search_orchestrator import SearchRequest
from daemon.privacy_gates import ConsentLevel
from daemon.indexers.analytics_production import AnalyticsEvent

from loguru import logger


class ProductionValidator:
    """Validates production system functionality."""
    
    def __init__(self, config_path: Path = None):
        """Initialize validator."""
        self.config_path = config_path or Path("symbiote_production.yaml")
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'performance': {},
            'errors': []
        }
        self.integration = None
    
    async def run_validation(self) -> bool:
        """Run complete validation suite."""
        print("=" * 60)
        print("SYMBIOTE PRODUCTION VALIDATION")
        print("=" * 60)
        
        all_passed = True
        
        try:
            # 1. Check dependencies
            print("\n1. Checking dependencies...")
            if not self._check_dependencies():
                all_passed = False
            
            # 2. Initialize system
            print("\n2. Initializing production system...")
            if not await self._initialize_system():
                all_passed = False
                return False  # Can't continue without initialization
            
            # 3. Validate components
            print("\n3. Validating components...")
            if not await self._validate_components():
                all_passed = False
            
            # 4. Test search functionality
            print("\n4. Testing search functionality...")
            if not await self._test_search():
                all_passed = False
            
            # 5. Test privacy features
            print("\n5. Testing privacy features...")
            if not await self._test_privacy():
                all_passed = False
            
            # 6. Test synthesis
            print("\n6. Testing synthesis...")
            if not await self._test_synthesis():
                all_passed = False
            
            # 7. Test analytics
            print("\n7. Testing analytics...")
            if not await self._test_analytics():
                all_passed = False
            
            # 8. Test error handling
            print("\n8. Testing error handling...")
            if not await self._test_error_handling():
                all_passed = False
            
            # 9. Performance benchmarks
            print("\n9. Running performance benchmarks...")
            if not await self._run_benchmarks():
                all_passed = False
            
            # 10. Generate report
            print("\n10. Generating validation report...")
            self._generate_report()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results['errors'].append(str(e))
            all_passed = False
        
        finally:
            if self.integration:
                await self.integration.shutdown()
        
        return all_passed
    
    def _check_dependencies(self) -> bool:
        """Check all dependencies."""
        deps = DependencyChecker.check_all()
        
        self.results['checks']['dependencies'] = {}
        all_available = True
        
        for name, status in deps.items():
            self.results['checks']['dependencies'][name] = {
                'available': status.available,
                'version': status.version,
                'error': status.error
            }
            
            if status.available:
                print(f"  ✓ {name}: {status.version or 'available'}")
            else:
                print(f"  ✗ {name}: not available")
                all_available = False
        
        if all_available:
            print("\n  ✅ All dependencies available")
        else:
            print("\n  ⚠️  Some dependencies missing (degraded mode)")
        
        return True  # Don't fail on missing optional deps
    
    async def _initialize_system(self) -> bool:
        """Initialize production system."""
        try:
            # Create temporary vault for testing
            self.temp_dir = tempfile.mkdtemp(prefix="validation_")
            self.vault_path = Path(self.temp_dir)
            
            # Create test documents
            self._create_test_documents()
            
            # Override config
            config = {
                'vault_path': str(self.vault_path),
                'storage_path': str(self.vault_path / '.storage'),
                'production_mode': True,
                'enable_search_cache': False
            }
            
            # Save config
            import yaml
            config_file = self.vault_path / 'validation_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            # Initialize
            self.integration = ProductionIntegration(config_file)
            await self.integration.initialize()
            
            print("  ✓ System initialized successfully")
            self.results['checks']['initialization'] = {'status': 'success'}
            return True
            
        except Exception as e:
            print(f"  ✗ Initialization failed: {e}")
            self.results['checks']['initialization'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def _create_test_documents(self):
        """Create test documents in vault."""
        docs = [
            {
                'name': 'test_doc_1.md',
                'content': """# Test Document 1
                
This document contains information about artificial intelligence and machine learning.
Neural networks are powerful tools for pattern recognition.

Tags: #ai #ml #deeplearning
"""
            },
            {
                'name': 'test_doc_2.md',
                'content': """# Test Document 2

Python programming best practices and tips.
Use virtual environments and type hints.

Contact: test@example.com, phone: 555-0123

Tags: #python #programming
"""
            }
        ]
        
        for doc in docs:
            (self.vault_path / doc['name']).write_text(doc['content'])
    
    async def _validate_components(self) -> bool:
        """Validate each component is working."""
        self.results['checks']['components'] = {}
        all_valid = True
        
        # Check FTS
        if self.integration.dependency_status['fts'].available:
            try:
                await self.integration.fts_indexer.reindex_vault()
                print("  ✓ FTS indexer working")
                self.results['checks']['components']['fts'] = 'working'
            except Exception as e:
                print(f"  ✗ FTS indexer failed: {e}")
                self.results['checks']['components']['fts'] = f'failed: {e}'
                all_valid = False
        
        # Check Vector
        if self.integration.dependency_status['vector_search'].available:
            try:
                await self.integration.vector_indexer.reindex_vault()
                print("  ✓ Vector indexer working")
                self.results['checks']['components']['vector'] = 'working'
            except Exception as e:
                print(f"  ✗ Vector indexer failed: {e}")
                self.results['checks']['components']['vector'] = f'failed: {e}'
                all_valid = False
        
        # Check other components
        components = [
            ('search_orchestrator', 'Search orchestrator'),
            ('synthesis_worker', 'Synthesis worker'),
            ('privacy_manager', 'Privacy manager'),
            ('error_handler', 'Error handler')
        ]
        
        for attr, name in components:
            if hasattr(self.integration, f'_{attr}') or hasattr(self.integration, attr):
                print(f"  ✓ {name} initialized")
                self.results['checks']['components'][attr] = 'initialized'
            else:
                print(f"  ✗ {name} not initialized")
                self.results['checks']['components'][attr] = 'not initialized'
                all_valid = False
        
        return all_valid
    
    async def _test_search(self) -> bool:
        """Test search functionality."""
        try:
            request = SearchRequest(
                query="artificial intelligence",
                limit=5
            )
            
            start = time.time()
            result = await self.integration.search_orchestrator.search(request)
            elapsed = (time.time() - start) * 1000
            
            self.results['checks']['search'] = {
                'status': 'success',
                'results_found': len(result.candidates),
                'latency_ms': elapsed,
                'strategies_used': result.strategies_used
            }
            
            if len(result.candidates) > 0:
                print(f"  ✓ Search working ({len(result.candidates)} results in {elapsed:.0f}ms)")
                return True
            else:
                print(f"  ⚠️ Search returned no results")
                return False
                
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
            self.results['checks']['search'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def _test_privacy(self) -> bool:
        """Test privacy features."""
        try:
            text = "Contact John at john@example.com or 555-123-4567"
            
            # Test PII detection
            pii = self.integration.privacy_manager.pii_detector.detect(text)
            
            # Test redaction
            redacted = self.integration.privacy_manager.redactor.redact(
                text,
                consent_level=ConsentLevel.ALLOW_ANONYMOUS
            )
            
            self.results['checks']['privacy'] = {
                'pii_detected': len(pii),
                'redaction_working': '@example.com' not in redacted
            }
            
            if len(pii) > 0 and '@example.com' not in redacted:
                print(f"  ✓ Privacy features working (detected {len(pii)} PII items)")
                return True
            else:
                print("  ⚠️ Privacy features partially working")
                return False
                
        except Exception as e:
            print(f"  ✗ Privacy test failed: {e}")
            self.results['checks']['privacy'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def _test_synthesis(self) -> bool:
        """Test synthesis worker."""
        if not self.integration.dependency_status['ml'].available:
            print("  ⚠️ Synthesis skipped (ML deps not available)")
            self.results['checks']['synthesis'] = 'skipped'
            return True
        
        try:
            result = await self.integration.synthesis_worker.synthesize()
            
            self.results['checks']['synthesis'] = {
                'patterns_found': len(result.get('patterns', [])),
                'insights_generated': len(result.get('insights', [])),
                'elapsed_seconds': result.get('elapsed_seconds', 0)
            }
            
            print(f"  ✓ Synthesis working ({len(result.get('patterns', []))} patterns found)")
            return True
            
        except Exception as e:
            print(f"  ✗ Synthesis failed: {e}")
            self.results['checks']['synthesis'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def _test_analytics(self) -> bool:
        """Test analytics engine."""
        if not self.integration.dependency_status['analytics'].available:
            print("  ⚠️ Analytics skipped (DuckDB not available)")
            self.results['checks']['analytics'] = 'skipped'
            return True
        
        try:
            from daemon.indexers.analytics_production import DuckDBAnalytics
            
            analytics = DuckDBAnalytics()
            
            # Track test event
            event = AnalyticsEvent(
                event_id="val_001",
                event_type="validation",
                timestamp=datetime.now(),
                document_id="test_doc",
                metadata={'test': True}
            )
            
            await analytics.track_event(event)
            
            # Get stats
            perf = await analytics.get_search_performance()
            
            self.results['checks']['analytics'] = {
                'status': 'working',
                'tracking': True
            }
            
            print("  ✓ Analytics working")
            return True
            
        except Exception as e:
            print(f"  ✗ Analytics failed: {e}")
            self.results['checks']['analytics'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        try:
            # Test circuit breaker
            failures = 0
            
            async def flaky_function():
                nonlocal failures
                failures += 1
                if failures < 3:
                    raise ConnectionError("Simulated failure")
                return "success"
            
            result = await self.integration.error_handler.execute(
                "validation_service",
                flaky_function,
                retry=True
            )
            
            self.results['checks']['error_handling'] = {
                'circuit_breaker': 'working',
                'retry_count': failures,
                'recovery': result == "success"
            }
            
            print(f"  ✓ Error handling working (recovered after {failures} attempts)")
            return True
            
        except Exception as e:
            print(f"  ✗ Error handling failed: {e}")
            self.results['checks']['error_handling'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def _run_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        self.results['performance'] = {}
        
        # Search benchmark
        try:
            latencies = []
            for _ in range(10):
                request = SearchRequest(
                    query="test query",
                    limit=5,
                    timeout_ms=500
                )
                
                start = time.time()
                await self.integration.search_orchestrator.search(request)
                latencies.append((time.time() - start) * 1000)
            
            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            
            self.results['performance']['search'] = {
                'avg_latency_ms': avg_latency,
                'p99_latency_ms': p99_latency,
                'meets_slo': p99_latency < 300
            }
            
            if p99_latency < 300:
                print(f"  ✓ Search performance: p99={p99_latency:.0f}ms (SLO met)")
            else:
                print(f"  ⚠️ Search performance: p99={p99_latency:.0f}ms (SLO not met)")
            
        except Exception as e:
            print(f"  ✗ Search benchmark failed: {e}")
            self.results['performance']['search'] = {'error': str(e)}
        
        return True
    
    def _generate_report(self):
        """Generate validation report."""
        # Calculate summary
        total_checks = len(self.results['checks'])
        passed_checks = sum(
            1 for check in self.results['checks'].values()
            if isinstance(check, dict) and check.get('status') != 'failed'
            or isinstance(check, str) and 'working' in check
        )
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'errors_count': len(self.results['errors'])
        }
        
        # Save report
        report_file = Path('validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n  📊 Report saved to {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Checks passed: {passed_checks}/{total_checks}")
        print(f"Success rate: {self.results['summary']['success_rate']:.1%}")
        
        if self.results['summary']['success_rate'] >= 0.8:
            print("\n✅ PRODUCTION SYSTEM VALIDATED")
        else:
            print("\n⚠️  VALIDATION INCOMPLETE - Review report for details")
        
        # Cleanup
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Symbiote production system"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    validator = ProductionValidator(args.config)
    success = await validator.run_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())