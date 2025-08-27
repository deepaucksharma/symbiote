#!/usr/bin/env python3
"""
Master Functional Test Runner for Symbiote
Orchestrates and runs all functional test suites with comprehensive reporting
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import all test modules
from test_framework import FunctionalTestFramework
from test_user_journeys import UserJourneyTester

class TestSuite(Enum):
    """Available test suites"""
    FRAMEWORK = "framework"
    JOURNEYS = "journeys"
    ALL = "all"

@dataclass
class TestResult:
    """Result from a test suite"""
    suite: str
    passed: int
    failed: int
    duration: float
    details: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class MasterTestRunner:
    """Orchestrates all functional tests"""
    
    def __init__(self, verbose: bool = False, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    async def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run specified test suite(s)"""
        print("\n" + "=" * 80)
        print("🚀 SYMBIOTE COMPREHENSIVE FUNCTIONAL TEST SUITE")
        print("=" * 80)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'Parallel' if self.parallel else 'Sequential'}")
        print(f"Verbosity: {'High' if self.verbose else 'Normal'}")
        
        self.start_time = time.time()
        
        if suite == TestSuite.ALL:
            await self._run_all_suites()
        else:
            await self._run_single_suite(suite)
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self._generate_master_report()
        
        return self.results
    
    async def _run_all_suites(self):
        """Run all test suites"""
        suites = [
            (TestSuite.FRAMEWORK, self._run_framework_tests),
            (TestSuite.JOURNEYS, self._run_journey_tests)
        ]
        
        if self.parallel:
            print("\n🔀 Running test suites in PARALLEL...")
            tasks = [test_func() for _, test_func in suites]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (suite_type, _), result in zip(suites, results):
                if isinstance(result, Exception):
                    self.results.append(TestResult(
                        suite=suite_type.value,
                        passed=0,
                        failed=1,
                        duration=0,
                        errors=[str(result)]
                    ))
                else:
                    self.results.append(result)
        else:
            print("\n📝 Running test suites SEQUENTIALLY...")
            for suite_type, test_func in suites:
                try:
                    result = await test_func()
                    self.results.append(result)
                except Exception as e:
                    self.results.append(TestResult(
                        suite=suite_type.value,
                        passed=0,
                        failed=1,
                        duration=0,
                        errors=[str(e)]
                    ))
    
    async def _run_single_suite(self, suite: TestSuite):
        """Run a single test suite"""
        suite_map = {
            TestSuite.FRAMEWORK: self._run_framework_tests,
            TestSuite.JOURNEYS: self._run_journey_tests
        }
        
        if suite in suite_map:
            result = await suite_map[suite]()
            self.results.append(result)
    
    async def _run_framework_tests(self) -> TestResult:
        """Run framework tests"""
        print("\n" + "-" * 60)
        print("📦 FRAMEWORK TESTS")
        print("-" * 60)
        
        start = time.time()
        passed = 0
        failed = 0
        
        try:
            symbiote_root = Path("/home/deepak/src/Tormentum/symbiote")
            framework = FunctionalTestFramework(symbiote_root)
            
            # Run basic tests
            print("\n🔹 Running Basic Tests...")
            await framework._run_basic_tests()
            
            # Run intermediate tests
            print("\n🔹 Running Intermediate Tests...")
            await framework._run_intermediate_tests()
            
            # Run advanced tests
            print("\n🔹 Running Advanced Tests...")
            await framework._run_advanced_tests()
            
            # Analyze results
            for result in framework.results:
                if result["success"]:
                    passed += 1
                else:
                    failed += 1
            
        except Exception as e:
            print(f"❌ Framework tests failed: {e}")
            failed += 1
        
        duration = time.time() - start
        
        return TestResult(
            suite="framework",
            passed=passed,
            failed=failed,
            duration=duration
        )
    
    async def _run_journey_tests(self) -> TestResult:
        """Run user journey tests"""
        print("\n" + "-" * 60)
        print("🚶 USER JOURNEY TESTS")
        print("-" * 60)
        
        start = time.time()
        passed = 0
        failed = 0
        errors = []
        
        try:
            tester = UserJourneyTester()
            await tester.setup()
            
            try:
                # Run all journeys
                journeys = [
                    ("Researcher Deep Dive", tester.test_researcher_deep_dive()),
                    ("Developer Debugging", tester.test_developer_debugging_session()),
                    ("Student Study Session", tester.test_student_study_session()),
                    ("Creative Brainstorming", tester.test_creative_brainstorming()),
                    ("Power User Workflow", tester.test_power_user_complex_workflow())
                ]
                
                for name, journey_coro in journeys:
                    print(f"\n🔹 {name}")
                    result = await journey_coro
                    if result["success"]:
                        passed += 1
                    else:
                        failed += 1
                        if result.get("errors"):
                            errors.extend(result["errors"])
                
            finally:
                await tester.teardown()
                
        except Exception as e:
            print(f"❌ Journey tests failed: {e}")
            failed += 1
            errors.append(str(e))
        
        duration = time.time() - start
        
        return TestResult(
            suite="journeys",
            passed=passed,
            failed=failed,
            duration=duration,
            errors=errors
        )
    
    
    def _generate_master_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 MASTER TEST REPORT")
        print("=" * 80)
        
        if not self.results:
            print("No test results to report")
            return
        
        total_duration = self.end_time - self.start_time
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_tests = total_passed + total_failed
        
        # Summary statistics
        print("\n📈 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Total Tests Run: {total_tests}")
        print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        print(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Average Test Time: {total_duration/total_tests:.2f} seconds")
        
        # Suite breakdown
        print("\n📦 SUITE BREAKDOWN")
        print("-" * 40)
        print(f"{'Suite':<15} {'Passed':<10} {'Failed':<10} {'Time (s)':<10} {'Status':<10}")
        print("-" * 60)
        
        for result in self.results:
            status = "✅ PASS" if result.failed == 0 else "❌ FAIL"
            print(f"{result.suite:<15} {result.passed:<10} {result.failed:<10} "
                  f"{result.duration:<10.2f} {status:<10}")
        
        # Detailed failures
        if total_failed > 0:
            print("\n⚠️  FAILURES SUMMARY")
            print("-" * 40)
            
            for result in self.results:
                if result.failed > 0:
                    print(f"\n{result.suite.upper()}:")
                    if result.errors:
                        for error in result.errors[:5]:  # Show first 5 errors
                            print(f"  • {error}")
                    else:
                        print(f"  • {result.failed} test(s) failed")
        
        # Performance insights
        print("\n⚡ PERFORMANCE INSIGHTS")
        print("-" * 40)
        
        # Find slowest suite
        slowest = max(self.results, key=lambda r: r.duration)
        print(f"Slowest Suite: {slowest.suite} ({slowest.duration:.2f}s)")
        
        # Find most failures
        most_failures = max(self.results, key=lambda r: r.failed)
        if most_failures.failed > 0:
            print(f"Most Failures: {most_failures.suite} ({most_failures.failed} failures)")
        
        # Overall verdict
        print("\n" + "=" * 80)
        if total_failed == 0:
            print("🎉 ALL FUNCTIONAL TESTS PASSED! 🎉")
            print("The Symbiote system has passed comprehensive functional validation.")
        else:
            print(f"⚠️  {total_failed} TESTS FAILED")
            print("Please review the failures above and fix the issues.")
        print("=" * 80)
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save test results to JSON file"""
        results_dir = Path("/home/deepak/src/Tormentum/symbiote/tests/functional/results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"functional_test_results_{timestamp}.json"
        
        results_data = {
            "timestamp": timestamp,
            "duration": self.end_time - self.start_time,
            "total_passed": sum(r.passed for r in self.results),
            "total_failed": sum(r.failed for r in self.results),
            "suites": [
                {
                    "name": r.suite,
                    "passed": r.passed,
                    "failed": r.failed,
                    "duration": r.duration,
                    "errors": r.errors
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Symbiote Functional Test Suite"
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["framework", "journeys", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run test suites in parallel"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Map string to enum
    suite_map = {
        "framework": TestSuite.FRAMEWORK,
        "journeys": TestSuite.JOURNEYS,
        "all": TestSuite.ALL
    }
    
    suite = suite_map[args.suite]
    
    # Run tests
    runner = MasterTestRunner(verbose=args.verbose, parallel=args.parallel)
    results = await runner.run_suite(suite)
    
    # Exit with appropriate code
    total_failed = sum(r.failed for r in results)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())