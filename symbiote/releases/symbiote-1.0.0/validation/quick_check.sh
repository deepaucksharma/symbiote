#!/bin/bash
# Quick validation checklist for Symbiote implementation
# Run this to verify current implementation integrity

set -e

echo "======================================"
echo "Symbiote Quick Validation Check"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Helper function
check() {
    local test_name=$1
    local command=$2
    
    echo -n "Checking $test_name... "
    
    if eval $command > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC}"
        ((FAILED++))
        return 1
    fi
}

echo "1. STRUCTURE VALIDATION"
echo "-----------------------"

check "Project structure" "test -d daemon"
check "Daemon module" "test -f daemon/main.py"
check "Capture service" "test -f daemon/capture.py"
check "Search orchestrator" "test -f daemon/search.py"
check "Algorithms module" "test -f daemon/algorithms.py"
check "Consent manager" "test -f daemon/consent.py"
check "Event bus" "test -f daemon/bus.py"
check "Metrics collector" "test -f daemon/metrics.py"
check "FTS indexer" "test -f daemon/indexers/fts.py"
check "Vector indexer" "test -f daemon/indexers/vector.py"
check "Analytics indexer" "test -f daemon/indexers/analytics.py"
check "Synthesis worker" "test -f daemon/synthesis_worker.py"

echo ""
echo "2. CLI TOOLS"
echo "------------"

check "Main CLI (sym)" "test -f cli/sym.py"
check "Search in CLI" "grep -q 'def search' cli/sym.py"
check "Doctor in CLI" "grep -q 'def doctor' cli/sym.py"

echo ""
echo "3. TESTING INFRASTRUCTURE"
echo "-------------------------"

check "Unit tests" "test -f tests/test_algorithms.py"
check "Integration tests" "test -d tests/integration"
check "E2E tests" "test -f tests/integration/test_end_to_end.py"
check "Security tests" "test -f tests/test_security_privacy.py"
check "Chaos tests" "test -f scripts/chaos_inject.py"
check "Benchmarks" "test -f scripts/run_benchmarks.py"
check "Eval harness" "test -f scripts/eval_retrieval.py"
check "Vault generator" "test -f scripts/gen_vault.py"

echo ""
echo "4. CONFIGURATION"
echo "----------------"

check "Requirements" "test -f requirements.txt"
check "Config module" "test -f daemon/config.py"
check "Models" "test -f daemon/models.py"
check "Documentation" "test -f CLAUDE.md"

echo ""
echo "5. VALIDATION ASSETS"
echo "--------------------"

check "Validation plan" "test -f validation/validation_plan.md"
check "Validation runner" "test -f validation/run_validation.py"

echo ""
echo "6. PYTHON IMPORTS"
echo "-----------------"

echo -n "Checking Python imports... "
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from symbiote.daemon import capture, search, bus, metrics
    from symbiote.daemon.algorithms import SearchCandidate, SuggestionGenerator
    from symbiote.daemon.consent import ConsentManager, RedactionEngine
    print('✓')
    sys.exit(0)
except ImportError as e:
    print(f'✗ ({e})')
    sys.exit(1)
" && ((PASSED++)) || ((FAILED++))

echo ""
echo "7. KEY REQUIREMENTS CHECK"
echo "-------------------------"

echo -n "Checking key dependencies... "
python3 -c "
import importlib
required = ['aiofiles', 'aiohttp', 'pyyaml', 'pydantic', 'ulid', 'click', 
            'rich', 'loguru', 'duckdb', 'faker', 'tabulate']
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg.replace('-', '_'))
    except:
        missing.append(pkg)

if missing:
    print(f'✗ Missing: {\", \".join(missing)}')
    exit(1)
else:
    print('✓')
" && ((PASSED++)) || ((FAILED++))

echo ""
echo "8. CRITICAL CODE PATTERNS"
echo "-------------------------"

echo -n "WAL implementation... "
grep -q "class WAL:" daemon/capture.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo -n "Racing search strategy... "
grep -q "class SearchRacer" daemon/algorithms.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo -n "Utility scoring... "
grep -q "calculate_utility" daemon/algorithms.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo -n "Receipts system... "
grep -q "class Receipt" daemon/models.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo -n "Consent gates... "
grep -q "class ConsentManager" daemon/consent.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo -n "PII redaction... "
grep -q "class RedactionEngine" daemon/consent.py 2>/dev/null && echo -e "${GREEN}✓${NC}" && ((PASSED++)) || (echo -e "${RED}✗${NC}" && ((FAILED++)))

echo ""
echo "======================================"
echo "VALIDATION SUMMARY"
echo "======================================"
echo ""

TOTAL=$((PASSED + FAILED))
PASS_RATE=$((PASSED * 100 / TOTAL))

echo "Total checks: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Pass rate: $PASS_RATE%"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Implementation is complete.${NC}"
    exit 0
elif [ $PASS_RATE -ge 90 ]; then
    echo -e "${YELLOW}⚠️  Most checks passed but some issues remain.${NC}"
    exit 1
else
    echo -e "${RED}❌ Significant issues found. Review implementation.${NC}"
    exit 1
fi