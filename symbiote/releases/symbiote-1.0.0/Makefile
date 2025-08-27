# Symbiote Makefile
.PHONY: help install test run daemon validate demo clean benchmark

help:
	@echo "Symbiote Development Commands"
	@echo "=============================="
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make daemon     - Start daemon in background"
	@echo "  make validate   - Run validation suite"
	@echo "  make demo       - Run interactive demo"
	@echo "  make benchmark  - Run performance tests"
	@echo "  make clean      - Clean generated files"
	@echo ""
	@echo "Quick Start: make install && make daemon && make demo"

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

test:
	@echo "Running test suite..."
	@pytest tests/ -v --tb=short

daemon:
	@echo "Starting daemon in background..."
	@python -m symbiote.daemon.main > daemon.log 2>&1 &
	@sleep 2
	@echo "✓ Daemon started"

daemon-stop:
	@pkill -f 'python.*daemon.main' || true

validate:
	@python validation/run_validation.py

validate-quick:
	@bash validation/quick_check.sh

demo:
	@python validation/demo_flow.py

benchmark:
	@python scripts/run_benchmarks.py --all --duration 60

clean:
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -f daemon.log validation_report.json

status:
	@curl -s http://localhost:8765/health | python -m json.tool || echo "Daemon not running"