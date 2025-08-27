# Symbiote Makefile
.PHONY: help install test run daemon validate demo clean benchmark production

help:
	@echo "Symbiote Commands"
	@echo "================="
	@echo "Development:"
	@echo "  make install        - Install basic dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make daemon        - Start daemon (mock mode)"
	@echo "  make validate      - Run validation suite"
	@echo "  make demo          - Run interactive demo"
	@echo "  make benchmark     - Run performance tests"
	@echo "  make clean         - Clean generated files"
	@echo ""
	@echo "Production:"
	@echo "  make production    - Run in PRODUCTION mode"
	@echo "  make prod-install  - Install production dependencies"
	@echo "  make prod-check    - Check production readiness"
	@echo "  make prod-daemon   - Start production daemon"
	@echo ""
	@echo "Quick Start (Dev): make install && make daemon && make demo"
	@echo "Quick Start (Prod): make prod-install && make production"

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

# Production targets
production:
	@echo "🚀 Launching Symbiote in PRODUCTION mode..."
	@python launch_production.py --config symbiote_production.yaml

prod-install:
	@echo "Installing production dependencies..."
	@pip install -r requirements_production.txt
	@python -m spacy download en_core_web_sm || true
	@echo "✅ Production dependencies installed"

prod-check:
	@echo "Checking production readiness..."
	@python launch_production.py --config symbiote_production.yaml --check-only 2>/dev/null || python -c "from daemon.production_integration import DependencyChecker; import json; print(json.dumps({k: v.to_dict() for k, v in DependencyChecker.check_all().items()}, indent=2))"

prod-daemon:
	@echo "Starting production daemon..."
	@python launch_production.py --config symbiote_production.yaml > production.log 2>&1 &
	@sleep 3
	@echo "✓ Production daemon started"
	@echo "Check logs: tail -f production.log"

prod-stop:
	@pkill -f 'launch_production.py' || true
	@echo "Production daemon stopped"