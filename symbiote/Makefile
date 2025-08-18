.PHONY: help install setup run test lint format clean doctor

help:
	@echo "Symbiote Development Commands"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Initialize vault with example data"
	@echo "  make run        - Start the daemon"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean temporary files"
	@echo "  make doctor     - Run health checks"

install:
	@echo "Installing dependencies..."
	@if command -v poetry > /dev/null; then \
		poetry install; \
	else \
		pip install -r requirements.txt; \
	fi

setup:
	@echo "Setting up vault..."
	@python scripts/setup.py --vault ./vault

run:
	@echo "Starting daemon..."
	@python -m symbiote.cli.sym daemon start

test:
	@echo "Running tests..."
	@pytest tests/ -v

lint:
	@echo "Running linters..."
	@ruff check .
	@mypy symbiote/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	@black symbiote/
	@ruff check --fix .

clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".DS_Store" -delete
	@rm -rf .ruff_cache .mypy_cache .pytest_cache

doctor:
	@echo "Running health checks..."
	@python scripts/doctor.py