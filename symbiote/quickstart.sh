#!/bin/bash

# Symbiote Quick Start Script

echo "🧠 Symbiote - Cognitive Prosthetic Quick Start"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version detected"

# Install dependencies
echo ""
echo "Installing dependencies..."
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install
else
    echo "Using pip..."
    pip install -r requirements.txt
fi

# Initialize vault
echo ""
echo "Initializing vault with example data..."
python scripts/setup.py --vault ./vault

# Update config to use local vault
echo ""
echo "Updating configuration..."
sed -i.bak 's|vault_path:.*|vault_path: "./vault"|' symbiote.yaml

# Start daemon in background
echo ""
echo "Starting daemon..."
python -m symbiote.cli.sym daemon start &
DAEMON_PID=$!

# Wait for daemon to start
sleep 3

# Run health check
echo ""
echo "Running health check..."
python scripts/doctor.py

echo ""
echo "=============================================="
echo "✅ Symbiote is ready!"
echo ""
echo "Try these commands:"
echo "  sym 'Remember to review the Q3 roadmap'"
echo "  sym search 'API design'"
echo "  sym daemon status"
echo ""
echo "Daemon PID: $DAEMON_PID"
echo "Stop with: kill $DAEMON_PID"
echo ""