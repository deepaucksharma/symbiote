#!/bin/bash
# Symbiote Run Script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start daemon
echo "Starting Symbiote daemon..."
python -m symbiote.daemon.main
