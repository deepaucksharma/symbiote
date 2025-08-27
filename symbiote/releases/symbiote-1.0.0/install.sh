#!/bin/bash
# Symbiote Installation Script

echo "Installing Symbiote..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '3\.\d+')
if [[ $(echo "$python_version < 3.9" | bc) -eq 1 ]]; then
    echo "❌ Python 3.9+ required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create vault structure
echo "Setting up vault..."
mkdir -p vault/{daily,notes,tasks,projects,synthesis}

# Copy configuration
if [ ! -f symbiote.yaml ]; then
    cp symbiote.yaml.example symbiote.yaml
    echo "✅ Created symbiote.yaml (please edit)"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit symbiote.yaml to configure your vault"
echo "2. Run: source venv/bin/activate"
echo "3. Start daemon: python -m symbiote.daemon.main"
echo "4. Or use: make daemon"
