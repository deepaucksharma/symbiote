# 🚀 Symbiote Quickstart Guide

Get up and running with Symbiote in under 5 minutes!

## Prerequisites

- Python 3.9+
- 2GB free RAM
- 500MB disk space

## 1. Installation (30 seconds)

```bash
# Clone the repository (if needed)
git clone <repository-url> symbiote
cd symbiote

# Install dependencies
pip install -r requirements.txt
```

## 2. Basic Setup (1 minute)

```bash
# Create your vault directory
mkdir -p vault/{daily,notes,tasks,projects}

# Copy default configuration
cp symbiote.yaml.example symbiote.yaml

# Edit configuration (optional)
# vim symbiote.yaml
```

## 3. Start the Daemon (10 seconds)

```bash
# Option 1: Using Make
make daemon

# Option 2: Direct Python
python -m symbiote.daemon.main

# Option 3: Background with logging
python -m symbiote.daemon.main > daemon.log 2>&1 &
```

## 4. Your First Capture (instant!)

```bash
# Using curl
curl -X POST http://localhost:8765/capture \
  -H "Content-Type: application/json" \
  -d '{"text": "My first thought in Symbiote!"}'

# Using the CLI (when available)
sym capture "My first thought in Symbiote!"
```

## 5. Search Your Thoughts

```bash
# Search for content
curl "http://localhost:8765/context?q=first+thought"

# Using CLI
sym search "first thought"
```

## 6. Get Suggestions

```bash
# Request a suggestion
curl -X POST http://localhost:8765/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "situation": {
      "query": "work",
      "free_minutes": 30
    }
  }'
```

## 🎯 Quick Demo

Run the interactive demo to see all features:

```bash
# Run the full demo
make demo

# Or directly
python validation/demo_flow.py
```

## 📊 Check System Health

```bash
# Health check
curl http://localhost:8765/health

# View metrics
curl http://localhost:8765/metrics?format=json

# Check daemon status
make status
```

## 🔧 Common Commands

| Task | Command |
|------|---------|
| Start daemon | `make daemon` |
| Stop daemon | `make daemon-stop` |
| Run tests | `make test` |
| Check validation | `make validate` |
| View logs | `tail -f daemon.log` |
| Clean up | `make clean` |

## 💡 Tips for Best Experience

### 1. Configure Your Vault Location
Edit `symbiote.yaml`:
```yaml
vault_path: /path/to/your/notes
```

### 2. Set Up Hotkeys (Optional)
Add to your window manager config:
```bash
# Example for i3/sway
bindsym $mod+c exec "sym capture"
bindsym $mod+s exec "sym search"
```

### 3. Enable Auto-start
Add to your shell profile:
```bash
# ~/.bashrc or ~/.zshrc
symbiote start 2>/dev/null || true
```

## 🚨 Troubleshooting

### Daemon won't start
```bash
# Check if port is in use
lsof -i :8765

# Kill existing daemon
pkill -f "python.*daemon.main"

# Start fresh
make daemon
```

### High latency
```bash
# Rebuild indexes
curl -X POST http://localhost:8765/admin/reindex

# Check memory usage
ps aux | grep daemon.main
```

### Missing dependencies
```bash
# Install core dependencies only
pip install aiohttp pydantic pyyaml loguru

# The system will use mocks for missing packages
```

## 📖 Next Steps

1. **Explore the API**: See [API Documentation](README.md#api)
2. **Configure Privacy**: Edit privacy settings in `symbiote.yaml`
3. **Run Validation**: `make validate` to check everything works
4. **Read Architecture**: See [CLAUDE.md](CLAUDE.md) for deep dive

## 🎉 You're Ready!

Your Symbiote is now running and ready to augment your thinking:

- ✅ **Capture** thoughts with <200ms latency
- ✅ **Search** with <100ms response time  
- ✅ **Get suggestions** with full explainability
- ✅ **Stay private** with local-first processing

Start capturing your thoughts and let Symbiote help you think better!

---

*Need help? Check the [full documentation](README.md) or [troubleshooting guide](DEPLOYMENT.md#troubleshooting)*