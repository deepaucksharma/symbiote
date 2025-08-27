# Symbiote Deployment Guide

## 🚀 Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start daemon
make daemon

# 3. Run demo
make demo

# 4. Validate
make validate
```

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Python 3.9+ installed
- [ ] 2GB RAM available
- [ ] 500MB disk space for indexes
- [ ] Write permissions for vault directory

### Dependencies
```bash
# Core dependencies (automatically installed)
pip install -r symbiote/requirements.txt

# Optional (for full features)
pip install tantivy      # Full-text search
pip install lancedb      # Vector search
pip install sentence-transformers  # Embeddings
```

## 🔧 Configuration

### 1. Basic Configuration
Create `symbiote.yaml` in your vault directory:

```yaml
vault_path: ./vault

performance:
  capture_timeout_ms: 200
  search_timeout_ms: 300
  max_memory_mb: 1500

privacy:
  allow_cloud: false
  mask_pii_default: true
  redaction_default: true

indices:
  fts:
    enabled: true
    rebuild_interval: 86400
  vector:
    enabled: true
    model: all-MiniLM-L6-v2
  analytics:
    enabled: true
```

### 2. Environment Variables
```bash
export SYMBIOTE_VAULT=/path/to/vault
export SYMBIOTE_PORT=8765
export SYMBIOTE_LOG_LEVEL=INFO
```

## 🎯 Deployment Steps

### Step 1: Validate Installation
```bash
cd symbiote
./validation/quick_check.sh

# Expected output:
# ✅ All checks passed! Implementation is complete.
```

### Step 2: Initialize Vault
```bash
# Create vault structure
mkdir -p vault/{daily,notes,tasks,projects,synthesis}

# Generate test data (optional)
python scripts/gen_vault.py --size 1000 --path vault
```

### Step 3: Start Daemon
```bash
# Start in foreground (for testing)
python -m symbiote.daemon.main

# Or start in background
make daemon

# Check status
make status
```

### Step 4: Test Core Functions
```bash
# Test capture
curl -X POST http://localhost:8765/capture \
  -H "Content-Type: application/json" \
  -d '{"text": "Test capture"}'

# Test search
curl "http://localhost:8765/context?q=test"

# Check health
curl http://localhost:8765/health
```

### Step 5: Run Validation Suite
```bash
# Quick validation
make validate-quick

# Full validation (requires daemon running)
make validate

# Performance benchmarks
make benchmark
```

## 🔍 Monitoring

### Health Checks
```bash
# Check daemon health
curl http://localhost:8765/health

# Get metrics
curl http://localhost:8765/metrics?format=json

# View logs
tail -f daemon.log
```

### Performance Monitoring
```bash
# Monitor latencies
watch -n 5 'curl -s http://localhost:8765/metrics | jq .latencies'

# Check memory usage
ps aux | grep "daemon.main" | awk '{print $6/1024 " MB"}'
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Daemon Won't Start
```bash
# Check if port is in use
lsof -i :8765

# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip list | grep -E "aiohttp|pydantic|duckdb"
```

#### 2. High Latency
```bash
# Rebuild indexes
curl -X POST http://localhost:8765/admin/reindex

# Check index sizes
du -sh vault/.sym/*

# Reduce memory usage by disabling vector search
# In symbiote.yaml: indices.vector.enabled: false
```

#### 3. WAL Recovery Issues
```bash
# Check WAL integrity
ls -la vault/.sym/wal/

# Force WAL replay
python -c "from symbiote.daemon.capture import WAL; WAL('./vault').replay()"
```

## 🔐 Security Hardening

### 1. Bind to Localhost Only
```python
# In daemon/main.py
app.run(host='127.0.0.1', port=8765)  # localhost only
```

### 2. Enable Authentication (Optional)
```yaml
# In symbiote.yaml
security:
  require_auth: true
  auth_token: "your-secret-token"
```

### 3. Audit Logging
```bash
# View audit logs
sqlite3 vault/.sym/analytics.db \
  "SELECT * FROM audit_outbound ORDER BY created_at DESC LIMIT 10"
```

## 📊 Performance Tuning

### Optimize for Speed
```yaml
# symbiote.yaml
performance:
  capture_batch_size: 10
  search_max_workers: 4
  index_commit_interval: 60
```

### Optimize for Memory
```yaml
performance:
  max_memory_mb: 1000
  cache_size_mb: 100
  disable_vector: true  # Save ~300MB
```

## 🚢 Production Deployment

### systemd Service
Create `/etc/systemd/system/symbiote.service`:

```ini
[Unit]
Description=Symbiote Cognitive Prosthetic
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/symbiote
ExecStart=/usr/bin/python -m symbiote.daemon.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable symbiote
sudo systemctl start symbiote
sudo systemctl status symbiote
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY symbiote/ ./symbiote/
COPY scripts/ ./scripts/

EXPOSE 8765
CMD ["python", "-m", "symbiote.daemon.main"]
```

Build and run:
```bash
docker build -t symbiote .
docker run -d -p 8765:8765 -v ./vault:/app/vault symbiote
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Run multiple daemons behind a load balancer
- Share vault via NFS or object storage
- Use Redis for distributed event bus

### Vertical Scaling
- Increase memory for larger indexes
- Use SSD for vault storage
- Enable parallel indexing

## 🔄 Backup & Recovery

### Backup Strategy
```bash
# Backup vault (source of truth)
tar -czf vault-backup-$(date +%Y%m%d).tar.gz vault/

# Backup indexes (optional, can rebuild)
tar -czf indexes-backup-$(date +%Y%m%d).tar.gz vault/.sym/

# Backup config
cp symbiote.yaml symbiote.yaml.backup
```

### Recovery Process
```bash
# Restore vault
tar -xzf vault-backup-20240101.tar.gz

# Rebuild indexes
curl -X POST http://localhost:8765/admin/reindex

# Verify
make validate
```

## 📝 Maintenance

### Daily Tasks
- Check daemon health
- Monitor disk usage
- Review error logs

### Weekly Tasks
- Run validation suite
- Check performance metrics
- Update dependencies

### Monthly Tasks
- Rebuild indexes
- Analyze usage patterns
- Optimize configuration

## 🆘 Support

### Getting Help
- Check logs: `tail -f daemon.log`
- Run doctor: `python cli/sym.py doctor`
- Validation: `make validate`

### Reporting Issues
Include:
1. System info (OS, Python version)
2. Error logs
3. Configuration (redacted)
4. Steps to reproduce

## ✅ Post-Deployment Validation

Run these commands to confirm successful deployment:

```bash
# 1. Structure check
./validation/quick_check.sh

# 2. API test
curl http://localhost:8765/health

# 3. Capture test
echo "Deployment test" | sym capture

# 4. Search test
sym search "deployment"

# 5. Full validation
make validate
```

Expected result: All tests pass with green checkmarks.

---

*Deployment Guide v1.0 - Symbiote Cognitive Prosthetic*