# ✅ Symbiote Production Deployment Checklist

## Pre-Deployment Verification

### 🔍 Code Quality
- [ ] All tests passing (`make test`)
- [ ] Validation suite passes (`make validate`)
- [ ] No critical security warnings
- [ ] Code review completed
- [ ] Documentation up to date

### 📊 Performance Validation
- [ ] Capture latency < 200ms p99
- [ ] Search latency < 100ms p50
- [ ] Memory usage < 1.5GB steady state
- [ ] Graceful degradation tested
- [ ] Load testing completed

### 🔒 Security & Privacy
- [ ] Privacy gates working
- [ ] PII redaction functional
- [ ] Consent flow tested
- [ ] Audit logging enabled
- [ ] No hardcoded secrets
- [ ] API bound to localhost only

## System Requirements

### 🖥️ Hardware
- [ ] CPU: 2+ cores recommended
- [ ] RAM: 4GB minimum (2GB for Symbiote)
- [ ] Disk: 1GB free space minimum
- [ ] SSD recommended for vault storage

### 🐍 Software
- [ ] Python 3.9+ installed
- [ ] pip/poetry available
- [ ] systemd (for service management)
- [ ] Git (for version control)

## Deployment Steps

### 1. Environment Setup
```bash
# Create deployment directory
sudo mkdir -p /opt/symbiote
sudo chown $USER:$USER /opt/symbiote

# Clone repository
git clone <repository> /opt/symbiote
cd /opt/symbiote
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "from symbiote.daemon import main; print('✓ Import successful')"
```

### 3. Configuration
```bash
# Copy configuration template
cp symbiote.yaml.example symbiote.yaml

# Edit configuration
vim symbiote.yaml
```

**Required configuration:**
- [ ] Set vault_path
- [ ] Configure privacy settings
- [ ] Set performance limits
- [ ] Enable/disable indexers
- [ ] Configure logging level

### 4. Vault Setup
```bash
# Create vault structure
mkdir -p /var/lib/symbiote/vault/{daily,notes,tasks,projects,synthesis}

# Set permissions
chmod 750 /var/lib/symbiote
chmod -R 640 /var/lib/symbiote/vault
```

### 5. Service Installation
```bash
# Create systemd service
sudo tee /etc/systemd/system/symbiote.service << EOF
[Unit]
Description=Symbiote Cognitive Prosthetic
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/symbiote
Environment="PATH=/opt/symbiote/venv/bin"
ExecStart=/opt/symbiote/venv/bin/python -m symbiote.daemon.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/symbiote

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload
```

### 6. Start Service
```bash
# Enable auto-start
sudo systemctl enable symbiote

# Start service
sudo systemctl start symbiote

# Check status
sudo systemctl status symbiote
```

## Post-Deployment Validation

### 🧪 Functional Tests
- [ ] Capture endpoint working
- [ ] Search returns results
- [ ] Suggestions generated
- [ ] Receipts accessible
- [ ] Privacy gates functional

```bash
# Test capture
curl -X POST http://localhost:8765/capture \
  -d '{"text": "Production test"}'

# Test search
curl "http://localhost:8765/context?q=test"

# Check health
curl http://localhost:8765/health
```

### 📈 Performance Monitoring
- [ ] Check initial latencies
- [ ] Monitor memory usage
- [ ] Verify index sizes
- [ ] Check log output

```bash
# View metrics
curl http://localhost:8765/metrics?format=json

# Monitor logs
sudo journalctl -u symbiote -f

# Check memory
ps aux | grep symbiote
```

### 🔐 Security Verification
- [ ] API only accessible locally
- [ ] No sensitive data in logs
- [ ] Vault permissions correct
- [ ] Service running as non-root

```bash
# Check listening ports
sudo netstat -tlnp | grep 8765

# Verify permissions
ls -la /var/lib/symbiote/

# Check service user
ps aux | grep symbiote
```

## Maintenance Tasks

### Daily
- [ ] Check service health
- [ ] Monitor disk usage
- [ ] Review error logs

### Weekly
- [ ] Run validation suite
- [ ] Check performance metrics
- [ ] Review audit logs
- [ ] Update dependencies (if needed)

### Monthly
- [ ] Rebuild indexes
- [ ] Archive old logs
- [ ] Performance analysis
- [ ] Security audit

## Monitoring Setup

### Logs
```bash
# View logs
journalctl -u symbiote --since today

# Follow logs
journalctl -u symbiote -f

# Export logs
journalctl -u symbiote > symbiote.log
```

### Metrics
```bash
# Create monitoring script
cat > /opt/symbiote/monitor.sh << 'EOF'
#!/bin/bash
while true; do
  curl -s http://localhost:8765/metrics | jq '.'
  sleep 60
done
EOF
chmod +x /opt/symbiote/monitor.sh
```

### Alerts (Optional)
```bash
# Create alert script
cat > /opt/symbiote/check_health.sh << 'EOF'
#!/bin/bash
if ! curl -sf http://localhost:8765/health > /dev/null; then
  echo "Symbiote is down!" | mail -s "Alert: Symbiote Down" admin@example.com
fi
EOF

# Add to crontab
crontab -e
# */5 * * * * /opt/symbiote/check_health.sh
```

## Backup Strategy

### What to Backup
- [ ] Vault directory (primary data)
- [ ] Configuration files
- [ ] WAL files
- [ ] Indexes (optional, can rebuild)

### Backup Script
```bash
#!/bin/bash
BACKUP_DIR="/backup/symbiote"
DATE=$(date +%Y%m%d_%H%M%S)

# Stop service
sudo systemctl stop symbiote

# Backup vault
tar -czf "$BACKUP_DIR/vault_$DATE.tar.gz" /var/lib/symbiote/vault

# Backup config
cp /opt/symbiote/symbiote.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Start service
sudo systemctl start symbiote

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type f -mtime +30 -delete
```

## Rollback Plan

### If Issues Occur
1. **Stop service**: `sudo systemctl stop symbiote`
2. **Check logs**: `journalctl -u symbiote --since "1 hour ago"`
3. **Restore backup** (if needed)
4. **Revert code**: `git checkout <previous-version>`
5. **Restart service**: `sudo systemctl start symbiote`

## Performance Tuning

### For Better Latency
```yaml
# symbiote.yaml
performance:
  capture_batch_size: 5  # Smaller batches
  search_max_workers: 8  # More parallelism
  cache_size_mb: 500     # Larger cache
```

### For Lower Memory
```yaml
performance:
  max_memory_mb: 1000
  disable_vector: true  # Save ~300MB
  cache_size_mb: 100
```

## Security Hardening

### Additional Steps
- [ ] Enable firewall rules
- [ ] Set up fail2ban
- [ ] Enable SELinux/AppArmor
- [ ] Regular security updates
- [ ] Audit log monitoring

```bash
# Firewall rules (example)
sudo ufw allow from 127.0.0.1 to any port 8765
sudo ufw deny 8765

# File permissions
chmod 600 /opt/symbiote/symbiote.yaml
chmod 700 /var/lib/symbiote/vault/.sym
```

## Troubleshooting Guide

### Common Issues

| Issue | Check | Solution |
|-------|-------|----------|
| Service won't start | `systemctl status symbiote` | Check logs, verify paths |
| High latency | Metrics endpoint | Rebuild indexes, increase cache |
| Memory issues | `ps aux` | Disable vector search, reduce cache |
| No search results | Index status | Run reindex command |
| Permission denied | `ls -la` | Fix ownership/permissions |

## Sign-off

### Deployment Approval
- [ ] Technical Lead approval
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Rollback plan tested

**Deployed by**: _________________  
**Date**: _________________  
**Version**: _________________  
**Environment**: _________________

## Post-Deployment Notes

_Space for deployment notes, issues encountered, and resolutions:_

---

### Quick Commands Reference

```bash
# Service management
sudo systemctl {start|stop|restart|status} symbiote

# Logs
sudo journalctl -u symbiote -f

# Health check
curl http://localhost:8765/health

# Metrics
curl http://localhost:8765/metrics

# Reindex
curl -X POST http://localhost:8765/admin/reindex

# Backup
/opt/symbiote/backup.sh
```

---

*This checklist ensures a smooth, secure, and performant production deployment of Symbiote.*