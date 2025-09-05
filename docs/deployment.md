# NFL Projects Deployment Guide

This comprehensive guide covers production deployment, operations, and maintenance procedures for the NFL Projects suite.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Procedures](#installation-procedures)
- [Configuration Management](#configuration-management)
- [Production Deployment](#production-deployment)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Operations Procedures](#operations-procedures)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## System Requirements

### Minimum Hardware Requirements

**Production Server:**
- **CPU**: 4 cores, 2.4GHz+ (Intel/AMD x64)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 100GB SSD, 1TB for data archive
- **Network**: 100Mbps+ internet connection

**Development Environment:**
- **CPU**: 2 cores, 2.0GHz+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 50GB available space

### Software Requirements

**Operating System:**
- **Primary**: Ubuntu 20.04+ LTS, CentOS 8+, RHEL 8+
- **Supported**: macOS 12+, Windows 10+ (development only)

**Runtime Environment:**
- **Python**: 3.12+ (required)
- **pip**: 23.0+
- **Git**: 2.30+

**Optional Services:**
- **Redis**: 6.0+ (caching, recommended)
- **PostgreSQL**: 13+ (data persistence, optional)
- **Nginx**: 1.20+ (reverse proxy, production)

## Installation Procedures

### Automated Installation

```bash
#!/bin/bash
# automated_install.sh - Production deployment script

set -e  # Exit on any error

# Configuration
APP_USER="nflprojects"
APP_DIR="/opt/nfl-projects"
PYTHON_VERSION="3.12"
VENV_DIR="$APP_DIR/venv"

# Create application user
sudo useradd -r -s /bin/false -d $APP_DIR $APP_USER 2>/dev/null || true

# Create application directory
sudo mkdir -p $APP_DIR
sudo chown $APP_USER:$APP_USER $APP_DIR

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev \
    git curl build-essential libssl-dev libffi-dev

# Clone repository
sudo -u $APP_USER git clone <repository-url> $APP_DIR/src
cd $APP_DIR

# Create virtual environment
sudo -u $APP_USER python3.12 -m venv $VENV_DIR
sudo -u $APP_USER $VENV_DIR/bin/pip install --upgrade pip

# Install dependencies
sudo -u $APP_USER $VENV_DIR/bin/pip install -r src/requirements.txt

# Install application
sudo -u $APP_USER $VENV_DIR/bin/pip install -e src/

# Create configuration directories
sudo -u $APP_USER mkdir -p $APP_DIR/{config,logs,data,cache}

# Set permissions
sudo chmod 750 $APP_DIR
sudo chmod -R 640 $APP_DIR/config
sudo chmod -R 755 $APP_DIR/{logs,data,cache}

echo "✅ NFL Projects installation completed"
```

### Manual Installation

```bash
# 1. System preparation
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv git

# 2. User and directory setup
sudo useradd -r -s /bin/false nflprojects
sudo mkdir -p /opt/nfl-projects
sudo chown nflprojects:nflprojects /opt/nfl-projects

# 3. Application deployment
cd /opt/nfl-projects
sudo -u nflprojects git clone <repository-url> src
sudo -u nflprojects python3.12 -m venv venv
sudo -u nflprojects venv/bin/pip install -r src/requirements.txt

# 4. Configuration setup
sudo -u nflprojects mkdir -p {config,logs,data,cache}
sudo -u nflprojects cp src/config/production.yaml config/config.yaml
```

## Configuration Management

### Production Configuration Template

Create `/opt/nfl-projects/config/config.yaml`:

```yaml
# NFL Projects Production Configuration

application:
  name: "nfl-projects"
  version: "1.0.0"
  environment: "production"
  debug: false

# API Configuration
espn_api:
  base_url: "https://sports.espn.com/nfl"
  rate_limit: 1.0  # requests per second
  timeout: 30      # seconds
  retries: 3
  cache_enabled: true
  cache_ttl: 3600  # seconds

# Database Configuration (optional)
database:
  enabled: false
  url: "postgresql://user:pass@localhost:5432/nflprojects"
  pool_size: 10
  max_overflow: 20

# Caching Configuration
cache:
  enabled: true
  backend: "redis"  # redis, memory, file
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
  max_memory: "512MB"

# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  file: "/opt/nfl-projects/logs/application.log"
  max_size: "100MB"
  backup_count: 10
  rotation: "daily"

# Performance Configuration (Phase 3 Integration)
performance:
  memory_monitoring: true
  memory_limit_mb: 1024
  streaming_chunk_size: 1000
  enable_gc_optimization: true
  profiling_enabled: false

# Power Rankings Configuration
power_rankings:
  home_field_advantage: 2.5
  sos_weight: 0.20
  margin_weight: 0.65
  trend_weight: 0.10
  playoff_weight: 0.05
  max_iterations: 10
  convergence_threshold: 0.01

# Spread Model Configuration
spread_model:
  model_type: "billy_walters"
  base_home_advantage: 2.5
  rating_scale_factor: 11.0
  confidence_threshold: 0.52

# Export Configuration
export:
  output_directory: "/opt/nfl-projects/data/output"
  formats: ["csv", "json"]
  include_metadata: true
  compress_files: false

# Security Configuration
security:
  api_key_required: false
  rate_limiting: true
  cors_enabled: false
  allowed_origins: []

# Monitoring Configuration
monitoring:
  enabled: true
  health_check_endpoint: "/health"
  metrics_enabled: true
  alerts:
    memory_threshold: 80  # percentage
    api_failure_threshold: 5  # consecutive failures
```

### Environment-Specific Configuration

**Development (`config/development.yaml`):**
```yaml
application:
  debug: true
  environment: "development"

logging:
  level: "DEBUG"
  format: "colored"

performance:
  profiling_enabled: true
  memory_monitoring: true

espn_api:
  rate_limit: 0.5  # Slower for development
```

**Testing (`config/testing.yaml`):**
```yaml
application:
  environment: "testing"

espn_api:
  cache_enabled: false  # Fresh data for tests

database:
  url: "sqlite:///test.db"  # In-memory for tests

logging:
  level: "WARNING"
```

## Production Deployment

### Systemd Service Configuration

Create `/etc/systemd/system/nfl-projects.service`:

```ini
[Unit]
Description=NFL Projects Power Rankings and Spread Model Service
After=network.target
Requires=network.target

[Service]
Type=exec
User=nflprojects
Group=nflprojects
WorkingDirectory=/opt/nfl-projects
Environment=PYTHONPATH=/opt/nfl-projects/src
ExecStart=/opt/nfl-projects/venv/bin/python -m power_ranking.cli --daemon
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
KillMode=process
TimeoutStopSec=30

# Security settings
NoNewPrivileges=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/opt/nfl-projects/logs /opt/nfl-projects/data /opt/nfl-projects/cache

# Resource limits (Phase 3 Integration)
MemoryLimit=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

### Service Management

```bash
# Enable and start service
sudo systemctl enable nfl-projects
sudo systemctl start nfl-projects

# Check status
sudo systemctl status nfl-projects

# View logs
sudo journalctl -u nfl-projects -f

# Restart service
sudo systemctl restart nfl-projects
```

### Nginx Reverse Proxy (Optional)

Create `/etc/nginx/sites-available/nfl-projects`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Application proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/nfl-projects/static/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }
}
```

### SSL/TLS Configuration (Production)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Alerting

### Health Check Endpoint

The application provides built-in health monitoring:

```python
# health_check.py - Integrated health monitoring
from typing import Dict, Any
import time
import psutil
from power_ranking.config_manager import ConfigManager
from power_ranking.memory.memory_monitor import MemoryMonitor

class HealthChecker:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.start_time = time.time()
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Returns detailed system status including:
        - Application uptime
        - Memory usage (Phase 3 integration)
        - API connectivity
        - Configuration validity
        """
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": {}
        }
        
        # Memory check (Phase 3)
        memory_stats = self.memory_monitor.get_memory_stats()
        memory_usage_pct = (memory_stats['current_memory']['rss_mb'] / 
                           (psutil.virtual_memory().total / 1024 / 1024)) * 100
        
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory_usage_pct < 80 else "warning",
            "usage_percentage": memory_usage_pct,
            "current_mb": memory_stats['current_memory']['rss_mb']
        }
        
        # API connectivity check
        try:
            from power_ranking.api.espn_client import ESPNClient
            client = ESPNClient()
            # Quick connectivity test
            response = client.test_connection()
            health_status["checks"]["espn_api"] = {
                "status": "healthy" if response else "error",
                "response_time_ms": getattr(response, 'response_time', 0)
            }
        except Exception as e:
            health_status["checks"]["espn_api"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall status
        check_statuses = [check["status"] for check in health_status["checks"].values()]
        if "error" in check_statuses:
            health_status["status"] = "unhealthy"
        elif "warning" in check_statuses:
            health_status["status"] = "degraded"
        
        return health_status
```

### Prometheus Monitoring (Optional)

Create `/opt/nfl-projects/config/prometheus.yml`:

```yaml
# Prometheus monitoring configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "nfl_projects_rules.yml"

scrape_configs:
  - job_name: 'nfl-projects'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 30s
    metrics_path: '/metrics'
    
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Metrics (Phase 3 Integration)

```python
# metrics.py - Custom application metrics
from prometheus_client import Counter, Histogram, Gauge
import time

# Application metrics
api_requests_total = Counter('nfl_api_requests_total', 'Total API requests', ['method', 'endpoint'])
ranking_calculation_duration = Histogram('ranking_calculation_seconds', 'Ranking calculation time')
memory_usage_mb = Gauge('memory_usage_megabytes', 'Current memory usage in MB')
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate percentage')

class MetricsCollector:
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
    
    def record_api_request(self, method: str, endpoint: str):
        api_requests_total.labels(method=method, endpoint=endpoint).inc()
    
    @ranking_calculation_duration.time()
    def calculate_rankings_with_metrics(self, games):
        # Existing ranking calculation logic
        return self.power_model.calculate_power_rankings(games)
    
    def update_system_metrics(self):
        # Memory usage (Phase 3)
        memory_stats = self.memory_monitor.get_current_memory()
        memory_usage_mb.set(memory_stats.rss_mb)
```

### Log Monitoring

**Structured Logging Configuration:**
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
            'extra': getattr(record, 'extra', {})
        }
        return json.dumps(log_entry)

# Configure logging
logger = logging.getLogger('nfl_projects')
handler = logging.FileHandler('/opt/nfl-projects/logs/application.log')
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

## Operations Procedures

### Daily Operations

#### Data Refresh Procedure

```bash
#!/bin/bash
# daily_data_refresh.sh - Automated daily data update

LOG_FILE="/opt/nfl-projects/logs/daily_refresh.log"
DATA_DIR="/opt/nfl-projects/data"
CONFIG_FILE="/opt/nfl-projects/config/config.yaml"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

log "Starting daily data refresh"

# Check system health before processing
HEALTH_CHECK=$(/opt/nfl-projects/venv/bin/python -c "
from power_ranking.health_check import HealthChecker
from power_ranking.config_manager import ConfigManager
config = ConfigManager('$CONFIG_FILE')
health = HealthChecker(config).check_system_health()
print(health['status'])
")

if [ "$HEALTH_CHECK" != "healthy" ]; then
    log "ERROR: System health check failed: $HEALTH_CHECK"
    exit 1
fi

# Update power rankings for current week
CURRENT_WEEK=$(date +%U)
/opt/nfl-projects/venv/bin/python -m power_ranking.cli \
    --config $CONFIG_FILE \
    --week $CURRENT_WEEK \
    --output-dir $DATA_DIR/current \
    --format csv,json 2>> $LOG_FILE

if [ $? -eq 0 ]; then
    log "Power rankings update completed successfully"
    
    # Update spread predictions
    /opt/nfl-projects/venv/bin/python -m nfl_model.cli \
        --config $CONFIG_FILE \
        --week $CURRENT_WEEK \
        --power-rankings $DATA_DIR/current/power_rankings.csv \
        --output-dir $DATA_DIR/current 2>> $LOG_FILE
    
    if [ $? -eq 0 ]; then
        log "Spread predictions update completed successfully"
    else
        log "ERROR: Spread predictions update failed"
        exit 1
    fi
else
    log "ERROR: Power rankings update failed"
    exit 1
fi

# Archive old data (keep last 30 days)
find $DATA_DIR/archive -type f -mtime +30 -delete

log "Daily data refresh completed successfully"
```

#### Weekly System Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh - Weekly system maintenance

# Log cleanup (keep last 30 days)
find /opt/nfl-projects/logs -name "*.log" -mtime +30 -delete

# Cache cleanup
redis-cli FLUSHDB

# System updates (if enabled)
if [ "$AUTO_UPDATE" = "true" ]; then
    cd /opt/nfl-projects/src
    git pull origin main
    /opt/nfl-projects/venv/bin/pip install -r requirements.txt --upgrade
    sudo systemctl restart nfl-projects
fi

# Backup configuration
cp -r /opt/nfl-projects/config /opt/nfl-projects/backups/config-$(date +%Y%m%d)

# Performance report generation
/opt/nfl-projects/venv/bin/python -c "
from power_ranking.memory.memory_profiler import AdvancedMemoryProfiler
profiler = AdvancedMemoryProfiler()
report = profiler.generate_weekly_performance_report()
print(report)
" > /opt/nfl-projects/logs/weekly_performance_$(date +%Y%m%d).log
```

### Backup Procedures

#### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh - Configuration backup script

BACKUP_DIR="/opt/nfl-projects/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup configuration files
cp -r /opt/nfl-projects/config $BACKUP_DIR/$DATE/
cp /etc/systemd/system/nfl-projects.service $BACKUP_DIR/$DATE/
cp /etc/nginx/sites-available/nfl-projects $BACKUP_DIR/$DATE/

# Create archive
cd $BACKUP_DIR
tar -czf config_backup_$DATE.tar.gz $DATE/
rm -rf $DATE/

# Retain only last 10 backups
ls -t config_backup_*.tar.gz | tail -n +11 | xargs rm -f

echo "Configuration backup completed: config_backup_$DATE.tar.gz"
```

#### Data Backup

```bash
#!/bin/bash
# backup_data.sh - Data backup script

DATA_DIR="/opt/nfl-projects/data"
BACKUP_DIR="/opt/nfl-projects/backups"
DATE=$(date +%Y%m%d)

# Backup current data
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz -C $DATA_DIR .

# Upload to remote storage (configure as needed)
# aws s3 cp $BACKUP_DIR/data_backup_$DATE.tar.gz s3://your-backup-bucket/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "data_backup_*.tar.gz" -mtime +30 -delete
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **High Memory Usage**

**Symptoms:**
- Memory usage > 80%
- Application slowdowns
- Out of memory errors

**Diagnosis:**
```bash
# Check memory usage
sudo systemctl status nfl-projects
free -h
ps aux | grep python

# Application-level memory analysis
/opt/nfl-projects/venv/bin/python -c "
from power_ranking.memory.memory_monitor import MemoryMonitor
monitor = MemoryMonitor()
stats = monitor.get_memory_stats()
print('Current memory:', stats['current_memory'])
print('Peak memory:', stats.get('peak_memory', 'N/A'))
"
```

**Solutions:**
```bash
# 1. Restart service to clear memory leaks
sudo systemctl restart nfl-projects

# 2. Adjust memory limits in config
# Edit /opt/nfl-projects/config/config.yaml:
# performance:
#   memory_limit_mb: 512  # Reduce from 1024
#   streaming_chunk_size: 500  # Reduce chunk size

# 3. Enable aggressive garbage collection
# performance:
#   enable_gc_optimization: true
```

#### 2. **ESPN API Connectivity Issues**

**Symptoms:**
- API timeout errors
- Rate limiting responses
- Connection refused errors

**Diagnosis:**
```bash
# Test ESPN API connectivity
curl -I "https://sports.espn.com/nfl/scoreboard"

# Check application logs
tail -f /opt/nfl-projects/logs/application.log | grep -i "espn\|api"

# Test from application
/opt/nfl-projects/venv/bin/python -c "
from power_ranking.api.espn_client import ESPNClient
client = ESPNClient()
try:
    result = client.test_connection()
    print('API test successful:', result)
except Exception as e:
    print('API test failed:', e)
"
```

**Solutions:**
```bash
# 1. Adjust rate limiting
# Edit config.yaml:
# espn_api:
#   rate_limit: 0.5  # Reduce to 0.5 requests/second
#   retries: 5       # Increase retry attempts

# 2. Enable caching to reduce API calls
# espn_api:
#   cache_enabled: true
#   cache_ttl: 7200  # Cache for 2 hours
```

#### 3. **Performance Degradation**

**Symptoms:**
- Slow ranking calculations
- High CPU usage
- Timeout errors

**Diagnosis:**
```bash
# CPU usage analysis
top -p $(pgrep -f nfl-projects)

# Application performance profiling
/opt/nfl-projects/venv/bin/python -c "
from power_ranking.memory.memory_profiler import AdvancedMemoryProfiler
profiler = AdvancedMemoryProfiler()
# Profile recent performance
report = profiler.get_performance_summary()
for recommendation in report.get('recommendations', []):
    print('RECOMMENDATION:', recommendation)
"
```

**Solutions:**
```bash
# 1. Optimize configuration
# Edit config.yaml:
# power_rankings:
#   max_iterations: 5  # Reduce from 10
# performance:
#   streaming_chunk_size: 2000  # Increase for better throughput

# 2. Enable performance monitoring
# monitoring:
#   enabled: true
#   profiling_enabled: true
```

### Log Analysis

#### Error Pattern Detection

```bash
# Common error patterns
ERROR_PATTERNS=(
    "OutOfMemoryError"
    "ConnectionError.*ESPN"
    "ValidationError"
    "TimeoutError"
    "PermissionError"
)

# Search for patterns in logs
for pattern in "${ERROR_PATTERNS[@]}"; do
    echo "=== Checking for: $pattern ==="
    grep -i "$pattern" /opt/nfl-projects/logs/*.log | tail -5
done
```

#### Performance Analysis

```bash
# Analyze response times
grep "response_time" /opt/nfl-projects/logs/application.log | \
    awk '{print $NF}' | sort -n | \
    awk '
    BEGIN{sum=0; count=0}
    {sum+=$1; count++; values[count]=$1}
    END{
        print "Count:", count
        print "Average:", sum/count
        print "Median:", values[int(count/2)]
        print "95th percentile:", values[int(count*0.95)]
    }'
```

## Maintenance

### Regular Maintenance Tasks

#### Monthly Tasks
- [ ] Review and rotate logs
- [ ] Update system dependencies
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] Backup verification

#### Quarterly Tasks
- [ ] Full system health assessment
- [ ] Configuration review and optimization
- [ ] Capacity planning analysis
- [ ] Documentation updates
- [ ] Disaster recovery testing

#### Annual Tasks
- [ ] Security penetration testing
- [ ] Performance baseline establishment
- [ ] Infrastructure upgrade planning
- [ ] Compliance audit

### Update Procedures

#### Application Updates

```bash
#!/bin/bash
# update_application.sh - Safe application update procedure

# 1. Backup current installation
cp -r /opt/nfl-projects /opt/nfl-projects-backup-$(date +%Y%m%d)

# 2. Stop service
sudo systemctl stop nfl-projects

# 3. Update code
cd /opt/nfl-projects/src
sudo -u nflprojects git fetch origin
sudo -u nflprojects git checkout main
sudo -u nflprojects git pull origin main

# 4. Update dependencies
sudo -u nflprojects /opt/nfl-projects/venv/bin/pip install -r requirements.txt --upgrade

# 5. Run tests
/opt/nfl-projects/venv/bin/python -m pytest power_ranking/tests/ nfl_model/tests/

# 6. Start service
sudo systemctl start nfl-projects

# 7. Verify service health
sleep 30
curl -f http://localhost:8000/health || {
    echo "Health check failed - rolling back"
    sudo systemctl stop nfl-projects
    rm -rf /opt/nfl-projects
    mv /opt/nfl-projects-backup-$(date +%Y%m%d) /opt/nfl-projects
    sudo systemctl start nfl-projects
    exit 1
}

echo "Application update completed successfully"
```

### Disaster Recovery

#### Recovery Procedures

**Service Failure Recovery:**
```bash
# 1. Check system resources
df -h
free -h
systemctl status nfl-projects

# 2. Restore from backup
tar -xzf /opt/nfl-projects/backups/config_backup_latest.tar.gz -C /opt/nfl-projects/

# 3. Restart services
sudo systemctl restart nfl-projects
sudo systemctl restart nginx

# 4. Verify recovery
curl -f http://localhost/health
```

**Data Corruption Recovery:**
```bash
# 1. Stop application
sudo systemctl stop nfl-projects

# 2. Restore data from backup
tar -xzf /opt/nfl-projects/backups/data_backup_latest.tar.gz -C /opt/nfl-projects/data/

# 3. Verify data integrity
/opt/nfl-projects/venv/bin/python -c "
from power_ranking.validation.data_quality import DataQualityValidator
validator = DataQualityValidator()
# Run validation checks
result = validator.validate_all_data()
print('Data validation:', result.is_valid)
"

# 4. Restart application
sudo systemctl start nfl-projects
```

---

This deployment guide provides comprehensive procedures for production deployment and ongoing operations of the NFL Projects suite. All procedures are designed with reliability, security, and maintainability in mind.

**Regular Updates**: This guide should be reviewed and updated with each major release of the NFL Projects software.

**Production Readiness**: Following these procedures ensures enterprise-grade deployment suitable for production environments.