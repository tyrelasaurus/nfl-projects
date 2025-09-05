"""
NFL Projects Unified Monitoring & Observability Framework

This module provides centralized monitoring, health checks, and alerting capabilities
for the NFL Projects suite, consolidating existing monitoring components from:

- Phase 1.3: Data Quality Monitoring (power_ranking.validation.data_monitoring)
- Phase 3: Memory Monitoring (power_ranking.memory.memory_monitor) 
- Validation Systems: Performance Metrics (nfl_model.validation.performance_metrics)

Key Components:
- health_checks: System health monitoring and endpoints
- performance_metrics: Centralized performance monitoring 
- alerts: Unified alerting framework
- dashboard: Web-based monitoring interface

Example:
    >>> from monitoring import HealthChecker, PerformanceMonitor, AlertManager
    >>> health = HealthChecker()
    >>> status = health.check_system_health()
    >>> print(f"System status: {status['overall_status']}")
"""

__version__ = "1.0.0"
__author__ = "NFL Projects Team"

# Import main monitoring components
from .health_checks import HealthChecker, SystemHealthStatus
from .performance_metrics import PerformanceMonitor, MetricsCollector
from .alerts import AlertManager, AlertLevel, Alert
from .dashboard import MonitoringDashboard

__all__ = [
    'HealthChecker',
    'SystemHealthStatus', 
    'PerformanceMonitor',
    'MetricsCollector',
    'AlertManager',
    'AlertLevel',
    'Alert',
    'MonitoringDashboard'
]