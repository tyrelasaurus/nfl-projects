"""
Unified Health Check System for NFL Projects

Consolidates health monitoring capabilities across both Power Rankings and NFL Spread Model
systems, providing comprehensive system status monitoring for production environments.

This module integrates existing health check functionality from:
- Deployment guide health check implementations
- Phase 3 memory monitoring health status
- Phase 1.3 data quality health indicators
"""

import time
import psutil
import logging
import traceback
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import sys

# Add project paths for existing component imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from power_ranking.memory.memory_monitor import MemoryMonitor
    from power_ranking.config_manager import ConfigManager as PowerConfigManager
except ImportError:
    # Fallback for testing environments
    MemoryMonitor = None
    PowerConfigManager = None

try:
    from nfl_model.config_manager import get_nfl_config
except ImportError:
    get_nfl_config = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components that can be monitored."""
    MEMORY = "memory"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    DISK = "disk"
    NETWORK = "network"
    APPLICATION = "application"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'response_time_ms': self.response_time_ms,
            'metadata': self.metadata
        }


@dataclass 
class SystemHealthStatus:
    """Overall system health status with component details."""
    overall_status: HealthStatus
    components: List[HealthCheckResult] = field(default_factory=list)
    uptime_seconds: float = 0
    system_load: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'overall_status': self.overall_status.value,
            'components': [comp.to_dict() for comp in self.components],
            'uptime_seconds': self.uptime_seconds,
            'system_load': self.system_load,
            'timestamp': self.timestamp.isoformat(),
            'summary': {
                'healthy_components': len([c for c in self.components if c.status == HealthStatus.HEALTHY]),
                'degraded_components': len([c for c in self.components if c.status == HealthStatus.DEGRADED]),
                'unhealthy_components': len([c for c in self.components if c.status == HealthStatus.UNHEALTHY]),
                'total_components': len(self.components)
            }
        }


class HealthChecker:
    """
    Unified health checking system for NFL Projects.
    
    Provides comprehensive system health monitoring by consolidating existing
    monitoring capabilities from Phases 1.3 and 3, plus additional system checks.
    
    Features:
    - Memory usage monitoring (Phase 3 integration)
    - API connectivity validation
    - System resource monitoring
    - Application component health
    - Configurable health thresholds
    
    Example:
        >>> checker = HealthChecker()
        >>> health = checker.check_system_health()
        >>> print(f"Status: {health.overall_status.value}")
        >>> for component in health.components:
        ...     print(f"{component.component}: {component.status.value}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health checker with configuration.
        
        Args:
            config: Optional configuration dict. If None, loads default config.
                   Expected keys:
                   - memory_threshold_percent: Memory usage warning threshold (default: 80)
                   - disk_threshold_percent: Disk usage warning threshold (default: 85)
                   - api_timeout_seconds: API health check timeout (default: 10)
                   - load_threshold: System load warning threshold (default: 2.0)
        """
        self.config = config or self._load_default_config()
        self.start_time = time.time()
        
        # Initialize monitoring components if available
        self.memory_monitor = MemoryMonitor() if MemoryMonitor else None
        self.power_config = PowerConfigManager() if PowerConfigManager else None
        
        # Health check registry
        self._health_checks = {
            ComponentType.MEMORY: self._check_memory_health,
            ComponentType.DISK: self._check_disk_health,
            ComponentType.API: self._check_api_health,
            ComponentType.APPLICATION: self._check_application_health,
            ComponentType.NETWORK: self._check_network_health
        }
        
        logger.info("HealthChecker initialized with monitoring capabilities")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default health check configuration."""
        return {
            'memory_threshold_percent': 80.0,
            'disk_threshold_percent': 85.0,
            'api_timeout_seconds': 10.0,
            'load_threshold': 2.0,
            'enable_detailed_checks': True,
            'max_response_time_ms': 5000.0
        }
    
    def check_system_health(self, components: Optional[List[ComponentType]] = None) -> SystemHealthStatus:
        """
        Perform comprehensive system health check.
        
        Args:
            components: List of specific components to check. If None, checks all components.
            
        Returns:
            SystemHealthStatus with overall status and individual component results.
            
        The overall status is determined by the worst individual component status:
        - HEALTHY: All components healthy
        - DEGRADED: At least one component degraded, none unhealthy  
        - UNHEALTHY: At least one component unhealthy
        """
        start_time = time.time()
        check_components = components or list(self._health_checks.keys())
        
        results = []
        worst_status = HealthStatus.HEALTHY
        
        logger.info(f"Starting system health check for {len(check_components)} components")
        
        # Run individual health checks
        for component_type in check_components:
            try:
                if component_type in self._health_checks:
                    result = self._health_checks[component_type]()
                    results.append(result)
                    
                    # Track worst status
                    if result.status == HealthStatus.UNHEALTHY:
                        worst_status = HealthStatus.UNHEALTHY
                    elif result.status == HealthStatus.DEGRADED and worst_status != HealthStatus.UNHEALTHY:
                        worst_status = HealthStatus.DEGRADED
                        
                else:
                    logger.warning(f"No health check registered for component: {component_type}")
                    
            except Exception as e:
                logger.error(f"Health check failed for {component_type}: {e}")
                results.append(HealthCheckResult(
                    component=component_type.value,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(e)}",
                    metadata={'error': str(e), 'traceback': traceback.format_exc()}
                ))
                worst_status = HealthStatus.UNHEALTHY
        
        # Collect system metrics
        system_load = self._get_system_load()
        uptime = time.time() - self.start_time
        
        health_status = SystemHealthStatus(
            overall_status=worst_status,
            components=results,
            uptime_seconds=uptime,
            system_load=system_load
        )
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"System health check completed in {total_time:.2f}ms - Status: {worst_status.value}")
        
        return health_status
    
    def _check_memory_health(self) -> HealthCheckResult:
        """Check memory usage and health."""
        start_time = time.time()
        
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Use Phase 3 memory monitor if available
            phase3_info = {}
            if self.memory_monitor:
                try:
                    memory_stats = self.memory_monitor.get_memory_stats()
                    phase3_info = {
                        'current_rss_mb': memory_stats.get('current_memory', {}).get('rss_mb', 0),
                        'peak_memory_mb': memory_stats.get('peak_memory', {}).get('rss_mb', 0),
                        'profiles_collected': len(memory_stats.get('profiles', []))
                    }
                except Exception as e:
                    logger.warning(f"Phase 3 memory monitor error: {e}")
            
            # Determine status based on memory usage
            if memory_percent >= self.config['memory_threshold_percent']:
                status = HealthStatus.DEGRADED if memory_percent < 90 else HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="memory",
                status=status,
                message=message,
                response_time_ms=response_time,
                metadata={
                    'memory_percent': memory_percent,
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    **phase3_info
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _check_disk_health(self) -> HealthCheckResult:
        """Check disk usage and health."""
        start_time = time.time()
        
        try:
            # Check root disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Determine status
            if disk_percent >= self.config['disk_threshold_percent']:
                status = HealthStatus.DEGRADED if disk_percent < 95 else HealthStatus.UNHEALTHY
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="disk",
                status=status,
                message=message,
                response_time_ms=response_time,
                metadata={
                    'disk_percent': disk_percent,
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'used_gb': round(disk_usage.used / (1024**3), 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _check_api_health(self) -> HealthCheckResult:
        """Check API connectivity and health."""
        start_time = time.time()
        
        try:
            # Test ESPN API connectivity (primary data source)
            import requests
            
            test_url = "https://sports.espn.com/nfl/scoreboard"
            response = requests.head(
                test_url, 
                timeout=self.config['api_timeout_seconds'],
                headers={'User-Agent': 'NFL-Projects-Health-Check/1.0'}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = f"API connectivity healthy (HTTP {response.status_code})"
            elif response.status_code < 500:
                status = HealthStatus.DEGRADED
                message = f"API connectivity degraded (HTTP {response.status_code})"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"API connectivity unhealthy (HTTP {response.status_code})"
            
            return HealthCheckResult(
                component="api",
                status=status,
                message=message,
                response_time_ms=response_time,
                metadata={
                    'status_code': response.status_code,
                    'test_url': test_url,
                    'response_headers': dict(response.headers)
                }
            )
            
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API timeout after {self.config['api_timeout_seconds']}s",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': 'timeout'}
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _check_application_health(self) -> HealthCheckResult:
        """Check application-specific health indicators."""
        start_time = time.time()
        
        try:
            health_indicators = []
            issues = []
            
            # Check if core modules can be imported
            try:
                import power_ranking
                health_indicators.append("power_ranking module available")
            except ImportError as e:
                issues.append(f"power_ranking import failed: {e}")
            
            try:
                import nfl_model
                health_indicators.append("nfl_model module available")
            except ImportError as e:
                issues.append(f"nfl_model import failed: {e}")
            
            # Check configuration accessibility
            config_status = []
            if self.power_config:
                config_status.append("power_ranking config loaded")
            else:
                issues.append("power_ranking config unavailable")
            
            if get_nfl_config:
                try:
                    get_nfl_config()
                    config_status.append("nfl_model config loaded")
                except Exception as e:
                    issues.append(f"nfl_model config error: {e}")
            
            # Determine overall application health
            if not issues:
                status = HealthStatus.HEALTHY
                message = "All application components healthy"
            elif len(issues) <= 1:
                status = HealthStatus.DEGRADED
                message = f"Application partially healthy: {len(issues)} issue(s)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Application unhealthy: {len(issues)} critical issues"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="application",
                status=status,
                message=message,
                response_time_ms=response_time,
                metadata={
                    'health_indicators': health_indicators,
                    'issues': issues,
                    'config_status': config_status,
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="application",
                status=HealthStatus.UNHEALTHY,
                message=f"Application check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _check_network_health(self) -> HealthCheckResult:
        """Check network connectivity and health."""
        start_time = time.time()
        
        try:
            # Basic connectivity test
            import socket
            
            # Test DNS resolution
            try:
                socket.gethostbyname('google.com')
                dns_ok = True
            except socket.gaierror:
                dns_ok = False
            
            # Test internet connectivity
            try:
                socket.create_connection(('8.8.8.8', 53), timeout=5)
                internet_ok = True
            except (socket.timeout, socket.error):
                internet_ok = False
            
            # Determine status
            if dns_ok and internet_ok:
                status = HealthStatus.HEALTHY
                message = "Network connectivity healthy"
            elif dns_ok or internet_ok:
                status = HealthStatus.DEGRADED
                message = "Network connectivity partially available"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Network connectivity unavailable"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="network",
                status=status,
                message=message,
                response_time_ms=response_time,
                metadata={
                    'dns_resolution': dns_ok,
                    'internet_connectivity': internet_ok,
                    'hostname': socket.gethostname()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="network",
                status=HealthStatus.UNHEALTHY,
                message=f"Network check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                'load_1min': load_avg[0],
                'load_5min': load_avg[1], 
                'load_15min': load_avg[2],
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count()
            }
        except Exception as e:
            logger.error(f"Failed to get system load: {e}")
            return {}
    
    def health_check_endpoint(self) -> Dict[str, Any]:
        """
        HTTP health check endpoint compatible response.
        
        Returns a dictionary suitable for JSON serialization in web endpoints.
        This method is designed for integration with Flask, FastAPI, or similar
        web frameworks for /health endpoint implementation.
        
        Returns:
            Dict with health status and component details in HTTP-friendly format.
        """
        health_status = self.check_system_health()
        
        # Convert to HTTP-friendly response
        response = health_status.to_dict()
        
        # Add HTTP status code suggestion
        if health_status.overall_status == HealthStatus.HEALTHY:
            response['http_status_code'] = 200
        elif health_status.overall_status == HealthStatus.DEGRADED:
            response['http_status_code'] = 200  # Still operational
        else:
            response['http_status_code'] = 503  # Service unavailable
        
        return response
    
    def quick_health_check(self) -> bool:
        """
        Quick binary health check for simple monitoring systems.
        
        Returns:
            True if system is healthy or degraded, False if unhealthy.
        """
        try:
            health = self.check_system_health([ComponentType.MEMORY, ComponentType.APPLICATION])
            return health.overall_status != HealthStatus.UNHEALTHY
        except Exception:
            return False


# Convenience function for quick health checks
def get_system_health() -> SystemHealthStatus:
    """Get current system health status using default configuration."""
    checker = HealthChecker()
    return checker.check_system_health()


# Flask/FastAPI integration helper
def create_health_endpoint():
    """
    Create a health check endpoint function for web frameworks.
    
    Example usage with Flask:
        from flask import Flask, jsonify
        from monitoring.health_checks import create_health_endpoint
        
        app = Flask(__name__)
        health_check = create_health_endpoint()
        
        @app.route('/health')
        def health():
            response = health_check()
            return jsonify(response), response['http_status_code']
    """
    checker = HealthChecker()
    
    def health_endpoint():
        return checker.health_check_endpoint()
    
    return health_endpoint