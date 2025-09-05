"""
Unified Performance Metrics System for NFL Projects

Consolidates performance monitoring capabilities from existing systems:
- Phase 3 Memory Monitoring (power_ranking.memory.memory_monitor)
- NFL Model Performance Metrics (nfl_model.validation.performance_metrics)
- Phase 1.3 Data Quality Monitoring (power_ranking.validation.data_monitoring)

Provides centralized performance tracking, metrics collection, and analysis
for both Power Rankings and NFL Spread Model systems.
"""

import time
import psutil
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
import os
import sys

# Add project paths for existing component imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from power_ranking.memory.memory_monitor import MemoryMonitor
    from power_ranking.memory.memory_profiler import AdvancedMemoryProfiler
except ImportError:
    MemoryMonitor = None
    AdvancedMemoryProfiler = None

try:
    from nfl_model.validation.performance_metrics import (
        AccuracyMetrics, ErrorAnalysis, BettingMetrics
    )
except ImportError:
    AccuracyMetrics = None
    ErrorAnalysis = None
    BettingMetrics = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a specific time."""
    timestamp: datetime
    memory_usage_mb: float
    cpu_percent: float
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float
    active_connections: int = 0
    response_time_ms: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_percent': self.cpu_percent,
            'disk_io_mb_per_sec': self.disk_io_mb_per_sec,
            'network_io_mb_per_sec': self.network_io_mb_per_sec,
            'active_connections': self.active_connections,
            'response_time_ms': self.response_time_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec
        }


@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics."""
    power_rankings_calculated: int = 0
    spreads_calculated: int = 0
    api_requests_made: int = 0
    api_request_failures: int = 0
    data_validation_runs: int = 0
    data_validation_failures: int = 0
    export_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'power_rankings_calculated': self.power_rankings_calculated,
            'spreads_calculated': self.spreads_calculated,
            'api_requests_made': self.api_requests_made,
            'api_request_failures': self.api_request_failures,
            'api_success_rate': self._calculate_api_success_rate(),
            'data_validation_runs': self.data_validation_runs,
            'data_validation_failures': self.data_validation_failures,
            'validation_success_rate': self._calculate_validation_success_rate(),
            'export_operations': self.export_operations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_api_success_rate(self) -> float:
        """Calculate API request success rate."""
        if self.api_requests_made == 0:
            return 0.0
        return ((self.api_requests_made - self.api_request_failures) / self.api_requests_made) * 100
    
    def _calculate_validation_success_rate(self) -> float:
        """Calculate data validation success rate."""
        if self.data_validation_runs == 0:
            return 0.0
        return ((self.data_validation_runs - self.data_validation_failures) / self.data_validation_runs) * 100
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return (self.cache_hits / total_cache_operations) * 100


class MetricsCollector:
    """
    Thread-safe metrics collection system.
    
    Collects and aggregates performance metrics from various system components
    with configurable sampling intervals and retention policies.
    """
    
    def __init__(self, 
                 sampling_interval: float = 60.0,  # seconds
                 max_snapshots: int = 1440):  # 24 hours at 1-minute intervals
        """
        Initialize metrics collector.
        
        Args:
            sampling_interval: Seconds between metric snapshots
            max_snapshots: Maximum number of snapshots to retain
        """
        self.sampling_interval = sampling_interval
        self.max_snapshots = max_snapshots
        
        # Thread-safe collections
        self._snapshots = deque(maxlen=max_snapshots)
        self._app_metrics = ApplicationMetrics()
        self._lock = threading.RLock()
        
        # System monitoring state
        self._last_disk_io = self._get_disk_io_counters()
        self._last_network_io = self._get_network_io_counters()
        self._last_sample_time = time.time()
        
        # Integration with existing monitoring systems
        self.memory_monitor = MemoryMonitor() if MemoryMonitor else None
        self.memory_profiler = AdvancedMemoryProfiler() if AdvancedMemoryProfiler else None
        
        logger.info(f"MetricsCollector initialized with {sampling_interval}s sampling")
    
    def collect_snapshot(self) -> PerformanceSnapshot:
        """
        Collect a performance snapshot with current system metrics.
        
        Returns:
            PerformanceSnapshot with current system performance data.
        """
        now = time.time()
        
        # Memory usage (with Phase 3 integration)
        memory_usage_mb = self._get_memory_usage()
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Disk I/O rate
        disk_io_rate = self._calculate_disk_io_rate(now)
        
        # Network I/O rate  
        network_io_rate = self._calculate_network_io_rate(now)
        
        # Connection count
        active_connections = len(psutil.net_connections())
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            disk_io_mb_per_sec=disk_io_rate,
            network_io_mb_per_sec=network_io_rate,
            active_connections=active_connections
        )
        
        # Store snapshot thread-safely
        with self._lock:
            self._snapshots.append(snapshot)
        
        self._last_sample_time = now
        return snapshot
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            # Use Phase 3 memory monitor if available
            if self.memory_monitor:
                try:
                    stats = self.memory_monitor.get_memory_stats()
                    current_memory = stats.get('current_memory', {})
                    return current_memory.get('rss_mb', 0.0)
                except Exception as e:
                    logger.warning(f"Phase 3 memory monitor error: {e}")
            
            # Fallback to system memory
            memory = psutil.virtual_memory()
            return memory.used / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Memory usage collection failed: {e}")
            return 0.0
    
    def _get_disk_io_counters(self) -> Dict[str, int]:
        """Get current disk I/O counters."""
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return {
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes
                }
        except Exception as e:
            logger.error(f"Disk I/O counter error: {e}")
        
        return {'read_bytes': 0, 'write_bytes': 0}
    
    def _get_network_io_counters(self) -> Dict[str, int]:
        """Get current network I/O counters."""
        try:
            io_counters = psutil.net_io_counters()
            if io_counters:
                return {
                    'bytes_sent': io_counters.bytes_sent,
                    'bytes_recv': io_counters.bytes_recv
                }
        except Exception as e:
            logger.error(f"Network I/O counter error: {e}")
        
        return {'bytes_sent': 0, 'bytes_recv': 0}
    
    def _calculate_disk_io_rate(self, current_time: float) -> float:
        """Calculate disk I/O rate in MB/s."""
        try:
            current_io = self._get_disk_io_counters()
            time_delta = current_time - self._last_sample_time
            
            if time_delta <= 0:
                return 0.0
            
            read_delta = current_io['read_bytes'] - self._last_disk_io['read_bytes']
            write_delta = current_io['write_bytes'] - self._last_disk_io['write_bytes']
            total_bytes = read_delta + write_delta
            
            self._last_disk_io = current_io
            
            # Convert bytes/s to MB/s
            return (total_bytes / time_delta) / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Disk I/O rate calculation error: {e}")
            return 0.0
    
    def _calculate_network_io_rate(self, current_time: float) -> float:
        """Calculate network I/O rate in MB/s."""
        try:
            current_io = self._get_network_io_counters()
            time_delta = current_time - self._last_sample_time
            
            if time_delta <= 0:
                return 0.0
            
            sent_delta = current_io['bytes_sent'] - self._last_network_io['bytes_sent']
            recv_delta = current_io['bytes_recv'] - self._last_network_io['bytes_recv']
            total_bytes = sent_delta + recv_delta
            
            self._last_network_io = current_io
            
            # Convert bytes/s to MB/s
            return (total_bytes / time_delta) / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Network I/O rate calculation error: {e}")
            return 0.0
    
    def record_power_ranking_calculation(self):
        """Record a power ranking calculation event."""
        with self._lock:
            self._app_metrics.power_rankings_calculated += 1
    
    def record_spread_calculation(self):
        """Record a spread calculation event.""" 
        with self._lock:
            self._app_metrics.spreads_calculated += 1
    
    def record_api_request(self, success: bool = True):
        """Record an API request event."""
        with self._lock:
            self._app_metrics.api_requests_made += 1
            if not success:
                self._app_metrics.api_request_failures += 1
    
    def record_data_validation(self, success: bool = True):
        """Record a data validation event."""
        with self._lock:
            self._app_metrics.data_validation_runs += 1
            if not success:
                self._app_metrics.data_validation_failures += 1
    
    def record_export_operation(self):
        """Record an export operation event."""
        with self._lock:
            self._app_metrics.export_operations += 1
    
    def record_cache_access(self, hit: bool = True):
        """Record a cache access event."""
        with self._lock:
            if hit:
                self._app_metrics.cache_hits += 1
            else:
                self._app_metrics.cache_misses += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            latest_snapshot = self._snapshots[-1] if self._snapshots else None
            
            return {
                'current_snapshot': latest_snapshot.to_dict() if latest_snapshot else None,
                'application_metrics': self._app_metrics.to_dict(),
                'snapshot_count': len(self._snapshots),
                'collection_period_hours': len(self._snapshots) * self.sampling_interval / 3600
            }
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical performance metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [
                snapshot.to_dict() 
                for snapshot in self._snapshots
                if snapshot.timestamp >= cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self._snapshots:
            return {'error': 'No performance data available'}
        
        with self._lock:
            snapshots = list(self._snapshots)
        
        # Calculate statistics
        memory_values = [s.memory_usage_mb for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]
        
        return {
            'time_period': {
                'start': snapshots[0].timestamp.isoformat(),
                'end': snapshots[-1].timestamp.isoformat(),
                'duration_hours': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 3600,
                'sample_count': len(snapshots)
            },
            'memory_usage_mb': {
                'current': memory_values[-1],
                'average': statistics.mean(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'median': statistics.median(memory_values)
            },
            'cpu_usage_percent': {
                'current': cpu_values[-1], 
                'average': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'median': statistics.median(cpu_values)
            },
            'application_metrics': self._app_metrics.to_dict()
        }


class PerformanceMonitor:
    """
    Main performance monitoring system for NFL Projects.
    
    Integrates existing monitoring capabilities from Phases 1.3 and 3 with
    centralized metrics collection, analysis, and reporting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Optional configuration dict with monitoring parameters.
        """
        self.config = config or self._get_default_config()
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(
            sampling_interval=self.config.get('sampling_interval', 60.0),
            max_snapshots=self.config.get('max_snapshots', 1440)
        )
        
        # Integration with existing systems
        self.memory_monitor = MemoryMonitor() if MemoryMonitor else None
        self.memory_profiler = AdvancedMemoryProfiler() if AdvancedMemoryProfiler else None
        
        # Performance thresholds for alerting
        self.thresholds = self.config.get('thresholds', {
            'memory_warning_mb': 1024,
            'memory_critical_mb': 2048,
            'cpu_warning_percent': 80.0,
            'cpu_critical_percent': 90.0,
            'response_time_warning_ms': 5000,
            'response_time_critical_ms': 10000
        })
        
        # Automatic collection thread
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        logger.info("PerformanceMonitor initialized with integrated monitoring")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'sampling_interval': 60.0,  # 1 minute
            'max_snapshots': 1440,      # 24 hours
            'auto_collect': True,
            'detailed_profiling': False,
            'export_metrics': True
        }
    
    def start_monitoring(self):
        """Start automatic performance monitoring."""
        if self._collection_thread and self._collection_thread.is_alive():
            logger.warning("Performance monitoring already running")
            return
        
        self._stop_collection.clear()
        self._collection_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._collection_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic performance monitoring."""
        if self._collection_thread:
            self._stop_collection.set()
            self._collection_thread.join(timeout=10)
            
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for automatic collection."""
        while not self._stop_collection.is_set():
            try:
                # Collect performance snapshot
                snapshot = self.metrics_collector.collect_snapshot()
                
                # Check thresholds and log warnings
                self._check_performance_thresholds(snapshot)
                
                # Wait for next collection
                if self._stop_collection.wait(self.config['sampling_interval']):
                    break
                    
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                # Continue monitoring despite errors
                time.sleep(self.config['sampling_interval'])
    
    def _check_performance_thresholds(self, snapshot: PerformanceSnapshot):
        """Check performance thresholds and log warnings."""
        # Memory threshold checking
        if snapshot.memory_usage_mb >= self.thresholds['memory_critical_mb']:
            logger.critical(f"Critical memory usage: {snapshot.memory_usage_mb:.1f}MB")
        elif snapshot.memory_usage_mb >= self.thresholds['memory_warning_mb']:
            logger.warning(f"High memory usage: {snapshot.memory_usage_mb:.1f}MB")
        
        # CPU threshold checking
        if snapshot.cpu_percent >= self.thresholds['cpu_critical_percent']:
            logger.critical(f"Critical CPU usage: {snapshot.cpu_percent:.1f}%")
        elif snapshot.cpu_percent >= self.thresholds['cpu_warning_percent']:
            logger.warning(f"High CPU usage: {snapshot.cpu_percent:.1f}%")
        
        # Response time checking (if available)
        if snapshot.response_time_ms:
            if snapshot.response_time_ms >= self.thresholds['response_time_critical_ms']:
                logger.critical(f"Critical response time: {snapshot.response_time_ms:.1f}ms")
            elif snapshot.response_time_ms >= self.thresholds['response_time_warning_ms']:
                logger.warning(f"High response time: {snapshot.response_time_ms:.1f}ms")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics and status."""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # Add Phase 3 memory profiling if available
        if self.memory_profiler:
            try:
                profiling_stats = self.memory_profiler.get_memory_stats()
                current_metrics['memory_profiling'] = {
                    'profiles_collected': len(profiling_stats.get('profiles', [])),
                    'peak_memory_mb': profiling_stats.get('peak_memory', {}).get('rss_mb', 0),
                    'optimization_recommendations': len(profiling_stats.get('optimization_recommendations', []))
                }
            except Exception as e:
                logger.error(f"Memory profiling stats error: {e}")
        
        return current_metrics
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        summary = self.metrics_collector.get_performance_summary()
        historical = self.metrics_collector.get_historical_metrics(hours)
        
        # Add integrated monitoring data
        integrated_data = {}
        
        if self.memory_monitor:
            try:
                memory_stats = self.memory_monitor.get_memory_stats()
                integrated_data['memory_monitoring'] = {
                    'current_memory': memory_stats.get('current_memory', {}),
                    'peak_memory': memory_stats.get('peak_memory', {}),
                    'profiles_count': len(memory_stats.get('profiles', []))
                }
            except Exception as e:
                logger.error(f"Memory monitor integration error: {e}")
        
        return {
            'report_generated': datetime.utcnow().isoformat(),
            'performance_summary': summary,
            'historical_data': historical,
            'integrated_monitoring': integrated_data,
            'monitoring_status': {
                'auto_collection_active': self._collection_thread and self._collection_thread.is_alive(),
                'collection_interval_seconds': self.config['sampling_interval'],
                'max_retention_hours': self.config['max_snapshots'] * self.config['sampling_interval'] / 3600
            }
        }
    
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling specific operations.
        
        Args:
            operation_name: Name of the operation being profiled.
            
        Example:
            with monitor.profile_operation("power_ranking_calculation"):
                rankings = calculate_power_rankings(data)
        """
        return OperationProfiler(self.metrics_collector, operation_name)
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics to file."""
        report = self.get_performance_report()
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
            elif format.lower() == 'csv':
                # Export summary as CSV
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write headers and data for historical metrics
                    if report['historical_data']:
                        headers = report['historical_data'][0].keys()
                        writer.writerow(headers)
                        for row in report['historical_data']:
                            writer.writerow(row.values())
            
            logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise


class OperationProfiler:
    """Context manager for profiling specific operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, operation_name: str):
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        duration_ms = (end_time - self.start_time) * 1000
        memory_delta_mb = end_memory - self.start_memory
        
        logger.info(f"Operation '{self.operation_name}' completed: "
                   f"{duration_ms:.2f}ms, memory delta: {memory_delta_mb:+.2f}MB")
        
        # Record operation-specific metrics
        if 'power_ranking' in self.operation_name.lower():
            self.metrics_collector.record_power_ranking_calculation()
        elif 'spread' in self.operation_name.lower():
            self.metrics_collector.record_spread_calculation()
        elif 'export' in self.operation_name.lower():
            self.metrics_collector.record_export_operation()


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


# Decorator for automatic operation profiling
def profile_performance(operation_name: Optional[str] = None):
    """
    Decorator for automatic performance profiling of functions.
    
    Args:
        operation_name: Optional operation name. If None, uses function name.
        
    Example:
        @profile_performance("ranking_calculation")
        def calculate_power_rankings(data):
            # Function implementation
            return rankings
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            monitor = get_performance_monitor()
            
            with monitor.profile_operation(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator