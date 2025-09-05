"""
Memory monitoring utilities for NFL Power Rankings system.
Provides comprehensive memory profiling, monitoring, and optimization tools.
"""

import psutil
import gc
import sys
import time
import logging
import tracemalloc
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
import threading
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    heap_objects: int
    gc_collections: Dict[int, int]
    tracemalloc_peak: Optional[float] = None
    tracemalloc_current: Optional[float] = None


@dataclass
class MemoryProfile:
    """Complete memory profile for a function or operation."""
    function_name: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    peak_snapshot: MemorySnapshot
    duration_seconds: float
    memory_delta_mb: float
    peak_memory_mb: float
    objects_created: int
    objects_deleted: int
    gc_triggered: bool


class MemoryMonitor:
    """Advanced memory monitoring and profiling system."""
    
    def __init__(self, enable_tracemalloc: bool = True, sample_interval: float = 0.1):
        self.enable_tracemalloc = enable_tracemalloc
        self.sample_interval = sample_interval
        self.snapshots: List[MemorySnapshot] = []
        self.profiles: List[MemoryProfile] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Object tracking
        self.object_refs = weakref.WeakSet()
        self.object_counts = defaultdict(int)
        
        # Memory alerts
        self.memory_threshold_mb = 500  # Alert if over 500MB
        self.alert_callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def get_current_memory(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        # Get GC statistics
        gc_stats = {i: gc.get_count()[i] for i in range(len(gc.get_count()))}
        
        # Get tracemalloc info if available
        tracemalloc_current = None
        tracemalloc_peak = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current = current / 1024 / 1024  # Convert to MB
            tracemalloc_peak = peak / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            available_mb=system_memory.available / 1024 / 1024,
            heap_objects=len(gc.get_objects()),
            gc_collections=gc_stats,
            tracemalloc_current=tracemalloc_current,
            tracemalloc_peak=tracemalloc_peak
        )
    
    def start_monitoring(self) -> None:
        """Start continuous memory monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self.get_current_memory()
                self.snapshots.append(snapshot)
                
                # Keep only recent snapshots (last 1000)
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]
                
                # Check for memory alerts
                if snapshot.rss_mb > self.memory_threshold_mb:
                    for callback in self.alert_callbacks:
                        try:
                            callback(snapshot)
                        except Exception as e:
                            logger.warning(f"Memory alert callback failed: {e}")
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(1)  # Longer sleep on error
    
    def add_alert_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add callback for memory alerts."""
        self.alert_callbacks.append(callback)
    
    def set_memory_threshold(self, threshold_mb: float) -> None:
        """Set memory threshold for alerts."""
        self.memory_threshold_mb = threshold_mb
        logger.info(f"Memory threshold set to {threshold_mb} MB")
    
    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager for memory profiling."""
        # Take initial snapshot
        start_snapshot = self.get_current_memory()
        start_objects = len(gc.get_objects())
        start_time = time.time()
        
        # Track peak memory during operation
        peak_snapshot = start_snapshot
        
        try:
            yield self
            
            # Monitor for peak memory usage
            current_snapshot = self.get_current_memory()
            if current_snapshot.rss_mb > peak_snapshot.rss_mb:
                peak_snapshot = current_snapshot
            
        finally:
            # Take final snapshot
            end_snapshot = self.get_current_memory()
            end_objects = len(gc.get_objects())
            end_time = time.time()
            
            # Calculate deltas
            memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
            objects_created = max(0, end_objects - start_objects)
            objects_deleted = max(0, start_objects - end_objects)
            duration = end_time - start_time
            
            # Check if GC was triggered
            gc_triggered = any(
                end_snapshot.gc_collections[i] > start_snapshot.gc_collections[i]
                for i in range(len(end_snapshot.gc_collections))
            )
            
            # Create profile
            profile = MemoryProfile(
                function_name=operation_name,
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                peak_snapshot=peak_snapshot,
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                peak_memory_mb=peak_snapshot.rss_mb,
                objects_created=objects_created,
                objects_deleted=objects_deleted,
                gc_triggered=gc_triggered
            )
            
            self.profiles.append(profile)
            
            # Log significant memory changes
            if abs(memory_delta) > 10:  # Log if delta > 10MB
                logger.info(f"Memory profile: {operation_name} - "
                          f"Delta: {memory_delta:+.1f}MB, "
                          f"Peak: {peak_snapshot.rss_mb:.1f}MB, "
                          f"Duration: {duration:.3f}s")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling function memory usage."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_memory(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.snapshots:
            current = self.get_current_memory()
            self.snapshots.append(current)
        
        current = self.snapshots[-1]
        
        # Calculate trends if we have multiple snapshots
        trends = {}
        if len(self.snapshots) >= 2:
            recent = self.snapshots[-10:]  # Last 10 snapshots
            rss_trend = (recent[-1].rss_mb - recent[0].rss_mb) / len(recent)
            obj_trend = (recent[-1].heap_objects - recent[0].heap_objects) / len(recent)
            trends = {
                'rss_trend_mb_per_sample': rss_trend,
                'object_trend_per_sample': obj_trend
            }
        
        # Profile statistics
        profile_stats = {}
        if self.profiles:
            total_profiles = len(self.profiles)
            avg_memory_delta = sum(p.memory_delta_mb for p in self.profiles) / total_profiles
            max_memory_delta = max(p.memory_delta_mb for p in self.profiles)
            min_memory_delta = min(p.memory_delta_mb for p in self.profiles)
            
            profile_stats = {
                'total_profiles': total_profiles,
                'avg_memory_delta_mb': avg_memory_delta,
                'max_memory_delta_mb': max_memory_delta,
                'min_memory_delta_mb': min_memory_delta,
                'functions_profiled': len(set(p.function_name for p in self.profiles))
            }
        
        return {
            'current_memory': {
                'rss_mb': current.rss_mb,
                'vms_mb': current.vms_mb,
                'percent': current.percent,
                'available_mb': current.available_mb,
                'heap_objects': current.heap_objects
            },
            'tracemalloc': {
                'current_mb': current.tracemalloc_current,
                'peak_mb': current.tracemalloc_peak,
                'enabled': tracemalloc.is_tracing()
            },
            'monitoring': {
                'active': self.monitoring_active,
                'snapshots_collected': len(self.snapshots),
                'sample_interval': self.sample_interval,
                'threshold_mb': self.memory_threshold_mb
            },
            'trends': trends,
            'profiles': profile_stats,
            'garbage_collection': {
                'collections': dict(current.gc_collections),
                'thresholds': gc.get_threshold(),
                'stats': gc.get_stats()
            }
        }
    
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consuming objects by type."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for stat in top_stats[:limit]:
            consumers.append({
                'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
                'avg_size_bytes': stat.size / max(stat.count, 1)
            })
        
        return consumers
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_objects = len(gc.get_objects())
        
        collected = {
            'generation_0': gc.collect(0),
            'generation_1': gc.collect(1), 
            'generation_2': gc.collect(2)
        }
        
        after_objects = len(gc.get_objects())
        collected['total_objects'] = collected['generation_0'] + collected['generation_1'] + collected['generation_2']
        collected['objects_before'] = before_objects
        collected['objects_after'] = after_objects
        collected['objects_freed'] = before_objects - after_objects
        
        logger.info(f"Garbage collection: freed {collected['objects_freed']} objects")
        return collected
    
    def analyze_memory_leaks(self) -> List[Dict[str, Any]]:
        """Analyze potential memory leaks."""
        if len(self.snapshots) < 10:
            return []
        
        leaks = []
        
        # Check for continuously growing memory
        recent_snapshots = self.snapshots[-20:]  # Last 20 snapshots
        if len(recent_snapshots) >= 10:
            memory_growth = []
            for i in range(1, len(recent_snapshots)):
                growth = recent_snapshots[i].rss_mb - recent_snapshots[i-1].rss_mb
                memory_growth.append(growth)
            
            # Check if memory is consistently growing
            positive_growth_count = sum(1 for g in memory_growth if g > 0)
            if positive_growth_count > len(memory_growth) * 0.8:  # 80% of samples show growth
                total_growth = sum(memory_growth)
                leaks.append({
                    'type': 'memory_growth',
                    'description': 'Continuous memory growth detected',
                    'total_growth_mb': total_growth,
                    'growth_rate_mb_per_sample': total_growth / len(memory_growth),
                    'confidence': positive_growth_count / len(memory_growth)
                })
        
        # Check for growing object counts
        if len(recent_snapshots) >= 5:
            object_growth = []
            for i in range(1, len(recent_snapshots)):
                growth = recent_snapshots[i].heap_objects - recent_snapshots[i-1].heap_objects
                object_growth.append(growth)
            
            avg_object_growth = sum(object_growth) / len(object_growth)
            if avg_object_growth > 100:  # More than 100 objects per sample
                leaks.append({
                    'type': 'object_growth',
                    'description': 'Continuous object count growth detected', 
                    'avg_objects_per_sample': avg_object_growth,
                    'total_object_growth': sum(object_growth)
                })
        
        return leaks
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get memory optimization suggestions based on profiling data."""
        suggestions = []
        stats = self.get_memory_stats()
        
        current_mb = stats['current_memory']['rss_mb']
        
        # High memory usage
        if current_mb > 200:
            suggestions.append(f"High memory usage detected ({current_mb:.1f}MB). Consider implementing data streaming.")
        
        # Many heap objects
        if stats['current_memory']['heap_objects'] > 100000:
            suggestions.append(f"Large number of heap objects ({stats['current_memory']['heap_objects']}). Consider object pooling.")
        
        # Memory growth trend
        if 'trends' in stats and 'rss_trend_mb_per_sample' in stats['trends']:
            trend = stats['trends']['rss_trend_mb_per_sample']
            if trend > 1.0:
                suggestions.append(f"Memory growth trend detected ({trend:.2f}MB per sample). Check for memory leaks.")
        
        # Profile-based suggestions
        if 'profiles' in stats and stats['profiles']:
            avg_delta = stats['profiles']['avg_memory_delta_mb']
            max_delta = stats['profiles']['max_memory_delta_mb']
            
            if avg_delta > 10:
                suggestions.append(f"Functions have high average memory delta ({avg_delta:.1f}MB). Consider optimizing algorithms.")
            
            if max_delta > 50:
                suggestions.append(f"Some functions use excessive memory ({max_delta:.1f}MB peak). Implement lazy loading.")
        
        # GC suggestions
        gc_stats = stats['garbage_collection']['stats']
        if gc_stats:
            total_collections = sum(stat['collections'] for stat in gc_stats)
            if total_collections > 100:
                suggestions.append("Frequent garbage collection detected. Consider reducing object creation.")
        
        return suggestions or ["Memory usage appears optimal."]
    
    def clear_profiles(self) -> None:
        """Clear stored memory profiles."""
        self.profiles.clear()
        logger.info("Memory profiles cleared")
    
    def clear_snapshots(self) -> None:
        """Clear stored memory snapshots."""
        self.snapshots.clear()
        logger.info("Memory snapshots cleared")


# Global instance for convenience
_global_monitor = None

def get_global_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


# Convenience functions
def profile_memory(operation_name: str):
    """Decorator for profiling memory usage of functions."""
    return get_global_monitor().profile_function(operation_name)


def start_monitoring():
    """Start global memory monitoring."""
    get_global_monitor().start_monitoring()


def stop_monitoring():
    """Stop global memory monitoring."""
    get_global_monitor().stop_monitoring()


def get_memory_stats():
    """Get current memory statistics."""
    return get_global_monitor().get_memory_stats()


def force_gc():
    """Force garbage collection."""
    return get_global_monitor().force_garbage_collection()


# Memory alert callback example
def default_memory_alert(snapshot: MemorySnapshot):
    """Default memory alert handler."""
    logger.warning(f"Memory alert: {snapshot.rss_mb:.1f}MB RSS, {snapshot.percent:.1f}% of system memory")