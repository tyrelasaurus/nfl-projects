"""
Advanced memory profiling utilities for detailed analysis and optimization.
Provides line-by-line profiling, object tracking, and memory leak detection.
"""

import sys
import gc
import tracemalloc
import linecache
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict, Counter
import weakref
import threading
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class MemoryFrame:
    """Memory allocation frame information."""
    filename: str
    lineno: int
    function: str
    code_snippet: str
    size_mb: float
    count: int


@dataclass
class ObjectProfile:
    """Profile of object allocations by type."""
    type_name: str
    count: int
    total_size_mb: float
    avg_size_bytes: float
    instances: List[weakref.ReferenceType] = field(default_factory=list)


@dataclass
class FunctionProfile:
    """Detailed function memory profile."""
    function_name: str
    module: str
    calls: int
    total_memory_mb: float
    avg_memory_mb: float
    max_memory_mb: float
    peak_objects: int
    frames: List[MemoryFrame]


class AdvancedMemoryProfiler:
    """Advanced memory profiler with detailed analysis capabilities."""
    
    def __init__(self, enable_tracemalloc: bool = True, trace_filters: Optional[List[str]] = None):
        self.enable_tracemalloc = enable_tracemalloc
        self.trace_filters = trace_filters or []
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.object_profiles: Dict[str, ObjectProfile] = {}
        self.allocation_history: List[Tuple[float, str, int, float]] = []
        self.profiling_active = False
        
        # Object tracking
        self.tracked_objects = weakref.WeakSet()
        self.object_creation_callbacks: List[Callable] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            # Start with more detailed tracing
            tracemalloc.start(25)  # Keep 25 frames
    
    def profile_function_detailed(self, include_line_profiling: bool = True):
        """Decorator for detailed function memory profiling."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_function_execution(func, args, kwargs, include_line_profiling)
            return wrapper
        return decorator
    
    def _profile_function_execution(self, func: Callable, args: tuple, kwargs: dict, 
                                  include_line_profiling: bool) -> Any:
        """Execute function with detailed profiling."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Take initial snapshot
        if tracemalloc.is_tracing():
            snapshot_before = tracemalloc.take_snapshot()
            if self.trace_filters:
                snapshot_before = snapshot_before.filter_traces(
                    [tracemalloc.Filter(False, pattern) for pattern in self.trace_filters]
                )
        
        objects_before = len(gc.get_objects())
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Take final snapshot
            if tracemalloc.is_tracing():
                snapshot_after = tracemalloc.take_snapshot()
                if self.trace_filters:
                    snapshot_after = snapshot_after.filter_traces(
                        [tracemalloc.Filter(False, pattern) for pattern in self.trace_filters]
                    )
                
                # Compare snapshots
                top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                
                # Extract memory frames
                frames = []
                total_memory_mb = 0.0
                max_memory_mb = 0.0
                
                for stat in top_stats[:50]:  # Top 50 allocations
                    size_mb = stat.size_diff / 1024 / 1024
                    total_memory_mb += abs(size_mb)
                    max_memory_mb = max(max_memory_mb, abs(size_mb))
                    
                    if stat.traceback:
                        frame_info = stat.traceback.format()[-1]  # Get most recent frame
                        filename = stat.traceback[0].filename if stat.traceback else 'unknown'
                        lineno = stat.traceback[0].lineno if stat.traceback else 0
                        
                        # Get code snippet
                        code_snippet = ''
                        if include_line_profiling:
                            try:
                                code_snippet = linecache.getline(filename, lineno).strip()
                            except:
                                code_snippet = 'Unable to retrieve code'
                        
                        frame = MemoryFrame(
                            filename=Path(filename).name if filename else 'unknown',
                            lineno=lineno,
                            function=func.__name__,
                            code_snippet=code_snippet,
                            size_mb=size_mb,
                            count=stat.count_diff
                        )
                        frames.append(frame)
                
                # Update function profile
                objects_after = len(gc.get_objects())
                execution_time = time.time() - start_time
                
                with self.lock:
                    if func_name not in self.function_profiles:
                        self.function_profiles[func_name] = FunctionProfile(
                            function_name=func.__name__,
                            module=func.__module__,
                            calls=0,
                            total_memory_mb=0.0,
                            avg_memory_mb=0.0,
                            max_memory_mb=0.0,
                            peak_objects=0,
                            frames=[]
                        )
                    
                    profile = self.function_profiles[func_name]
                    profile.calls += 1
                    profile.total_memory_mb += total_memory_mb
                    profile.avg_memory_mb = profile.total_memory_mb / profile.calls
                    profile.max_memory_mb = max(profile.max_memory_mb, total_memory_mb)
                    profile.peak_objects = max(profile.peak_objects, objects_after - objects_before)
                    
                    # Keep only top frames to prevent memory bloat
                    profile.frames.extend(frames[:10])
                    if len(profile.frames) > 100:
                        profile.frames = profile.frames[-100:]
            
            return result
            
        except Exception as e:
            # Still record the profile data for failed executions
            logger.warning(f"Function {func_name} failed during profiling: {e}")
            raise
    
    def track_object_allocations(self, obj_type: type, callback: Optional[Callable] = None) -> None:
        """Track allocations of specific object types."""
        original_new = obj_type.__new__
        
        def tracked_new(cls, *args, **kwargs):
            instance = original_new(cls) if original_new is object.__new__ else original_new(cls, *args, **kwargs)
            
            # Track the instance
            self.tracked_objects.add(instance)
            
            # Update object profile
            type_name = cls.__name__
            instance_size = sys.getsizeof(instance)
            
            with self.lock:
                if type_name not in self.object_profiles:
                    self.object_profiles[type_name] = ObjectProfile(
                        type_name=type_name,
                        count=0,
                        total_size_mb=0.0,
                        avg_size_bytes=0.0,
                        instances=[]
                    )
                
                profile = self.object_profiles[type_name]
                profile.count += 1
                profile.total_size_mb += instance_size / 1024 / 1024
                profile.avg_size_bytes = (profile.total_size_mb * 1024 * 1024) / profile.count
                profile.instances.append(weakref.ref(instance))
                
                # Keep only recent instances to prevent memory leaks
                if len(profile.instances) > 1000:
                    profile.instances = profile.instances[-1000:]
            
            # Call callback if provided
            if callback:
                callback(instance, instance_size)
            
            return instance
        
        obj_type.__new__ = tracked_new
        logger.info(f"Started tracking allocations for {obj_type.__name__}")
    
    def analyze_memory_hotspots(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Analyze memory hotspots in the application."""
        hotspots = []
        
        # Function-based hotspots
        for func_name, profile in sorted(self.function_profiles.items(), 
                                       key=lambda x: x[1].total_memory_mb, reverse=True)[:top_n]:
            hotspots.append({
                'type': 'function',
                'name': func_name,
                'total_memory_mb': profile.total_memory_mb,
                'avg_memory_mb': profile.avg_memory_mb,
                'calls': profile.calls,
                'max_memory_mb': profile.max_memory_mb,
                'peak_objects': profile.peak_objects,
                'frames_count': len(profile.frames)
            })
        
        # Object type hotspots
        for type_name, profile in sorted(self.object_profiles.items(),
                                       key=lambda x: x[1].total_size_mb, reverse=True)[:top_n]:
            active_instances = sum(1 for ref in profile.instances if ref() is not None)
            hotspots.append({
                'type': 'object_type',
                'name': type_name,
                'total_memory_mb': profile.total_size_mb,
                'count': profile.count,
                'avg_size_bytes': profile.avg_size_bytes,
                'active_instances': active_instances
            })
        
        # Current memory snapshot hotspots
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            for i, stat in enumerate(top_stats[:top_n]):
                filename = stat.traceback[0].filename if stat.traceback else 'unknown'
                lineno = stat.traceback[0].lineno if stat.traceback else 0
                
                hotspots.append({
                    'type': 'current_allocation',
                    'name': f"{Path(filename).name}:{lineno}",
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count,
                    'avg_size_bytes': stat.size / max(stat.count, 1),
                    'rank': i + 1
                })
        
        return hotspots
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        # Check tracked objects for potential leaks
        for type_name, profile in self.object_profiles.items():
            # Count active instances
            active_instances = []
            for ref in profile.instances:
                obj = ref()
                if obj is not None:
                    active_instances.append(obj)
            
            active_count = len(active_instances)
            active_size_mb = sum(sys.getsizeof(obj) for obj in active_instances) / 1024 / 1024
            
            # Potential leak if many instances are still alive
            if active_size_mb > threshold_mb and active_count > 100:
                leaks.append({
                    'type': 'object_accumulation',
                    'object_type': type_name,
                    'active_instances': active_count,
                    'active_size_mb': active_size_mb,
                    'total_created': profile.count,
                    'leak_probability': min(1.0, active_count / max(profile.count, 1))
                })
        
        # Check function profiles for functions that consistently allocate memory
        for func_name, profile in self.function_profiles.items():
            if profile.calls > 10 and profile.avg_memory_mb > 5.0:
                # High average memory allocation might indicate a leak
                leak_score = (profile.avg_memory_mb * profile.calls) / 100
                if leak_score > 1.0:
                    leaks.append({
                        'type': 'function_leak',
                        'function': func_name,
                        'avg_memory_mb': profile.avg_memory_mb,
                        'calls': profile.calls,
                        'total_memory_mb': profile.total_memory_mb,
                        'leak_score': leak_score
                    })
        
        return sorted(leaks, key=lambda x: x.get('leak_score', x.get('active_size_mb', 0)), reverse=True)
    
    def generate_memory_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive memory profiling report."""
        report = {
            'timestamp': time.time(),
            'profiling_summary': {
                'functions_profiled': len(self.function_profiles),
                'object_types_tracked': len(self.object_profiles),
                'tracemalloc_enabled': tracemalloc.is_tracing()
            },
            'memory_hotspots': self.analyze_memory_hotspots(20),
            'memory_leaks': self.detect_memory_leaks(),
            'function_profiles': {},
            'object_profiles': {},
            'optimization_recommendations': []
        }
        
        # Add function profiles
        for func_name, profile in self.function_profiles.items():
            report['function_profiles'][func_name] = {
                'calls': profile.calls,
                'total_memory_mb': profile.total_memory_mb,
                'avg_memory_mb': profile.avg_memory_mb,
                'max_memory_mb': profile.max_memory_mb,
                'peak_objects': profile.peak_objects,
                'top_frames': [
                    {
                        'filename': frame.filename,
                        'lineno': frame.lineno,
                        'size_mb': frame.size_mb,
                        'count': frame.count,
                        'code_snippet': frame.code_snippet[:100]  # Truncate long lines
                    }
                    for frame in sorted(profile.frames, key=lambda f: abs(f.size_mb), reverse=True)[:5]
                ]
            }
        
        # Add object profiles
        for type_name, profile in self.object_profiles.items():
            active_instances = sum(1 for ref in profile.instances if ref() is not None)
            report['object_profiles'][type_name] = {
                'total_created': profile.count,
                'active_instances': active_instances,
                'total_size_mb': profile.total_size_mb,
                'avg_size_bytes': profile.avg_size_bytes
            }
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(report)
        report['optimization_recommendations'] = recommendations
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Memory profiling report saved to {output_file}")
        
        return report
    
    def _generate_optimization_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # High memory functions
        high_memory_functions = [
            hotspot for hotspot in report['memory_hotspots']
            if hotspot['type'] == 'function' and hotspot['total_memory_mb'] > 50
        ]
        
        if high_memory_functions:
            func_names = [f['name'] for f in high_memory_functions[:3]]
            recommendations.append(
                f"High memory functions detected: {', '.join(func_names)}. "
                f"Consider implementing data streaming or chunking."
            )
        
        # Memory leaks
        if report['memory_leaks']:
            leak_types = set(leak['type'] for leak in report['memory_leaks'])
            recommendations.append(
                f"Potential memory leaks detected: {', '.join(leak_types)}. "
                f"Review object lifecycle management."
            )
        
        # Object accumulation
        object_leaks = [
            leak for leak in report['memory_leaks']
            if leak['type'] == 'object_accumulation'
        ]
        
        if object_leaks:
            object_types = [leak['object_type'] for leak in object_leaks[:3]]
            recommendations.append(
                f"Object accumulation in: {', '.join(object_types)}. "
                f"Consider implementing object pooling or weak references."
            )
        
        # High object creation
        high_creation_objects = [
            obj for obj_name, obj in report['object_profiles'].items()
            if obj['total_created'] > 10000
        ]
        
        if high_creation_objects:
            recommendations.append(
                f"High object creation rate detected. "
                f"Consider object reuse patterns or lazy initialization."
            )
        
        return recommendations or ["Memory usage appears optimal based on current profiling data."]
    
    def clear_profiles(self) -> None:
        """Clear all profiling data."""
        with self.lock:
            self.function_profiles.clear()
            self.object_profiles.clear()
            self.allocation_history.clear()
            self.tracked_objects.clear()
        
        logger.info("Memory profiling data cleared")
    
    def get_profiling_overhead(self) -> Dict[str, Any]:
        """Estimate the overhead of memory profiling."""
        overhead_mb = 0.0
        
        # Estimate function profile overhead
        for profile in self.function_profiles.values():
            overhead_mb += sys.getsizeof(profile) / 1024 / 1024
            overhead_mb += sum(sys.getsizeof(frame) for frame in profile.frames) / 1024 / 1024
        
        # Estimate object profile overhead
        for profile in self.object_profiles.values():
            overhead_mb += sys.getsizeof(profile) / 1024 / 1024
            overhead_mb += sys.getsizeof(profile.instances) / 1024 / 1024
        
        # Estimate tracemalloc overhead
        tracemalloc_overhead = 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_overhead = peak / 1024 / 1024 * 0.1  # Rough estimate
        
        return {
            'total_overhead_mb': overhead_mb + tracemalloc_overhead,
            'function_profiles_mb': overhead_mb,
            'tracemalloc_overhead_mb': tracemalloc_overhead,
            'functions_tracked': len(self.function_profiles),
            'object_types_tracked': len(self.object_profiles)
        }


# Global profiler instance
_global_profiler = None

def get_global_profiler() -> AdvancedMemoryProfiler:
    """Get global memory profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = AdvancedMemoryProfiler()
    return _global_profiler


# Convenience decorators
def profile_memory_detailed(include_line_profiling: bool = True):
    """Decorator for detailed memory profiling."""
    return get_global_profiler().profile_function_detailed(include_line_profiling)


def track_object_type(obj_type: type, callback: Optional[Callable] = None):
    """Track allocations of specific object type."""
    get_global_profiler().track_object_allocations(obj_type, callback)


def generate_memory_report(output_file: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive memory report."""
    output_path = Path(output_file) if output_file else None
    return get_global_profiler().generate_memory_report(output_path)