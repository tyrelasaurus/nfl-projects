"""
Lazy loading system for efficient data access in NFL Power Rankings.
Provides lazy evaluation, caching, and on-demand data loading capabilities.
"""

import weakref
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps, partial
from collections import defaultdict
from contextlib import contextmanager
import gc

from .memory_monitor import get_global_monitor
from .optimized_structures import MemoryEfficientCache

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading behavior."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size: int = 1000
    max_cache_memory_mb: float = 50.0
    auto_unload_threshold_mb: float = 100.0
    enable_memory_monitoring: bool = True
    preload_on_access: bool = False


class LazyProperty:
    """Lazy property descriptor that loads data on first access."""
    
    def __init__(self, loader: Callable[[], Any], cache_key: Optional[str] = None):
        self.loader = loader
        self.cache_key = cache_key or f"lazy_prop_{id(self)}"
        self.loaded = False
        self.value = None
        self.lock = threading.RLock()
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        if not self.loaded:
            with self.lock:
                if not self.loaded:  # Double-check locking
                    logger.debug(f"Lazy loading property: {self.cache_key}")
                    self.value = self.loader()
                    self.loaded = True
        
        return self.value
    
    def __set_name__(self, owner, name):
        self.cache_key = f"{owner.__name__}.{name}"
    
    def reset(self):
        """Reset the lazy property to force reload."""
        with self.lock:
            self.loaded = False
            self.value = None


class LazyLoader(Generic[T]):
    """Generic lazy loader with caching and memory management."""
    
    def __init__(self, key: str, loader: Callable[[], T], config: Optional[LazyLoadConfig] = None):
        self.key = key
        self.loader = loader
        self.config = config or LazyLoadConfig()
        self._value: Optional[T] = None
        self._loaded = False
        self._loading = False
        self._load_time = 0.0
        self._access_count = 0
        self._lock = threading.RLock()
        
        # Memory monitoring
        self.memory_monitor = get_global_monitor() if self.config.enable_memory_monitoring else None
        
    def get(self, force_reload: bool = False) -> T:
        """Get the lazy-loaded value."""
        with self._lock:
            if force_reload or not self._loaded:
                if self._loading:
                    # Another thread is loading, wait for it
                    while self._loading:
                        time.sleep(0.01)
                    if self._loaded:
                        self._access_count += 1
                        return self._value
                
                self._load_value()
            
            self._access_count += 1
            return self._value
    
    def _load_value(self):
        """Internal method to load the value."""
        self._loading = True
        start_time = time.time()
        
        try:
            logger.info(f"Lazy loading: {self.key}")
            
            if self.memory_monitor:
                with self.memory_monitor.profile_memory(f"lazy_load_{self.key}"):
                    self._value = self.loader()
            else:
                self._value = self.loader()
            
            self._loaded = True
            self._load_time = time.time() - start_time
            
            logger.debug(f"Lazy loaded {self.key} in {self._load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to lazy load {self.key}: {e}")
            raise
        finally:
            self._loading = False
    
    def is_loaded(self) -> bool:
        """Check if value is loaded."""
        return self._loaded
    
    def unload(self):
        """Unload the value to free memory."""
        with self._lock:
            if self._loaded:
                self._value = None
                self._loaded = False
                gc.collect()
                logger.debug(f"Unloaded lazy value: {self.key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'key': self.key,
            'loaded': self._loaded,
            'access_count': self._access_count,
            'load_time_seconds': self._load_time,
            'memory_size_mb': self._estimate_memory_size()
        }
    
    def _estimate_memory_size(self) -> float:
        """Estimate memory size of loaded value."""
        if not self._loaded or self._value is None:
            return 0.0
        
        import sys
        return sys.getsizeof(self._value) / 1024 / 1024


class LazyDataManager:
    """Manager for lazy-loaded data with caching and memory management."""
    
    def __init__(self, config: Optional[LazyLoadConfig] = None):
        self.config = config or LazyLoadConfig()
        self.loaders: Dict[str, LazyLoader] = {}
        self.cache = MemoryEfficientCache(
            max_size=self.config.max_cache_size,
            max_memory_mb=self.config.max_cache_memory_mb
        ) if self.config.cache_enabled else None
        
        self.access_patterns = defaultdict(list)
        self.memory_monitor = get_global_monitor() if self.config.enable_memory_monitoring else None
        self._lock = threading.RLock()
    
    def register_loader(self, key: str, loader: Callable[[], Any]) -> LazyLoader:
        """Register a lazy loader."""
        with self._lock:
            lazy_loader = LazyLoader(key, loader, self.config)
            self.loaders[key] = lazy_loader
            logger.debug(f"Registered lazy loader: {key}")
            return lazy_loader
    
    def get(self, key: str, force_reload: bool = False) -> Any:
        """Get data using lazy loading."""
        if key not in self.loaders:
            raise KeyError(f"No lazy loader registered for key: {key}")
        
        # Record access pattern
        self.access_patterns[key].append(time.time())
        
        # Check cache first
        if self.cache and not force_reload:
            cached_value = self.cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for lazy key: {key}")
                return cached_value
        
        # Load using lazy loader
        loader = self.loaders[key]
        value = loader.get(force_reload)
        
        # Cache the value
        if self.cache:
            self.cache.set(key, value)
        
        # Check memory usage and auto-unload if needed
        if self.config.auto_unload_threshold_mb > 0:
            self._check_memory_and_unload()
        
        return value
    
    def preload(self, keys: List[str]) -> Dict[str, bool]:
        """Preload multiple keys."""
        results = {}
        
        for key in keys:
            try:
                if key in self.loaders:
                    self.get(key)
                    results[key] = True
                else:
                    results[key] = False
            except Exception as e:
                logger.error(f"Failed to preload {key}: {e}")
                results[key] = False
        
        return results
    
    def unload(self, key: str) -> bool:
        """Unload specific data."""
        if key in self.loaders:
            self.loaders[key].unload()
            if self.cache:
                # Remove from cache as well
                self.cache.set(key, None)  # This will effectively remove it
            return True
        return False
    
    def unload_all(self):
        """Unload all data to free memory."""
        with self._lock:
            for loader in self.loaders.values():
                loader.unload()
            
            if self.cache:
                self.cache.clear()
            
            gc.collect()
            logger.info("Unloaded all lazy data")
    
    def _check_memory_and_unload(self):
        """Check memory usage and unload old data if needed."""
        if not self.memory_monitor:
            return
        
        current_memory = self.memory_monitor.get_current_memory()
        if current_memory.rss_mb > self.config.auto_unload_threshold_mb:
            # Find least recently used items
            lru_keys = self._get_lru_keys(5)  # Get 5 least recently used
            
            for key in lru_keys:
                self.unload(key)
                logger.info(f"Auto-unloaded {key} due to memory pressure")
    
    def _get_lru_keys(self, count: int) -> List[str]:
        """Get least recently used keys."""
        current_time = time.time()
        key_scores = []
        
        for key, access_times in self.access_patterns.items():
            if access_times:
                # Score based on recency and frequency
                recent_accesses = [t for t in access_times if current_time - t < 3600]  # Last hour
                if recent_accesses:
                    last_access = max(recent_accesses)
                    frequency = len(recent_accesses)
                    score = (current_time - last_access) / max(frequency, 1)
                else:
                    score = current_time  # Very old, high score
                
                key_scores.append((score, key))
        
        # Sort by score (higher = less recently used)
        key_scores.sort(reverse=True)
        return [key for score, key in key_scores[:count]]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'loaders_count': len(self.loaders),
            'loaded_count': sum(1 for loader in self.loaders.values() if loader.is_loaded()),
            'total_memory_mb': 0.0,
            'cache_stats': self.cache.get_stats() if self.cache else {},
            'loader_stats': {}
        }
        
        for key, loader in self.loaders.items():
            loader_stats = loader.get_stats()
            stats['loader_stats'][key] = loader_stats
            stats['total_memory_mb'] += loader_stats['memory_size_mb']
        
        return stats
    
    def get_access_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze access patterns for optimization."""
        patterns = {}
        current_time = time.time()
        
        for key, access_times in self.access_patterns.items():
            if access_times:
                recent_accesses = [t for t in access_times if current_time - t < 3600]
                patterns[key] = {
                    'total_accesses': len(access_times),
                    'recent_accesses': len(recent_accesses),
                    'last_access_seconds_ago': current_time - max(access_times),
                    'avg_access_interval': self._calculate_avg_interval(access_times),
                    'should_preload': len(recent_accesses) > 5  # Frequently accessed
                }
        
        return patterns
    
    def _calculate_avg_interval(self, access_times: List[float]) -> float:
        """Calculate average interval between accesses."""
        if len(access_times) < 2:
            return 0.0
        
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        return sum(intervals) / len(intervals)


class LazyNFLDataset:
    """Lazy-loaded NFL dataset with optimized access patterns."""
    
    def __init__(self, config: Optional[LazyLoadConfig] = None):
        self.config = config or LazyLoadConfig()
        self.manager = LazyDataManager(config)
        self._setup_nfl_loaders()
    
    def _setup_nfl_loaders(self):
        """Setup common NFL data loaders."""
        
        # Team data loader
        def load_teams():
            from ..api.espn_client import ESPNClient
            client = ESPNClient()
            return client.get_teams()
        
        self.manager.register_loader('teams', load_teams)
        
        # Current season data loader
        def load_current_season():
            from ..api.espn_client import ESPNClient
            client = ESPNClient()
            current_week = client.get_current_week()
            season_data = []
            for week in range(1, current_week + 1):
                try:
                    scoreboard = client.get_scoreboard(week)
                    season_data.extend(scoreboard.get('events', []))
                except:
                    continue
            return season_data
        
        self.manager.register_loader('current_season', load_current_season)
        
        # Historical rankings loader
        def load_historical_rankings():
            from ..api.espn_client import ESPNClient
            client = ESPNClient()
            return client.get_last_season_final_rankings()
        
        self.manager.register_loader('historical_rankings', load_historical_rankings)
    
    @property
    def teams(self):
        """Lazy-loaded team data."""
        return self.manager.get('teams')
    
    @property
    def current_season(self):
        """Lazy-loaded current season data."""
        return self.manager.get('current_season')
    
    @property
    def historical_rankings(self):
        """Lazy-loaded historical rankings."""
        return self.manager.get('historical_rankings')
    
    def preload_essentials(self):
        """Preload essential data."""
        essentials = ['teams', 'historical_rankings']
        return self.manager.preload(essentials)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        return self.manager.get_memory_stats()


def lazy_property(loader: Callable[[], Any]) -> LazyProperty:
    """Decorator to create a lazy property."""
    return LazyProperty(loader)


def lazy_method(cache_key: Optional[str] = None, ttl_seconds: int = 300):
    """Decorator to make method results lazy-loaded with caching."""
    def decorator(method):
        cache = {}
        
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            key = cache_key or f"{method.__name__}_{hash((args, tuple(kwargs.items())))}"
            
            # Check cache
            if key in cache:
                cached_time, value = cache[key]
                if time.time() - cached_time < ttl_seconds:
                    return value
                else:
                    del cache[key]
            
            # Load and cache
            logger.debug(f"Lazy loading method result: {method.__name__}")
            value = method(self, *args, **kwargs)
            cache[key] = (time.time(), value)
            
            return value
        
        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_size = lambda: len(cache)
        
        return wrapper
    return decorator


# Context managers for lazy loading
@contextmanager
def lazy_loading_context(auto_unload: bool = True):
    """Context manager for lazy loading operations."""
    manager = LazyDataManager()
    
    try:
        yield manager
    finally:
        if auto_unload:
            manager.unload_all()


@contextmanager
def memory_aware_loading(memory_limit_mb: float = 100.0):
    """Context manager with memory-aware lazy loading."""
    config = LazyLoadConfig(auto_unload_threshold_mb=memory_limit_mb)
    manager = LazyDataManager(config)
    
    try:
        yield manager
    finally:
        manager.unload_all()


# Global instances for convenience
_global_nfl_dataset = None

def get_nfl_dataset() -> LazyNFLDataset:
    """Get global NFL dataset instance."""
    global _global_nfl_dataset
    if _global_nfl_dataset is None:
        _global_nfl_dataset = LazyNFLDataset()
    return _global_nfl_dataset


# Example usage classes
class LazyPowerRankingsCalculator:
    """Power rankings calculator with lazy loading."""
    
    def __init__(self):
        self.dataset = get_nfl_dataset()
        
    @lazy_property
    def team_mappings(self):
        """Lazy-loaded team ID mappings."""
        teams = self.dataset.teams
        return {team['team']['id']: team['team']['displayName'] for team in teams}
    
    @lazy_method(ttl_seconds=600)  # Cache for 10 minutes
    def calculate_rankings(self, week: int):
        """Calculate power rankings for specific week with caching."""
        # Implementation would go here
        return {'week': week, 'rankings': []}
    
    def get_memory_info(self):
        """Get memory usage information."""
        return {
            'dataset_memory': self.dataset.get_memory_usage(),
            'calculator_memory': {
                'team_mappings_loaded': hasattr(self.team_mappings, 'loaded')
            }
        }


# Performance monitoring for lazy loading
class LazyLoadingProfiler:
    """Profiler for lazy loading performance."""
    
    def __init__(self):
        self.load_times = defaultdict(list)
        self.access_counts = defaultdict(int)
        self.memory_usage = defaultdict(list)
    
    def record_load(self, key: str, load_time: float, memory_mb: float):
        """Record a lazy load operation."""
        self.load_times[key].append(load_time)
        self.memory_usage[key].append(memory_mb)
        self.access_counts[key] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {}
        
        for key in self.load_times.keys():
            load_times = self.load_times[key]
            memory_usage = self.memory_usage[key]
            
            report[key] = {
                'access_count': self.access_counts[key],
                'avg_load_time': sum(load_times) / len(load_times),
                'max_load_time': max(load_times),
                'min_load_time': min(load_times),
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'max_memory_mb': max(memory_usage)
            }
        
        return report


# Global profiler instance
_lazy_loading_profiler = LazyLoadingProfiler()

def get_lazy_loading_profiler() -> LazyLoadingProfiler:
    """Get global lazy loading profiler."""
    return _lazy_loading_profiler