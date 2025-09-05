"""
Intelligent caching layer for ESPN API responses.
Provides persistent caching with TTL and intelligent invalidation.
"""

import json
import os
import time
import logging
import hashlib
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """Intelligent cache manager for API responses."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 300):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache configuration by endpoint type
        self.ttl_config = {
            'teams': 86400,  # Team data changes rarely (24 hours)
            'scoreboard': 300,  # Scoreboards change frequently (5 minutes)
            'current_week': 3600,  # Current week changes weekly (1 hour)
            'season_data': 1800,  # Historical data is more stable (30 minutes)
            'default': default_ttl
        }
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate consistent cache key."""
        key_data = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_ttl_for_endpoint(self, endpoint: str) -> int:
        """Get appropriate TTL for endpoint type."""
        for key, ttl in self.ttl_config.items():
            if key in endpoint:
                return ttl
        return self.ttl_config['default']
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_key: str, custom_ttl: Optional[int] = None) -> bool:
        """Check if cache entry is valid."""
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_time, _, ttl = self.memory_cache[cache_key]
            return (time.time() - cached_time) < (custom_ttl or ttl)
        
        # Check file cache
        cache_file = self._get_cache_file_path(cache_key)
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cached_time = cache_data.get('timestamp', 0)
            ttl = cache_data.get('ttl', self.default_ttl)
            return (time.time() - cached_time) < (custom_ttl or ttl)
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            return False
    
    def get(self, endpoint: str, params: Optional[Dict] = None, 
            custom_ttl: Optional[int] = None) -> Optional[Any]:
        """Get cached data if valid."""
        cache_key = self._get_cache_key(endpoint, params)
        
        # Check memory cache first (faster)
        if cache_key in self.memory_cache:
            cached_time, data, ttl = self.memory_cache[cache_key]
            if (time.time() - cached_time) < (custom_ttl or ttl):
                self.cache_stats['hits'] += 1
                logger.debug(f"Memory cache hit for {endpoint}")
                return data
            else:
                # Expired, remove from memory
                del self.memory_cache[cache_key]
        
        # Check file cache
        if self._is_cache_valid(cache_key, custom_ttl):
            try:
                cache_file = self._get_cache_file_path(cache_key)
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                data = cache_data.get('data')
                ttl = cache_data.get('ttl', self.default_ttl)
                
                # Load into memory cache
                self.memory_cache[cache_key] = (time.time(), data, ttl)
                
                self.cache_stats['hits'] += 1
                logger.debug(f"File cache hit for {endpoint}")
                return data
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache file: {e}")
        
        self.cache_stats['misses'] += 1
        logger.debug(f"Cache miss for {endpoint}")
        return None
    
    def set(self, endpoint: str, data: Any, params: Optional[Dict] = None, 
            custom_ttl: Optional[int] = None) -> None:
        """Cache data with appropriate TTL."""
        cache_key = self._get_cache_key(endpoint, params)
        ttl = custom_ttl or self._get_ttl_for_endpoint(endpoint)
        current_time = time.time()
        
        # Store in memory cache
        self.memory_cache[cache_key] = (current_time, data, ttl)
        
        # Store in file cache for persistence
        try:
            cache_data = {
                'timestamp': current_time,
                'ttl': ttl,
                'endpoint': endpoint,
                'params': params,
                'data': data
            }
            
            cache_file = self._get_cache_file_path(cache_key)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.debug(f"Cached {endpoint} with TTL {ttl}s")
            
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to write cache file: {e}")
    
    def invalidate(self, endpoint: str, params: Optional[Dict] = None) -> None:
        """Invalidate specific cache entry."""
        cache_key = self._get_cache_key(endpoint, params)
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove file cache
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.debug(f"Invalidated cache for {endpoint}")
            except OSError as e:
                logger.warning(f"Failed to remove cache file: {e}")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching pattern."""
        count = 0
        
        # Invalidate memory cache
        keys_to_remove = []
        for key in self.memory_cache.keys():
            if pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_cache[key]
            count += 1
        
        # Invalidate file cache
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    endpoint = cache_data.get('endpoint', '')
                    if pattern in endpoint:
                        cache_file.unlink()
                        count += 1
                        
                except (json.JSONDecodeError, IOError):
                    continue
        except OSError as e:
            logger.warning(f"Failed to scan cache directory: {e}")
        
        logger.info(f"Invalidated {count} cache entries matching '{pattern}'")
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        count = 0
        current_time = time.time()
        
        # Cleanup memory cache
        expired_keys = []
        for key, (cached_time, _, ttl) in self.memory_cache.items():
            if (current_time - cached_time) >= ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            count += 1
            self.cache_stats['evictions'] += 1
        
        # Cleanup file cache
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    cached_time = cache_data.get('timestamp', 0)
                    ttl = cache_data.get('ttl', self.default_ttl)
                    
                    if (current_time - cached_time) >= ttl:
                        cache_file.unlink()
                        count += 1
                        self.cache_stats['evictions'] += 1
                        
                except (json.JSONDecodeError, IOError):
                    # Remove invalid cache files
                    try:
                        cache_file.unlink()
                        count += 1
                    except OSError:
                        pass
                        
        except OSError as e:
            logger.warning(f"Failed to cleanup cache directory: {e}")
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def clear_all(self) -> int:
        """Clear all cache entries."""
        count = 0
        
        # Clear memory cache
        count += len(self.memory_cache)
        self.memory_cache.clear()
        
        # Clear file cache
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except OSError:
                    pass
        except OSError as e:
            logger.warning(f"Failed to clear cache directory: {e}")
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        file_cache_count = 0
        try:
            file_cache_count = len(list(self.cache_dir.glob("*.json")))
        except OSError:
            pass
        
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / max(total_requests, 1)) * 100
        
        return {
            'memory_cache_entries': len(self.memory_cache),
            'file_cache_entries': file_cache_count,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_evictions': self.cache_stats['evictions'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_directory': str(self.cache_dir)
        }
    
    def get_size_info(self) -> Dict:
        """Get cache size information."""
        total_size = 0
        file_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    total_size += cache_file.stat().st_size
                    file_count += 1
                except OSError:
                    pass
        except OSError:
            pass
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_count': file_count,
            'average_file_size_bytes': round(total_size / max(file_count, 1), 2)
        }