"""
Memory-optimized data structures for NFL Power Rankings system.
Provides efficient alternatives to standard Python data structures.
"""

import sys
import array
import struct
import weakref
import threading
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompactGameRecord:
    """Memory-optimized NFL game record using __slots__."""
    __slots__ = ('game_id', 'week', 'season', 'home_team_id', 'away_team_id', 
                 'home_score', 'away_score', 'status_code', '_date_timestamp', '__weakref__')
    
    # Status codes for memory efficiency
    STATUS_CODES = {
        'scheduled': 0,
        'in_progress': 1, 
        'completed': 2,
        'postponed': 3,
        'cancelled': 4
    }
    
    STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
    
    def __init__(self, game_id: str, week: int, season: int, 
                 home_team_id: int, away_team_id: int,
                 home_score: Optional[int] = None, away_score: Optional[int] = None,
                 status: str = 'scheduled', date_timestamp: Optional[float] = None):
        self.game_id = game_id
        self.week = week
        self.season = season
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.home_score = home_score or 0
        self.away_score = away_score or 0
        self.status_code = self.STATUS_CODES.get(status, 0)
        self._date_timestamp = date_timestamp or 0.0
    
    @property
    def status(self) -> str:
        return self.STATUS_NAMES.get(self.status_code, 'unknown')
    
    @property
    def margin(self) -> int:
        """Calculate margin (home team perspective)."""
        return self.home_score - self.away_score
    
    @property
    def total_points(self) -> int:
        """Calculate total points in the game."""
        return self.home_score + self.away_score
    
    def __sizeof__(self) -> int:
        """Return memory size in bytes."""
        # Much smaller than regular dict due to __slots__
        return object.__sizeof__(self) + sum(
            sys.getsizeof(getattr(self, slot)) for slot in self.__slots__
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            'game_id': self.game_id,
            'week': self.week,
            'season': self.season,
            'home_team_id': self.home_team_id,
            'away_team_id': self.away_team_id,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'status': self.status,
            'date_timestamp': self._date_timestamp,
            'margin': self.margin,
            'total_points': self.total_points
        }


class CompactTeamRanking:
    """Memory-optimized team ranking record."""
    __slots__ = ('team_id', 'power_score_int', 'rank', 'wins', 'losses', 'ties',
                 'season_margin_int', 'rolling_margin_int', 'sos_int', '__weakref__')
    
    # Scale factors for fixed-point arithmetic
    SCORE_SCALE = 100  # Power score precision: 0.01
    MARGIN_SCALE = 100  # Margin precision: 0.01
    SOS_SCALE = 10000  # SOS precision: 0.0001
    
    def __init__(self, team_id: int, power_score: float, rank: int,
                 wins: int = 0, losses: int = 0, ties: int = 0,
                 season_avg_margin: Optional[float] = None,
                 rolling_avg_margin: Optional[float] = None,
                 strength_of_schedule: Optional[float] = None):
        self.team_id = team_id
        self.power_score_int = int(power_score * self.SCORE_SCALE)
        self.rank = rank
        self.wins = wins
        self.losses = losses
        self.ties = ties
        self.season_margin_int = int((season_avg_margin or 0) * self.MARGIN_SCALE)
        self.rolling_margin_int = int((rolling_avg_margin or 0) * self.MARGIN_SCALE)
        self.sos_int = int((strength_of_schedule or 0) * self.SOS_SCALE)
    
    @property
    def power_score(self) -> float:
        return self.power_score_int / self.SCORE_SCALE
    
    @property
    def season_avg_margin(self) -> float:
        return self.season_margin_int / self.MARGIN_SCALE
    
    @property
    def rolling_avg_margin(self) -> float:
        return self.rolling_margin_int / self.MARGIN_SCALE
    
    @property
    def strength_of_schedule(self) -> float:
        return self.sos_int / self.SOS_SCALE
    
    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.ties
    
    def __sizeof__(self) -> int:
        """Return memory size in bytes."""
        return object.__sizeof__(self) + sum(
            sys.getsizeof(getattr(self, slot)) for slot in self.__slots__
        )


class CompactArray:
    """Memory-efficient array for numeric data using array.array."""
    
    def __init__(self, typecode: str = 'f', initial_data: Optional[List] = None):
        """
        Initialize compact array.
        
        Args:
            typecode: Array type ('f' for float, 'i' for int, 'd' for double)
            initial_data: Initial data to populate array
        """
        self._array = array.array(typecode, initial_data or [])
        self.typecode = typecode
    
    def append(self, value: Union[int, float]) -> None:
        """Append value to array."""
        self._array.append(value)
    
    def extend(self, values: List[Union[int, float]]) -> None:
        """Extend array with multiple values."""
        self._array.extend(values)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, float, 'CompactArray']:
        if isinstance(index, slice):
            # Return new CompactArray for slice
            return CompactArray(self.typecode, self._array[index])
        return self._array[index]
    
    def __setitem__(self, index: int, value: Union[int, float]) -> None:
        self._array[index] = value
    
    def __len__(self) -> int:
        return len(self._array)
    
    def __iter__(self) -> Iterator[Union[int, float]]:
        return iter(self._array)
    
    def to_list(self) -> List[Union[int, float]]:
        """Convert to regular Python list."""
        return self._array.tolist()
    
    def __sizeof__(self) -> int:
        """Return memory size in bytes."""
        return object.__sizeof__(self) + sys.getsizeof(self._array)
    
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.__sizeof__() / 1024 / 1024


class ObjectPool:
    """Generic object pool for reducing allocation overhead."""
    
    def __init__(self, factory: callable, max_size: int = 1000, 
                 reset_func: Optional[callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self._pool = deque()
        self._in_use = weakref.WeakSet()
        self._lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire object from pool."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
            else:
                obj = self.factory()
            
            self._in_use.add(obj)
            return obj
    
    def release(self, obj: Any) -> None:
        """Release object back to pool."""
        with self._lock:
            if obj in self._in_use:
                # Reset object state if reset function provided
                if self.reset_func:
                    self.reset_func(obj)
                
                # Add back to pool if not at max capacity
                if len(self._pool) < self.max_size:
                    self._pool.append(obj)
                
                self._in_use.discard(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'max_size': self.max_size
            }


class MemoryEfficientCache:
    """Memory-efficient LRU cache with automatic cleanup."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 50.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._cache = {}
        self._access_order = deque()
        self._lock = threading.RLock()
        self._current_memory_mb = 0.0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]['value']
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            item_size_mb = sys.getsizeof(value) / 1024 / 1024
            
            # Remove old value if updating
            if key in self._cache:
                old_size_mb = self._cache[key]['size_mb']
                self._current_memory_mb -= old_size_mb
                self._access_order.remove(key)
            
            # Check memory limit
            while (self._current_memory_mb + item_size_mb > self.max_memory_mb 
                   and self._access_order):
                self._evict_lru()
            
            # Check size limit
            while len(self._cache) >= self.max_size and self._access_order:
                self._evict_lru()
            
            # Add new item
            self._cache[key] = {'value': value, 'size_mb': item_size_mb}
            self._access_order.append(key)
            self._current_memory_mb += item_size_mb
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                self._current_memory_mb -= self._cache[lru_key]['size_mb']
                del self._cache[lru_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_mb': self._current_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'utilization_pct': len(self._cache) / self.max_size * 100,
                'memory_utilization_pct': self._current_memory_mb / self.max_memory_mb * 100
            }


class MemoryOptimizedDataFrame:
    """Memory-optimized alternative to pandas DataFrame for NFL data."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._data = {col: [] for col in columns}
        self._index = 0
        self._dtypes = {}
        self._optimized_arrays = {}
    
    def add_row(self, row_data: Dict[str, Any]) -> None:
        """Add a row to the DataFrame."""
        for col in self.columns:
            value = row_data.get(col)
            self._data[col].append(value)
        self._index += 1
    
    def optimize_memory(self) -> Dict[str, str]:
        """Optimize memory usage by converting to appropriate array types."""
        optimizations = {}
        
        for col in self.columns:
            values = self._data[col]
            if not values:
                continue
            
            # Determine optimal type
            if all(isinstance(v, (int, float)) and v is not None for v in values):
                # Check if all values are integers
                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) 
                       for v in values if v is not None):
                    # Use integer array
                    int_values = [int(v) if v is not None else 0 for v in values]
                    self._optimized_arrays[col] = CompactArray('i', int_values)
                    optimizations[col] = 'int_array'
                else:
                    # Use float array
                    float_values = [float(v) if v is not None else 0.0 for v in values]
                    self._optimized_arrays[col] = CompactArray('f', float_values)
                    optimizations[col] = 'float_array'
            else:
                # Keep as list for mixed types
                optimizations[col] = 'list'
        
        return optimizations
    
    def get_column(self, col: str) -> Union[List, CompactArray]:
        """Get column data."""
        if col in self._optimized_arrays:
            return self._optimized_arrays[col]
        return self._data[col]
    
    def get_row(self, index: int) -> Dict[str, Any]:
        """Get row data by index."""
        row = {}
        for col in self.columns:
            if col in self._optimized_arrays:
                row[col] = self._optimized_arrays[col][index]
            else:
                row[col] = self._data[col][index]
        return row
    
    def __len__(self) -> int:
        """Return number of rows."""
        return self._index
    
    def memory_usage(self) -> Dict[str, float]:
        """Get memory usage by column in MB."""
        usage = {}
        for col in self.columns:
            if col in self._optimized_arrays:
                usage[col] = self._optimized_arrays[col].memory_usage_mb()
            else:
                usage[col] = sys.getsizeof(self._data[col]) / 1024 / 1024
        
        usage['total'] = sum(usage.values())
        return usage
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary format."""
        result = {}
        for col in self.columns:
            if col in self._optimized_arrays:
                result[col] = self._optimized_arrays[col].to_list()
            else:
                result[col] = self._data[col].copy()
        return result


class BatchProcessor:
    """Memory-efficient batch processor for large datasets."""
    
    def __init__(self, batch_size: int = 1000, memory_limit_mb: float = 100.0):
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.current_batch = []
        self.processed_count = 0
        self.memory_usage_mb = 0.0
    
    def add_item(self, item: Any, processor: callable) -> Optional[Any]:
        """Add item to batch and process when batch is full."""
        self.current_batch.append(item)
        self.memory_usage_mb += sys.getsizeof(item) / 1024 / 1024
        
        # Process batch if size or memory limit reached
        if (len(self.current_batch) >= self.batch_size or 
            self.memory_usage_mb >= self.memory_limit_mb):
            return self._process_batch(processor)
        
        return None
    
    def _process_batch(self, processor: callable) -> Any:
        """Process current batch."""
        if not self.current_batch:
            return None
        
        result = processor(self.current_batch)
        self.processed_count += len(self.current_batch)
        
        # Clear batch and reset memory tracking
        self.current_batch.clear()
        self.memory_usage_mb = 0.0
        
        return result
    
    def flush(self, processor: callable) -> Optional[Any]:
        """Process any remaining items in batch."""
        return self._process_batch(processor)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_count': self.processed_count,
            'current_batch_size': len(self.current_batch),
            'current_memory_mb': self.memory_usage_mb,
            'batch_size_limit': self.batch_size,
            'memory_limit_mb': self.memory_limit_mb
        }


def compare_memory_usage() -> Dict[str, Dict[str, float]]:
    """Compare memory usage between standard and optimized structures."""
    results = {}
    
    # Compare game records
    standard_games = []
    compact_games = []
    
    for i in range(1000):
        # Standard dict
        standard_game = {
            'game_id': f'game_{i}',
            'week': i % 18 + 1,
            'season': 2024,
            'home_team_id': i % 32,
            'away_team_id': (i + 1) % 32,
            'home_score': 24,
            'away_score': 21,
            'status': 'completed'
        }
        standard_games.append(standard_game)
        
        # Compact record
        compact_game = CompactGameRecord(
            f'game_{i}', i % 18 + 1, 2024,
            i % 32, (i + 1) % 32, 24, 21, 'completed'
        )
        compact_games.append(compact_game)
    
    results['game_records'] = {
        'standard_mb': sys.getsizeof(standard_games) / 1024 / 1024,
        'compact_mb': sys.getsizeof(compact_games) / 1024 / 1024,
        'memory_saving_pct': 0.0
    }
    
    # Calculate savings
    standard_size = results['game_records']['standard_mb']
    compact_size = results['game_records']['compact_mb']
    if standard_size > 0:
        savings = ((standard_size - compact_size) / standard_size) * 100
        results['game_records']['memory_saving_pct'] = savings
    
    # Compare arrays
    standard_list = list(range(10000))
    compact_array = CompactArray('i', list(range(10000)))
    
    results['numeric_arrays'] = {
        'standard_mb': sys.getsizeof(standard_list) / 1024 / 1024,
        'compact_mb': compact_array.memory_usage_mb(),
        'memory_saving_pct': 0.0
    }
    
    # Calculate array savings
    standard_size = results['numeric_arrays']['standard_mb']
    compact_size = results['numeric_arrays']['compact_mb']
    if standard_size > 0:
        savings = ((standard_size - compact_size) / standard_size) * 100
        results['numeric_arrays']['memory_saving_pct'] = savings
    
    return results


# Global instances for common use cases
game_record_pool = ObjectPool(
    factory=lambda: CompactGameRecord('', 0, 0, 0, 0),
    reset_func=lambda obj: setattr(obj, 'game_id', '')
)

team_ranking_cache = MemoryEfficientCache(max_size=500, max_memory_mb=10.0)