"""
Data streaming utilities for processing large datasets efficiently.
Provides streaming, chunking, and batching capabilities to minimize memory usage.
"""

import csv
import json
import logging
from typing import Iterator, List, Dict, Any, Optional, Callable, Union, TextIO
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from contextlib import contextmanager
import gc
import sys

from .memory_monitor import get_global_monitor

logger = logging.getLogger(__name__)


@dataclass 
class StreamingConfig:
    """Configuration for data streaming operations."""
    chunk_size: int = 1000
    memory_limit_mb: float = 100.0
    enable_gc_per_chunk: bool = True
    enable_memory_monitoring: bool = True
    buffer_size: int = 8192


class DataStreamProcessor:
    """Efficient data streaming processor for large datasets."""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = get_global_monitor() if self.config.enable_memory_monitoring else None
        
    @contextmanager
    def memory_managed_processing(self, operation_name: str):
        """Context manager for memory-managed data processing."""
        if self.memory_monitor:
            with self.memory_monitor.profile_memory(operation_name):
                yield
        else:
            yield
        
        # Force GC if enabled
        if self.config.enable_gc_per_chunk:
            gc.collect()
    
    def stream_csv_file(self, file_path: Union[str, Path], 
                       field_names: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Stream CSV file row by row to minimize memory usage."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Streaming CSV file: {file_path}")
        
        try:
            with open(file_path, 'r', newline='', buffering=self.config.buffer_size) as csvfile:
                # Detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                # Create reader
                reader = csv.DictReader(csvfile, fieldnames=field_names, delimiter=delimiter)
                
                row_count = 0
                for row in reader:
                    yield row
                    row_count += 1
                    
                    # Memory management
                    if row_count % self.config.chunk_size == 0:
                        if self.config.enable_gc_per_chunk:
                            gc.collect()
                        
                        # Check memory usage
                        if self.memory_monitor:
                            current_memory = self.memory_monitor.get_current_memory()
                            if current_memory.rss_mb > self.config.memory_limit_mb:
                                logger.warning(f"Memory usage high: {current_memory.rss_mb:.1f}MB")
                
                logger.info(f"Streamed {row_count} rows from {file_path}")
                
        except Exception as e:
            logger.error(f"Error streaming CSV file {file_path}: {e}")
            raise
    
    def stream_json_file(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Stream JSON file for large JSON arrays."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        logger.info(f"Streaming JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', buffering=self.config.buffer_size) as jsonfile:
                # For large JSON arrays, we need to parse incrementally
                # This is a simplified implementation - for very large files,
                # consider using ijson library
                data = json.load(jsonfile)
                
                if isinstance(data, list):
                    row_count = 0
                    for item in data:
                        yield item
                        row_count += 1
                        
                        if row_count % self.config.chunk_size == 0:
                            if self.config.enable_gc_per_chunk:
                                gc.collect()
                elif isinstance(data, dict):
                    yield data
                else:
                    raise ValueError(f"Unsupported JSON structure: {type(data)}")
                
                logger.info(f"Streamed JSON file: {file_path}")
                
        except Exception as e:
            logger.error(f"Error streaming JSON file {file_path}: {e}")
            raise
    
    def process_in_chunks(self, data_iterator: Iterator[Any], 
                         processor: Callable[[List[Any]], Any],
                         chunk_size: Optional[int] = None) -> Iterator[Any]:
        """Process data in chunks to control memory usage."""
        chunk_size = chunk_size or self.config.chunk_size
        
        chunk = []
        chunk_count = 0
        
        for item in data_iterator:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                chunk_count += 1
                
                with self.memory_managed_processing(f"chunk_processing_{chunk_count}"):
                    result = processor(chunk)
                    yield result
                
                # Clear chunk and collect garbage
                chunk.clear()
                
                # Memory monitoring
                if self.memory_monitor and chunk_count % 10 == 0:
                    stats = self.memory_monitor.get_memory_stats()
                    current_mb = stats['current_memory']['rss_mb']
                    logger.debug(f"Processed {chunk_count} chunks, memory: {current_mb:.1f}MB")
        
        # Process remaining items in final chunk
        if chunk:
            chunk_count += 1
            with self.memory_managed_processing(f"final_chunk_{chunk_count}"):
                result = processor(chunk)
                yield result
        
        logger.info(f"Completed chunked processing: {chunk_count} chunks")
    
    def batch_process_games(self, games: List[Dict[str, Any]], 
                           batch_processor: Callable[[List[Dict]], Any]) -> Iterator[Any]:
        """Process NFL games data in memory-efficient batches."""
        logger.info(f"Processing {len(games)} games in batches of {self.config.chunk_size}")
        
        def game_iterator():
            for game in games:
                yield game
        
        return self.process_in_chunks(game_iterator(), batch_processor)
    
    def stream_pandas_dataframe(self, df: pd.DataFrame, 
                               chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Stream pandas DataFrame in chunks."""
        chunk_size = chunk_size or self.config.chunk_size
        
        logger.info(f"Streaming DataFrame with {len(df)} rows in chunks of {chunk_size}")
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            
            with self.memory_managed_processing(f"df_chunk_{i//chunk_size + 1}"):
                yield chunk
            
            # Memory management
            del chunk
            if self.config.enable_gc_per_chunk:
                gc.collect()
    
    def efficient_csv_writer(self, file_path: Union[str, Path], 
                           fieldnames: List[str]) -> 'EfficientCSVWriter':
        """Create an efficient CSV writer for streaming output."""
        return EfficientCSVWriter(file_path, fieldnames, self.config)
    
    def process_large_dataset(self, data_source: Union[str, Path, List, Iterator],
                             processor: Callable[[Any], Any],
                             output_handler: Optional[Callable[[Any], None]] = None) -> Dict[str, Any]:
        """Process a large dataset efficiently with memory monitoring."""
        
        stats = {
            'items_processed': 0,
            'chunks_processed': 0,
            'memory_peak_mb': 0.0,
            'processing_time_seconds': 0.0,
            'gc_collections': 0
        }
        
        # Determine data source type and create iterator
        if isinstance(data_source, (str, Path)):
            file_path = Path(data_source)
            if file_path.suffix.lower() == '.csv':
                data_iter = self.stream_csv_file(file_path)
            elif file_path.suffix.lower() == '.json':
                data_iter = self.stream_json_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        elif isinstance(data_source, list):
            data_iter = iter(data_source)
        else:
            data_iter = data_source
        
        # Process with memory monitoring
        with self.memory_managed_processing("large_dataset_processing"):
            import time
            start_time = time.time()
            
            for chunk_result in self.process_in_chunks(data_iter, processor):
                stats['chunks_processed'] += 1
                
                if output_handler:
                    output_handler(chunk_result)
                
                # Update memory peak
                if self.memory_monitor:
                    current_memory = self.memory_monitor.get_current_memory()
                    stats['memory_peak_mb'] = max(stats['memory_peak_mb'], current_memory.rss_mb)
            
            stats['processing_time_seconds'] = time.time() - start_time
        
        logger.info(f"Large dataset processing complete: {stats}")
        return stats


class EfficientCSVWriter:
    """Memory-efficient CSV writer for streaming output."""
    
    def __init__(self, file_path: Union[str, Path], fieldnames: List[str], 
                 config: StreamingConfig):
        self.file_path = Path(file_path)
        self.fieldnames = fieldnames
        self.config = config
        self.file_handle: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None
        self.rows_written = 0
    
    def __enter__(self):
        self.file_handle = open(self.file_path, 'w', newline='', 
                               buffering=self.config.buffer_size)
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
        logger.info(f"CSV writer closed: {self.rows_written} rows written to {self.file_path}")
    
    def write_row(self, row: Dict[str, Any]) -> None:
        """Write a single row."""
        if not self.writer:
            raise RuntimeError("CSV writer not initialized")
        
        self.writer.writerow(row)
        self.rows_written += 1
        
        # Periodic flush and GC
        if self.rows_written % self.config.chunk_size == 0:
            self.file_handle.flush()
            if self.config.enable_gc_per_chunk:
                gc.collect()
    
    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Write multiple rows efficiently."""
        if not self.writer:
            raise RuntimeError("CSV writer not initialized")
        
        self.writer.writerows(rows)
        self.rows_written += len(rows)
        
        # Flush and GC
        self.file_handle.flush()
        if self.config.enable_gc_per_chunk:
            gc.collect()


class LazyDataLoader:
    """Lazy loading data loader for efficient memory usage."""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self._cached_data = {}
        self._data_loaders = {}
    
    def register_loader(self, key: str, loader: Callable[[], Any]) -> None:
        """Register a data loader function."""
        self._data_loaders[key] = loader
        logger.debug(f"Registered lazy loader: {key}")
    
    def get_data(self, key: str, force_reload: bool = False) -> Any:
        """Get data, loading lazily if not already loaded."""
        if force_reload or key not in self._cached_data:
            if key not in self._data_loaders:
                raise KeyError(f"No loader registered for key: {key}")
            
            logger.info(f"Lazy loading data: {key}")
            
            # Load data with memory monitoring
            monitor = get_global_monitor()
            with monitor.profile_memory(f"lazy_load_{key}"):
                self._cached_data[key] = self._data_loaders[key]()
            
            # Memory management
            if self.config.enable_gc_per_chunk:
                gc.collect()
        
        return self._cached_data[key]
    
    def unload_data(self, key: str) -> None:
        """Unload data from cache to free memory."""
        if key in self._cached_data:
            del self._cached_data[key]
            gc.collect()
            logger.info(f"Unloaded data: {key}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        cache_size = len(self._cached_data)
        self._cached_data.clear()
        gc.collect()
        logger.info(f"Cleared lazy loader cache: {cache_size} items")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            'cached_items': list(self._cached_data.keys()),
            'cache_size': len(self._cached_data),
            'registered_loaders': list(self._data_loaders.keys()),
            'memory_estimate_mb': sum(
                sys.getsizeof(data) / 1024 / 1024 
                for data in self._cached_data.values()
            )
        }


# Convenience functions for common streaming operations

def stream_csv(file_path: Union[str, Path], chunk_size: int = 1000) -> Iterator[Dict[str, Any]]:
    """Stream CSV file with default configuration."""
    config = StreamingConfig(chunk_size=chunk_size)
    processor = DataStreamProcessor(config)
    return processor.stream_csv_file(file_path)


def stream_json(file_path: Union[str, Path], chunk_size: int = 1000) -> Iterator[Dict[str, Any]]:
    """Stream JSON file with default configuration."""
    config = StreamingConfig(chunk_size=chunk_size)
    processor = DataStreamProcessor(config)
    return processor.stream_json_file(file_path)


def process_games_in_batches(games: List[Dict[str, Any]], 
                           processor: Callable[[List[Dict]], Any],
                           chunk_size: int = 1000) -> Iterator[Any]:
    """Process NFL games in memory-efficient batches."""
    config = StreamingConfig(chunk_size=chunk_size)
    stream_processor = DataStreamProcessor(config)
    return stream_processor.batch_process_games(games, processor)


# Global lazy loader instance
_global_lazy_loader = LazyDataLoader()

def get_lazy_loader() -> LazyDataLoader:
    """Get global lazy data loader."""
    return _global_lazy_loader