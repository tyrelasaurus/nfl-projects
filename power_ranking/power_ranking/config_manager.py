"""
Configuration Management System for Power Rankings
Handles loading, validation, and environment-specific configurations.
"""

import yaml
import os
import sys
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Import from power_ranking exceptions
from .exceptions import ConfigurationError
from .config_utils import find_config_file, deep_merge, validate_logging_level

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """ESPN API configuration settings."""
    base_url: str = "https://site.api.espn.com/apis/site/v2"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    rate_limit_requests_per_second: float = 2.0

@dataclass  
class ModelWeights:
    """Power ranking model weight configuration."""
    season_avg_margin: float = 0.5
    rolling_avg_margin: float = 0.25
    sos: float = 0.2
    recency_factor: float = 0.05

@dataclass
class ModelConfig:
    """Power ranking model configuration."""
    weights: ModelWeights = field(default_factory=ModelWeights)
    rolling_window: int = 5
    week18_weight: float = 0.3
    confidence: Dict[str, Any] = field(default_factory=lambda: {
        'default_level': 0.95,
        'min_sample_size': 4,
        'stability_window': 3
    })

@dataclass
class ValidationConfig:
    """Data validation configuration."""
    data_quality: Dict[str, Any] = field(default_factory=dict)
    anomaly_detection: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutputConfig:
    """Output configuration settings."""
    directory: str = "./output"
    file_formats: Dict[str, bool] = field(default_factory=lambda: {
        'csv': True, 'json': True, 'excel': False
    })
    csv_options: Dict[str, Any] = field(default_factory=lambda: {
        'delimiter': ',', 'encoding': 'utf-8', 'include_index': False
    })
    naming_convention: Dict[str, Any] = field(default_factory=lambda: {
        'include_timestamp': True, 'timestamp_format': '%Y%m%d_%H%M%S'
    })

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False, 'filename': 'power_rankings.log',
        'max_size_mb': 50, 'backup_count': 5
    })

@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    parallel_processing: bool = False
    max_workers: int = 4
    memory_limit_mb: int = 500

@dataclass
class PowerRankingConfig:
    """Complete power ranking configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    environment: str = "development"

class ConfigManager:
    """
    Manages loading and validation of power ranking configurations.
    Supports environment-specific overrides and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (defaults to config_enhanced.yaml)
            environment: Environment name for overrides (development/production/testing)
        """
        self.config_path = config_path or self._find_config_file()
        self.environment = environment or os.getenv('POWER_RANKING_ENV', 'development')
        self._config: Optional[PowerRankingConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        base_dir = os.path.dirname(__file__)
        package_root = os.path.abspath(os.path.join(base_dir))
        possible_paths = [
            'config_enhanced.yaml',
            'config.yaml',
            os.path.join(package_root, 'config_enhanced.yaml'),
            os.path.join(package_root, 'config.yaml'),
            # Also check project-level package directory for config.yaml
            os.path.join(os.path.dirname(package_root), 'config.yaml')
        ]
        try:
            return find_config_file(possible_paths)
        except FileNotFoundError:
            raise ConfigurationError(
            "No configuration file found",
            context={
                'searched_paths': possible_paths,
                'current_directory': os.getcwd()
            },
            recovery_suggestions=[
                "Create a config_enhanced.yaml file in the project root",
                "Set the config_path parameter explicitly",
                "Check file permissions and paths"
            ]
        )
    
    def load_config(self, reload: bool = False) -> PowerRankingConfig:
        """
        Load and validate configuration.
        
        Args:
            reload: Force reload even if already loaded
            
        Returns:
            Validated configuration object
        """
        if self._config is not None and not reload:
            return self._config
            
        try:
            logger.info(f"Loading configuration from: {self.config_path}")
            
            with open(self.config_path, 'r') as file:
                self._raw_config = yaml.safe_load(file)
                
            if not isinstance(self._raw_config, dict):
                raise ConfigurationError(
                    "Configuration file must contain a YAML dictionary",
                    context={'config_path': self.config_path}
                )
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
            # Build typed configuration
            self._config = self._build_typed_config()
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            return self._config
            
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}",
                context={'config_path': self.config_path},
                recovery_suggestions=[
                    "Create the configuration file",
                    "Check the file path",
                    "Verify file permissions"
                ]
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {str(e)}",
                context={'config_path': self.config_path, 'yaml_error': str(e)},
                recovery_suggestions=[
                    "Check YAML syntax",
                    "Validate indentation",
                    "Remove invalid characters"
                ]
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                context={'config_path': self.config_path, 'error_type': type(e).__name__}
            )
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        if 'environments' not in self._raw_config:
            return
            
        env_config = self._raw_config['environments'].get(self.environment)
        if not env_config:
            logger.warning(f"No configuration found for environment: {self.environment}")
            return
        
        logger.debug(f"Applying {self.environment} environment overrides")
        deep_merge(self._raw_config, env_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        deep_merge(base, override)
    
    def _build_typed_config(self) -> PowerRankingConfig:
        """Build typed configuration from raw dictionary."""
        try:
            # Extract main sections
            api_data = self._raw_config.get('api', {})
            model_data = self._raw_config.get('model', {})
            validation_data = self._raw_config.get('validation', {})
            output_data = self._raw_config.get('output', {})
            logging_data = self._raw_config.get('logging', {})
            performance_data = self._raw_config.get('performance', {})
            
            # Build model weights
            weights_data = model_data.get('weights', {})
            weights = ModelWeights(
                season_avg_margin=weights_data.get('season_avg_margin', 0.5),
                rolling_avg_margin=weights_data.get('rolling_avg_margin', 0.25),
                sos=weights_data.get('sos', 0.2),
                recency_factor=weights_data.get('recency_factor', 0.05)
            )
            
            # Build configuration objects
            api_config = APIConfig(
                base_url=api_data.get('base_url', 'https://site.api.espn.com/apis/site/v2'),
                timeout_seconds=api_data.get('timeout_seconds', 30),
                max_retries=api_data.get('max_retries', 3),
                retry_delay_seconds=api_data.get('retry_delay_seconds', 1.0),
                rate_limit_requests_per_second=api_data.get('rate_limit_requests_per_second', 2.0)
            )
            
            model_config = ModelConfig(
                weights=weights,
                rolling_window=model_data.get('rolling_window', 5),
                week18_weight=model_data.get('week18_weight', 0.3),
                confidence=model_data.get('confidence', {
                    'default_level': 0.95,
                    'min_sample_size': 4,
                    'stability_window': 3
                })
            )
            
            validation_config = ValidationConfig(
                data_quality=validation_data.get('data_quality', {}),
                anomaly_detection=validation_data.get('anomaly_detection', {})
            )
            
            output_config = OutputConfig(
                directory=output_data.get('directory', './output'),
                file_formats=output_data.get('file_formats', {'csv': True, 'json': True, 'excel': False}),
                csv_options=output_data.get('csv_options', {'delimiter': ',', 'encoding': 'utf-8', 'include_index': False}),
                naming_convention=output_data.get('naming_convention', {'include_timestamp': True, 'timestamp_format': '%Y%m%d_%H%M%S'})
            )
            
            logging_config = LoggingConfig(
                level=logging_data.get('level', 'INFO'),
                format=logging_data.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                file_logging=logging_data.get('file_logging', {'enabled': False, 'filename': 'power_rankings.log', 'max_size_mb': 50, 'backup_count': 5})
            )
            
            performance_config = PerformanceConfig(
                enable_caching=performance_data.get('enable_caching', True),
                cache_ttl_seconds=performance_data.get('cache_ttl_seconds', 3600),
                parallel_processing=performance_data.get('parallel_processing', False),
                max_workers=performance_data.get('max_workers', 4),
                memory_limit_mb=performance_data.get('memory_limit_mb', 500)
            )
            
            return PowerRankingConfig(
                api=api_config,
                model=model_config,
                validation=validation_config,
                output=output_config,
                logging=logging_config,
                performance=performance_config,
                environment=self.environment
            )
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to build typed configuration: {str(e)}",
                context={'error_type': type(e).__name__, 'config_keys': list(self._raw_config.keys())},
                recovery_suggestions=[
                    "Check configuration file structure",
                    "Ensure all required sections are present",
                    "Validate data types in configuration"
                ]
            )
    
    def _validate_config(self) -> None:
        """Validate configuration values and constraints."""
        if not self._config:
            raise ConfigurationError("No configuration to validate")
        
        errors = []
        
        # Validate model weights sum to reasonable value
        weights = self._config.model.weights
        total_weight = (weights.season_avg_margin + weights.rolling_avg_margin + 
                       weights.sos + weights.recency_factor)
        if not (0.9 <= total_weight <= 1.1):
            errors.append(f"Model weights sum to {total_weight:.3f}, expected ~1.0")
        
        # Validate positive values
        if self._config.model.rolling_window <= 0:
            errors.append(f"Rolling window must be positive, got {self._config.model.rolling_window}")
            
        if not (0.0 <= self._config.model.week18_weight <= 1.0):
            errors.append(f"Week 18 weight must be between 0 and 1, got {self._config.model.week18_weight}")
        
        # Validate API configuration
        if self._config.api.timeout_seconds <= 0:
            errors.append(f"API timeout must be positive, got {self._config.api.timeout_seconds}")
            
        if self._config.api.max_retries < 0:
            errors.append(f"Max retries cannot be negative, got {self._config.api.max_retries}")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config.logging.level.upper() not in valid_levels:
            errors.append(f"Invalid logging level: {self._config.logging.level}")
        
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {len(errors)} error(s)",
                context={'validation_errors': errors},
                recovery_suggestions=[
                    "Fix the configuration errors listed",
                    "Check the configuration file syntax",
                    "Refer to configuration documentation"
                ]
            )
        
        logger.debug("Configuration validation passed")
    
    def get_config(self) -> PowerRankingConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get_raw_config(self) -> Optional[Dict[str, Any]]:
        """Get the raw configuration dictionary."""
        return self._raw_config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if not self._raw_config:
            raise ConfigurationError("No configuration loaded to update")
        
        logger.info("Updating configuration with new values")
        self._deep_merge(self._raw_config, updates)
        
        # Rebuild typed configuration
        self._config = self._build_typed_config()
        self._validate_config()
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Output path (defaults to current config path)
        """
        if not self._raw_config:
            raise ConfigurationError("No configuration to save")
        
        output_path = path or self.config_path
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self._raw_config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {str(e)}",
                context={'output_path': output_path}
            )

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None, environment: Optional[str] = None) -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path, environment)
    
    return _config_manager

def get_config() -> PowerRankingConfig:
    """Get the current configuration."""
    return get_config_manager().get_config()
