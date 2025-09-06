"""
Configuration Management System for NFL Spread Model
Handles loading, validation, and environment-specific configurations.
"""

import yaml
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Add project root to path for exception imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exceptions import NFLModelError, ModelConfigurationError
from .config_utils import find_config_file, deep_merge, validate_logging_level

# Preserve legacy raise-sites that referenced ConfigurationError
ConfigurationError = ModelConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Spread model configuration."""
    home_field_advantage: float = 2.0
    calculation: Dict[str, Any] = field(default_factory=lambda: {
        'neutral_spread_precision': 1,
        'confidence_level': 0.95
    })
    validation: Dict[str, Any] = field(default_factory=lambda: {
        'tolerance_points': 3.0,
        'min_sample_size': 10,
        'autocorrelation_lag': 1
    })
    betting: Dict[str, Any] = field(default_factory=lambda: {
        'standard_odds': 1.91,
        'unit_size': 1.0,
        'max_bet_percentage': 0.05
    })

@dataclass
class MetricsConfig:
    """Performance metrics configuration."""
    rmse_target: float = 10.0
    accuracy_threshold: float = 0.55
    r_squared_threshold: float = 0.6
    sharpe_ratio_target: float = 1.0
    max_drawdown_limit: float = 0.2
    bootstrap_iterations: int = 1000
    confidence_interval: float = 0.95

@dataclass
class DataConfig:
    """Data loading and validation configuration."""
    default_paths: Dict[str, str] = field(default_factory=lambda: {
        'power_rankings': '../power_ranking/output/power_rankings_week_initial_adjusted.csv',
        'schedule': '../power_ranking/nfl_schedule_2025_20250831_220432.csv'
    })
    validation: Dict[str, Any] = field(default_factory=lambda: {
        'required_columns': {
            'power_rankings': ['team_name', 'power_score', 'rank'],
            'schedule': ['home_team', 'away_team', 'week', 'date']
        },
        'data_types': {
            'power_score': 'float',
            'week': 'int',
            'rank': 'int'
        },
        'value_ranges': {
            'week': [1, 22],
            'power_score': [-50.0, 50.0],
            'rank': [1, 32]
        }
    })

@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = "output"
    file_naming: Dict[str, str] = field(default_factory=lambda: {
        'spreads': 'nfl_spreads_week_{week}_{timestamp}.csv',
        'summary': 'spread_summary_week_{week}_{timestamp}.csv',
        'include_timestamp': True,
        'timestamp_format': '%Y%m%d_%H%M%S'
    })
    export_formats: Dict[str, bool] = field(default_factory=lambda: {
        'csv': True, 'json': False, 'excel': False
    })

@dataclass
class CLIConfig:
    """CLI configuration."""
    defaults: Dict[str, Any] = field(default_factory=lambda: {
        'verbose': False,
        'log_file': None,
        'no_fallback': False,
        'no_validation': False
    })
    validation: Dict[str, List[float]] = field(default_factory=lambda: {
        'week_range': [1, 22],
        'home_field_range': [-5.0, 10.0]
    })
    examples: Dict[str, str] = field(default_factory=lambda: {
        'basic': '--week 1',
        'custom_home_field': '--week 5 --home-field 2.5',
        'verbose': '--week 1 --verbose',
        'with_logging': '--week 1 --log-file nfl.log'
    })

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(levelname)s: %(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True, 'level': 'INFO'
    })
    file: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False, 'filename': 'nfl_model.log',
        'level': 'DEBUG', 'max_size_mb': 10, 'backup_count': 3
    })

@dataclass
class FeatureConfig:
    """Feature flags configuration."""
    enhanced_logging: bool = True
    data_validation: bool = True
    fallback_data: bool = True
    performance_monitoring: bool = False
    caching: bool = False

@dataclass
class NFLModelConfig:
    """Complete NFL model configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    environment: str = "development"

class NFLConfigManager:
    """
    Manages loading and validation of NFL model configurations.
    Supports environment-specific overrides and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (defaults to config.yaml)
            environment: Environment name for overrides (development/production/testing)
        """
        self.config_path = config_path or self._find_config_file()
        self.environment = environment or os.getenv('NFL_MODEL_ENV', 'development')
        self._config: Optional[NFLModelConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            'config.yaml',
            'config.yml', 
            os.path.join(os.path.dirname(__file__), 'config.yaml'),
            os.path.join(os.path.dirname(__file__), 'config.yml')
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
                "Create a config.yaml file in the project root",
                "Set the config_path parameter explicitly", 
                "Check file permissions and paths"
            ]
        )
    
    def load_config(self, reload: bool = False) -> NFLModelConfig:
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
            logger.info(f"Loading NFL model configuration from: {self.config_path}")
            
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
            
            logger.info(f"NFL model configuration loaded successfully for environment: {self.environment}")
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
                f"Failed to load NFL model configuration: {str(e)}",
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
        """Recursively merge override configuration into base."""
        deep_merge(base, override)
    
    def _build_typed_config(self) -> NFLModelConfig:
        """Build typed configuration from raw dictionary."""
        try:
            # Extract main sections
            model_data = self._raw_config.get('model', {})
            metrics_data = self._raw_config.get('metrics', {})
            data_data = self._raw_config.get('data', {})
            output_data = self._raw_config.get('output', {})
            cli_data = self._raw_config.get('cli', {})
            logging_data = self._raw_config.get('logging', {})
            features_data = self._raw_config.get('features', {})
            
            # Build configuration objects
            model_config = ModelConfig(
                home_field_advantage=model_data.get('home_field_advantage', 2.0),
                calculation=model_data.get('calculation', {}),
                validation=model_data.get('validation', {}),
                betting=model_data.get('betting', {})
            )
            
            metrics_config = MetricsConfig(
                rmse_target=metrics_data.get('rmse_target', 10.0),
                accuracy_threshold=metrics_data.get('accuracy_threshold', 0.55),
                r_squared_threshold=metrics_data.get('r_squared_threshold', 0.6),
                sharpe_ratio_target=metrics_data.get('sharpe_ratio_target', 1.0),
                max_drawdown_limit=metrics_data.get('max_drawdown_limit', 0.2),
                bootstrap_iterations=metrics_data.get('bootstrap_iterations', 1000),
                confidence_interval=metrics_data.get('confidence_interval', 0.95)
            )
            
            data_config = DataConfig(
                default_paths=data_data.get('default_paths', {}),
                validation=data_data.get('validation', {})
            )
            
            output_config = OutputConfig(
                directory=output_data.get('directory', 'output'),
                file_naming=output_data.get('file_naming', {}),
                export_formats=output_data.get('export_formats', {})
            )
            
            cli_config = CLIConfig(
                defaults=cli_data.get('defaults', {}),
                validation=cli_data.get('validation', {}),
                examples=cli_data.get('examples', {})
            )
            
            logging_config = LoggingConfig(
                level=logging_data.get('level', 'INFO'),
                format=logging_data.get('format', '%(levelname)s: %(message)s'),
                file_format=logging_data.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                console=logging_data.get('console', {}),
                file=logging_data.get('file', {})
            )
            
            features_config = FeatureConfig(
                enhanced_logging=features_data.get('enhanced_logging', True),
                data_validation=features_data.get('data_validation', True),
                fallback_data=features_data.get('fallback_data', True),
                performance_monitoring=features_data.get('performance_monitoring', False),
                caching=features_data.get('caching', False)
            )
            
            return NFLModelConfig(
                model=model_config,
                metrics=metrics_config,
                data=data_config,
                output=output_config,
                cli=cli_config,
                logging=logging_config,
                features=features_config,
                environment=self.environment
            )
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to build typed NFL model configuration: {str(e)}",
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
        
        # Validate home field advantage range
        hfa = self._config.model.home_field_advantage
        if not (-5.0 <= hfa <= 10.0):
            errors.append(f"Home field advantage {hfa} outside reasonable range [-5.0, 10.0]")
        
        # Validate tolerance points
        tolerance = self._config.model.validation.get('tolerance_points', 3.0)
        if tolerance <= 0:
            errors.append(f"Tolerance points must be positive, got {tolerance}")
        
        # Validate confidence levels
        conf_level = self._config.model.calculation.get('confidence_level', 0.95)
        if not (0.5 <= conf_level <= 0.99):
            errors.append(f"Confidence level {conf_level} outside valid range [0.5, 0.99]")
        
        # Validate metrics thresholds
        if self._config.metrics.accuracy_threshold <= 0.5:
            errors.append(f"Accuracy threshold {self._config.metrics.accuracy_threshold} too low")
            
        if self._config.metrics.bootstrap_iterations < 100:
            errors.append(f"Bootstrap iterations {self._config.metrics.bootstrap_iterations} too low")
        
        # Validate week ranges
        week_range = self._config.cli.validation.get('week_range', [1, 22])
        if len(week_range) != 2 or week_range[0] >= week_range[1]:
            errors.append(f"Invalid week range: {week_range}")
        
        # Validate logging level
        if not validate_logging_level(self._config.logging.level):
            errors.append(f"Invalid logging level: {self._config.logging.level}")
        
        if errors:
            raise ConfigurationError(
                f"NFL model configuration validation failed: {len(errors)} error(s)",
                context={'validation_errors': errors},
                recovery_suggestions=[
                    "Fix the configuration errors listed",
                    "Check the configuration file syntax",
                    "Refer to configuration documentation"
                ]
            )
        
        logger.debug("NFL model configuration validation passed")
    
    def get_config(self) -> NFLModelConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get_raw_config(self) -> Optional[Dict[str, Any]]:
        """Get the raw configuration dictionary."""
        return self._raw_config
    
    def get_home_field_advantage(self) -> float:
        """Get the configured home field advantage."""
        return self.get_config().model.home_field_advantage
    
    def get_tolerance_points(self) -> float:
        """Get the configured tolerance points for accuracy calculations."""
        return self.get_config().model.validation.get('tolerance_points', 3.0)
    
    def get_confidence_level(self) -> float:
        """Get the configured confidence level."""
        return self.get_config().model.calculation.get('confidence_level', 0.95)
    
    def get_week_range(self) -> Tuple[int, int]:
        """Get the valid week range."""
        week_range = self.get_config().cli.validation.get('week_range', [1, 22])
        return tuple(week_range)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        return getattr(self.get_config().features, feature_name, False)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if not self._raw_config:
            raise ConfigurationError("No configuration loaded to update")
        
        logger.info("Updating NFL model configuration with new values")
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
            
            logger.info(f"NFL model configuration saved to: {output_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save NFL model configuration: {str(e)}",
                context={'output_path': output_path}
            )

# Global configuration manager instance
_nfl_config_manager: Optional[NFLConfigManager] = None

def get_nfl_config_manager(config_path: Optional[str] = None, environment: Optional[str] = None) -> NFLConfigManager:
    """Get or create the global NFL configuration manager."""
    global _nfl_config_manager
    
    if _nfl_config_manager is None or config_path is not None:
        _nfl_config_manager = NFLConfigManager(config_path, environment)
    
    return _nfl_config_manager

def get_nfl_config() -> NFLModelConfig:
    """Get the current NFL model configuration."""
    return get_nfl_config_manager().get_config()
