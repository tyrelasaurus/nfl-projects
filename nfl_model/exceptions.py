"""
Custom exception classes for the NFL Spread Model system.
Provides structured error handling with detailed context and recovery guidance.
"""

from typing import Dict, Any, Optional, List
import logging

# Base Exception Classes

class NFLModelError(Exception):
    """Base exception for all NFL Model system errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 recovery_suggestions: Optional[List[str]] = None):
        """
        Initialize NFLModelError.
        
        Args:
            message: Human-readable error message
            context: Additional context about the error
            recovery_suggestions: List of suggested recovery actions
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.logger = logging.getLogger(self.__class__.__module__)
        
        # Log the error automatically
        self._log_error()
    
    def _log_error(self):
        """Log the error with context."""
        log_message = f"{self.__class__.__name__}: {self.message}"
        if self.context:
            log_message += f" | Context: {self.context}"
        self.logger.error(log_message)
    
    def get_context(self) -> Dict[str, Any]:
        """Get error context."""
        return self.context.copy()
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get recovery suggestions."""
        return self.recovery_suggestions.copy()

# Data Loading Exceptions

class DataLoadingError(NFLModelError):
    """Base exception for data loading errors."""
    pass

class PowerRankingsLoadError(DataLoadingError):
    """Error loading power rankings data."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 missing_columns: Optional[List[str]] = None, **kwargs):
        context = {
            'file_path': file_path,
            'missing_columns': missing_columns or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check power rankings file path and permissions",
            "Verify CSV file format and headers",
            "Ensure required columns are present",
            "Generate power rankings if file is missing"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class ScheduleLoadError(DataLoadingError):
    """Error loading NFL schedule data."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 week: Optional[int] = None, **kwargs):
        context = {
            'file_path': file_path,
            'week': week,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check schedule file path and permissions",
            "Verify CSV file format and structure",
            "Ensure schedule data is up to date",
            "Use NFL API to download schedule if missing"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class DataFormatError(DataLoadingError):
    """Data file has incorrect format."""
    
    def __init__(self, message: str, expected_format: Optional[str] = None, 
                 found_format: Optional[str] = None, **kwargs):
        context = {
            'expected_format': expected_format,
            'found_format': found_format,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check file format specifications",
            "Verify column names and data types",
            "Use data conversion tools if needed",
            "Refer to data format documentation"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Spread Calculation Exceptions

class SpreadCalculationError(NFLModelError):
    """Base exception for spread calculation errors."""
    pass

class InvalidPowerRatingError(SpreadCalculationError):
    """Power rating values are invalid or missing."""
    
    def __init__(self, message: str, team: Optional[str] = None, 
                 power_rating: Optional[float] = None, **kwargs):
        context = {
            'team': team,
            'power_rating': power_rating,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check power ratings data quality",
            "Verify team name consistency",
            "Use default power rating if data is missing",
            "Update power rankings calculation"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class MatchupCalculationError(SpreadCalculationError):
    """Error calculating spread for specific matchup."""
    
    def __init__(self, message: str, home_team: Optional[str] = None, 
                 away_team: Optional[str] = None, week: Optional[int] = None, **kwargs):
        context = {
            'home_team': home_team,
            'away_team': away_team,
            'week': week,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check team names and spellings",
            "Verify both teams have power ratings",
            "Ensure week number is valid",
            "Use neutral site calculation if needed"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class HomeFieldAdvantageError(SpreadCalculationError):
    """Error with home field advantage calculation."""
    
    def __init__(self, message: str, home_team: Optional[str] = None, 
                 hfa_value: Optional[float] = None, **kwargs):
        context = {
            'home_team': home_team,
            'home_field_advantage': hfa_value,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check home field advantage configuration",
            "Verify team-specific HFA values",
            "Use league average HFA if data missing",
            "Review HFA calculation methodology"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Model Configuration Exceptions

class ModelConfigurationError(NFLModelError):
    """Base exception for model configuration errors."""
    pass

class InvalidConfigurationError(ModelConfigurationError):
    """Model configuration contains invalid values."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 invalid_value: Optional[Any] = None, **kwargs):
        context = {
            'config_key': config_key,
            'invalid_value': invalid_value,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check configuration file syntax",
            "Verify configuration value types and ranges",
            "Use default configuration values",
            "Refer to configuration documentation"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class MissingConfigurationError(ModelConfigurationError):
    """Required configuration is missing."""
    
    def __init__(self, message: str, missing_keys: Optional[List[str]] = None, **kwargs):
        context = {
            'missing_keys': missing_keys or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Add missing configuration keys",
            "Use default values for missing configuration",
            "Check configuration file completeness",
            "Refer to configuration template"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Output and Export Exceptions

class OutputError(NFLModelError):
    """Base exception for output and export errors."""
    pass

class SpreadExportError(OutputError):
    """Error exporting spread predictions."""
    
    def __init__(self, message: str, output_format: Optional[str] = None, 
                 file_path: Optional[str] = None, **kwargs):
        context = {
            'output_format': output_format,
            'file_path': file_path,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check output directory permissions",
            "Verify export format specifications",
            "Ensure sufficient disk space",
            "Use alternative export format"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class ReportGenerationError(OutputError):
    """Error generating spread analysis reports."""
    
    def __init__(self, message: str, report_type: Optional[str] = None, 
                 week: Optional[int] = None, **kwargs):
        context = {
            'report_type': report_type,
            'week': week,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check report template availability",
            "Verify input data completeness",
            "Use simplified report format",
            "Review report generation logic"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# CLI and Interface Exceptions

class CLIError(NFLModelError):
    """Base exception for command-line interface errors."""
    pass

class InvalidArgumentError(CLIError):
    """Invalid command-line arguments provided."""
    
    def __init__(self, message: str, argument: Optional[str] = None, 
                 valid_options: Optional[List[str]] = None, **kwargs):
        context = {
            'invalid_argument': argument,
            'valid_options': valid_options or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check command-line argument syntax",
            "Refer to help documentation (--help)",
            "Use valid argument values",
            "Check argument spelling and format"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class MissingArgumentError(CLIError):
    """Required command-line argument is missing."""
    
    def __init__(self, message: str, missing_argument: Optional[str] = None, **kwargs):
        context = {
            'missing_argument': missing_argument,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Provide required command-line arguments",
            "Refer to help documentation (--help)",
            "Check argument names and syntax",
            "Use configuration file for default values"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Validation Exceptions

class ValidationError(NFLModelError):
    """Base exception for data validation errors."""
    pass

class ScheduleValidationError(ValidationError):
    """Schedule data validation failed."""
    
    def __init__(self, message: str, week: Optional[int] = None, 
                 invalid_games: Optional[int] = None, **kwargs):
        context = {
            'week': week,
            'invalid_games': invalid_games,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check schedule data integrity",
            "Verify game information completeness",
            "Update schedule data source",
            "Use manual data correction if needed"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class PowerRatingValidationError(ValidationError):
    """Power rating data validation failed."""
    
    def __init__(self, message: str, invalid_teams: Optional[List[str]] = None, 
                 validation_failures: Optional[List[str]] = None, **kwargs):
        context = {
            'invalid_teams': invalid_teams or [],
            'validation_failures': validation_failures or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check power ratings data quality",
            "Verify team name consistency",
            "Update power rankings source",
            "Use fallback power rating values"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Model Performance Exceptions

class ModelPerformanceError(NFLModelError):
    """Base exception for model performance issues."""
    pass

class PredictionAccuracyError(ModelPerformanceError):
    """Model prediction accuracy is below acceptable threshold."""
    
    def __init__(self, message: str, accuracy_rate: Optional[float] = None, 
                 threshold: Optional[float] = None, **kwargs):
        context = {
            'accuracy_rate': accuracy_rate,
            'threshold': threshold,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Review model parameters and weights",
            "Check input data quality",
            "Consider model retraining",
            "Analyze prediction patterns for systematic bias"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class ModelCalibrationError(ModelPerformanceError):
    """Model calibration is poor."""
    
    def __init__(self, message: str, calibration_score: Optional[float] = None, **kwargs):
        context = {
            'calibration_score': calibration_score,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Recalibrate model parameters",
            "Review training data distribution",
            "Implement model ensemble methods",
            "Update model architecture"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Integration Exceptions

class IntegrationError(NFLModelError):
    """Base exception for system integration errors."""
    pass

class PowerRankingIntegrationError(IntegrationError):
    """Error integrating with power ranking system."""
    
    def __init__(self, message: str, integration_point: Optional[str] = None, **kwargs):
        context = {
            'integration_point': integration_point,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check power ranking system availability",
            "Verify integration API compatibility",
            "Use cached power rankings if available",
            "Implement fallback integration method"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Utility Functions

def handle_nfl_model_exception(func):
    """
    Decorator to handle exceptions with automatic recovery suggestions.
    
    Usage:
        @handle_nfl_model_exception
        def my_function():
            # function code
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NFLModelError:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise NFLModelError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                context={'function': func.__name__, 'args': str(args)[:200]},
                recovery_suggestions=[
                    "Check input parameters",
                    "Review function implementation", 
                    "Check system resources",
                    "Report bug if issue persists"
                ]
            ) from e
    
    return wrapper

def log_model_error(error: Exception, logger: logging.Logger, 
                   context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Centralized error logging and handling for NFL model.
    
    Args:
        error: Exception to handle
        logger: Logger instance
        context: Additional context for logging
        
    Returns:
        True if error was successfully handled, False otherwise
    """
    try:
        if isinstance(error, NFLModelError):
            # Our custom exceptions are already logged
            if context:
                logger.error(f"Additional context: {context}")
            
            if error.recovery_suggestions:
                logger.info("Recovery suggestions:")
                for suggestion in error.recovery_suggestions:
                    logger.info(f"  - {suggestion}")
            
            return True
        else:
            # Handle unexpected exceptions
            logger.error(f"Unexpected error: {str(error)}", exc_info=True)
            if context:
                logger.error(f"Context: {context}")
            return False
    
    except Exception as logging_error:
        # Fallback logging
        print(f"Error in error handling: {logging_error}")
        print(f"Original error: {error}")
        return False

def validate_spread_input(home_team: str, away_team: str, week: int) -> None:
    """
    Validate input parameters for spread calculation.
    
    Args:
        home_team: Home team name
        away_team: Away team name  
        week: Week number
        
    Raises:
        InvalidArgumentError: If parameters are invalid
    """
    if not home_team or not isinstance(home_team, str):
        raise InvalidArgumentError(
            "Home team must be a non-empty string",
            context={'home_team': home_team}
        )
    
    if not away_team or not isinstance(away_team, str):
        raise InvalidArgumentError(
            "Away team must be a non-empty string",
            context={'away_team': away_team}
        )
    
    if home_team == away_team:
        raise InvalidArgumentError(
            "Home team and away team cannot be the same",
            context={'team': home_team}
        )
    
    if not isinstance(week, int) or week < 1 or week > 22:  # Including playoffs
        raise InvalidArgumentError(
            "Week must be an integer between 1 and 22",
            context={'week': week}
        )