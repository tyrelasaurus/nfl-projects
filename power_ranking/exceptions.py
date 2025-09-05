"""
Custom exception classes for the Power Rankings system.
Provides structured error handling with detailed context and recovery guidance.
"""

from typing import Dict, Any, Optional, List
import logging

# Base Exception Classes

class PowerRankingError(Exception):
    """Base exception for all Power Ranking system errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 recovery_suggestions: Optional[List[str]] = None):
        """
        Initialize PowerRankingError.
        
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

# API-Related Exceptions

class ESPNClientError(PowerRankingError):
    """Base exception for ESPN API client errors."""
    pass

class ESPNAPIError(ESPNClientError):
    """ESPN API returned an error response."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 endpoint: Optional[str] = None, **kwargs):
        context = {
            'status_code': status_code,
            'endpoint': endpoint,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check ESPN API status and rate limits",
            "Verify endpoint URL and parameters",
            "Retry request with exponential backoff",
            "Check network connectivity"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class ESPNRateLimitError(ESPNAPIError):
    """ESPN API rate limit exceeded."""
    
    def __init__(self, message: str = "ESPN API rate limit exceeded", **kwargs):
        recovery_suggestions = [
            "Wait before making additional requests",
            "Implement exponential backoff strategy",
            "Reduce request frequency",
            "Consider caching responses"
        ]
        super().__init__(message, recovery_suggestions=recovery_suggestions, **kwargs)

class ESPNTimeoutError(ESPNAPIError):
    """ESPN API request timed out."""
    
    def __init__(self, message: str = "ESPN API request timed out", timeout: Optional[float] = None, **kwargs):
        context = {'timeout_seconds': timeout, **kwargs.get('context', {})}
        recovery_suggestions = [
            "Increase request timeout value",
            "Check network connectivity",
            "Retry request",
            "Use alternative data source if available"
        ]
        super().__init__(message, context=context, recovery_suggestions=recovery_suggestions, **kwargs)

class ESPNDataError(ESPNClientError):
    """ESPN API returned invalid or unexpected data."""
    
    def __init__(self, message: str, expected_format: Optional[str] = None, 
                 received_data: Optional[Any] = None, **kwargs):
        context = {
            'expected_format': expected_format,
            'received_data_type': type(received_data).__name__ if received_data else None,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check ESPN API documentation for format changes",
            "Validate data structure before processing",
            "Implement fallback data parsing methods",
            "Report issue to ESPN API support"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Data Processing Exceptions

class DataProcessingError(PowerRankingError):
    """Base exception for data processing errors."""
    pass

class DataValidationError(DataProcessingError):
    """Data failed validation checks."""
    
    def __init__(self, message: str, validation_failures: Optional[List[str]] = None, 
                 invalid_records: Optional[int] = None, **kwargs):
        context = {
            'validation_failures': validation_failures or [],
            'invalid_records': invalid_records,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Review data source for quality issues",
            "Implement data cleaning procedures",
            "Check data collection processes",
            "Update validation rules if appropriate"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class DataIncompleteError(DataProcessingError):
    """Required data is missing or incomplete."""
    
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None, 
                 completeness_percentage: Optional[float] = None, **kwargs):
        context = {
            'missing_fields': missing_fields or [],
            'completeness_percentage': completeness_percentage,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check data source connectivity",
            "Verify data collection processes",
            "Use fallback data sources if available",
            "Implement data interpolation where appropriate"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class DataTransformationError(DataProcessingError):
    """Error occurred during data transformation."""
    
    def __init__(self, message: str, transformation_step: Optional[str] = None, 
                 input_data_type: Optional[str] = None, **kwargs):
        context = {
            'transformation_step': transformation_step,
            'input_data_type': input_data_type,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Validate input data format",
            "Check transformation logic",
            "Implement data type conversions",
            "Add error handling for edge cases"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Model Calculation Exceptions

class ModelCalculationError(PowerRankingError):
    """Base exception for model calculation errors."""
    pass

class PowerRankingCalculationError(ModelCalculationError):
    """Error occurred during power ranking calculations."""
    
    def __init__(self, message: str, calculation_step: Optional[str] = None, 
                 teams_affected: Optional[List[str]] = None, **kwargs):
        context = {
            'calculation_step': calculation_step,
            'teams_affected': teams_affected or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check input data quality",
            "Verify calculation parameters",
            "Review mathematical formulas",
            "Implement fallback calculation methods"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class StatisticalCalculationError(ModelCalculationError):
    """Error in statistical calculations."""
    
    def __init__(self, message: str, statistic_type: Optional[str] = None, 
                 sample_size: Optional[int] = None, **kwargs):
        context = {
            'statistic_type': statistic_type,
            'sample_size': sample_size,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check sample size adequacy",
            "Validate input data distribution",
            "Handle division by zero cases",
            "Use robust statistical methods"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class StrengthOfScheduleError(ModelCalculationError):
    """Error in strength of schedule calculations."""
    
    def __init__(self, message: str, team_id: Optional[str] = None, 
                 opponents: Optional[List[str]] = None, **kwargs):
        context = {
            'team_id': team_id,
            'opponents': opponents or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Verify opponent data availability",
            "Check for circular dependencies",
            "Implement iterative SOS calculation",
            "Use default SOS values for incomplete data"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Configuration and Setup Exceptions

class ConfigurationError(PowerRankingError):
    """Base exception for configuration errors."""
    pass

class ConfigFileError(ConfigurationError):
    """Error loading or parsing configuration file."""
    
    def __init__(self, message: str, config_file: Optional[str] = None, 
                 config_section: Optional[str] = None, **kwargs):
        context = {
            'config_file': config_file,
            'config_section': config_section,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check configuration file syntax",
            "Verify file permissions and accessibility",
            "Use default configuration values",
            "Validate configuration schema"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class InvalidConfigurationError(ConfigurationError):
    """Configuration contains invalid values."""
    
    def __init__(self, message: str, invalid_keys: Optional[List[str]] = None, 
                 validation_errors: Optional[List[str]] = None, **kwargs):
        context = {
            'invalid_keys': invalid_keys or [],
            'validation_errors': validation_errors or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check configuration value types and ranges",
            "Review configuration documentation",
            "Use validated default values",
            "Implement configuration validation"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# File I/O Exceptions

class FileOperationError(PowerRankingError):
    """Base exception for file operation errors."""
    pass

class DataFileError(FileOperationError):
    """Error reading or writing data files."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        context = {
            'file_path': file_path,
            'operation': operation,  # 'read', 'write', 'create', etc.
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check file path and permissions",
            "Verify disk space availability",
            "Ensure parent directories exist",
            "Use alternative file formats if needed"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class CSVFormatError(DataFileError):
    """Error in CSV file format."""
    
    def __init__(self, message: str, expected_columns: Optional[List[str]] = None, 
                 found_columns: Optional[List[str]] = None, **kwargs):
        context = {
            'expected_columns': expected_columns or [],
            'found_columns': found_columns or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check CSV file headers",
            "Verify column naming conventions",
            "Implement flexible column mapping",
            "Use CSV validation tools"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Export and Output Exceptions

class ExportError(PowerRankingError):
    """Base exception for export and output errors."""
    pass

class RankingExportError(ExportError):
    """Error exporting power rankings."""
    
    def __init__(self, message: str, export_format: Optional[str] = None, 
                 output_path: Optional[str] = None, **kwargs):
        context = {
            'export_format': export_format,
            'output_path': output_path,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check export format specifications",
            "Verify output directory permissions",
            "Ensure required data is available",
            "Use alternative export formats"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Validation System Exceptions

class ValidationSystemError(PowerRankingError):
    """Base exception for data quality validation system errors."""
    pass

class DataQualityError(ValidationSystemError):
    """Critical data quality issue detected."""
    
    def __init__(self, message: str, quality_score: Optional[float] = None, 
                 critical_issues: Optional[List[str]] = None, **kwargs):
        context = {
            'quality_score': quality_score,
            'critical_issues': critical_issues or [],
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Review data quality validation results",
            "Implement data cleaning procedures",
            "Check data source reliability",
            "Consider using validated fallback data"
        ]
        
        super().__init__(message, context, recovery_suggestions)

class MonitoringError(ValidationSystemError):
    """Error in real-time monitoring system."""
    
    def __init__(self, message: str, monitoring_component: Optional[str] = None, **kwargs):
        context = {
            'monitoring_component': monitoring_component,
            **kwargs.get('context', {})
        }
        
        recovery_suggestions = [
            "Check monitoring system configuration",
            "Verify monitoring thread health",
            "Review system resource availability",
            "Restart monitoring components if needed"
        ]
        
        super().__init__(message, context, recovery_suggestions)

# Utility Functions

def handle_exception_with_recovery(func):
    """
    Decorator to handle exceptions with automatic recovery suggestions.
    
    Usage:
        @handle_exception_with_recovery
        def my_function():
            # function code
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PowerRankingError:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise PowerRankingError(
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

def log_and_handle_error(error: Exception, logger: logging.Logger, 
                        context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Centralized error logging and handling.
    
    Args:
        error: Exception to handle
        logger: Logger instance
        context: Additional context for logging
        
    Returns:
        True if error was successfully handled, False otherwise
    """
    try:
        if isinstance(error, PowerRankingError):
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