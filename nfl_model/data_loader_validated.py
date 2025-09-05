"""
Enhanced data loader with comprehensive Pydantic validation for NFL Spread Model.
Extends the original data loader with type-safe validation and error handling.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

# Import validation components
from .data_validator import (
    NFLModelDataValidator, validate_power_rankings_csv, validate_schedule_csv,
    DataValidationReport
)
from .schemas import PowerRankingRecord, ScheduleRecord, ModelConfiguration

logger = logging.getLogger(__name__)

class ValidatedDataLoader:
    """
    Enhanced data loader with comprehensive validation using Pydantic schemas.
    Provides type-safe loading with detailed error reporting.
    """
    
    def __init__(self, strict_mode: bool = True, log_validation_details: bool = True):
        """
        Initialize validated data loader.
        
        Args:
            strict_mode: If True, validation errors will raise exceptions
            log_validation_details: If True, validation details will be logged
        """
        self.strict_mode = strict_mode
        self.log_validation_details = log_validation_details
        self.validator = NFLModelDataValidator(strict_mode=strict_mode)
        self.validation_reports: List[DataValidationReport] = []
    
    def load_power_rankings(self, csv_path: Union[str, Path]) -> Tuple[Dict[str, float], DataValidationReport]:
        """
        Load and validate power rankings from CSV file.
        
        Args:
            csv_path: Path to power rankings CSV file
            
        Returns:
            Tuple of (power_rankings_dict, validation_report)
            
        Raises:
            ValueError: If validation fails in strict mode
        """
        logger.info(f"Loading power rankings from: {csv_path}")
        
        # Use convenience function for CSV validation
        validated_df, report = validate_power_rankings_csv(csv_path, self.strict_mode)
        self.validation_reports.append(report)
        
        if validated_df is None:
            if self.strict_mode:
                raise ValueError(f"Power rankings validation failed: {report.validation_errors}")
            else:
                logger.warning(f"Power rankings validation failed, returning empty dict")
                return {}, report
        
        if self.log_validation_details:
            logger.info(f"Power rankings validation: {report.valid_records}/{report.total_records} records valid")
            if report.has_warnings:
                for warning in report.warnings:
                    logger.warning(f"Power rankings: {warning}")
        
        # Convert validated DataFrame to dictionary
        power_rankings = {}
        for _, row in validated_df.iterrows():
            team_name = row['team_name']
            power_score = float(row['power_score'])
            power_rankings[team_name] = power_score
        
        logger.info(f"Loaded {len(power_rankings)} power rankings successfully")
        return power_rankings, report
    
    def load_schedule(self, csv_path: Union[str, Path], target_week: Optional[int] = None) -> Tuple[List[Tuple[str, str, str]], DataValidationReport]:
        """
        Load and validate schedule data from CSV file.
        
        Args:
            csv_path: Path to schedule CSV file
            target_week: Optional week filter
            
        Returns:
            Tuple of (matchups_list, validation_report)
            Format: [(home_team, away_team, date), ...]
            
        Raises:
            ValueError: If validation fails in strict mode
        """
        logger.info(f"Loading schedule from: {csv_path}")
        if target_week:
            logger.info(f"Filtering for week: {target_week}")
        
        # Use convenience function for CSV validation
        validated_df, report = validate_schedule_csv(csv_path, self.strict_mode)
        self.validation_reports.append(report)
        
        if validated_df is None:
            if self.strict_mode:
                raise ValueError(f"Schedule validation failed: {report.validation_errors}")
            else:
                logger.warning("Schedule validation failed, returning empty list")
                return [], report
        
        # Filter by week if specified
        if target_week is not None:
            week_filtered = validated_df[validated_df['week'] == target_week]
            if week_filtered.empty:
                logger.warning(f"No games found for week {target_week}")
                return [], report
            validated_df = week_filtered
        
        if self.log_validation_details:
            logger.info(f"Schedule validation: {report.valid_records}/{report.total_records} records valid")
            if target_week:
                logger.info(f"Found {len(validated_df)} games for week {target_week}")
            if report.has_warnings:
                for warning in report.warnings:
                    logger.warning(f"Schedule: {warning}")
        
        # Convert validated DataFrame to list of tuples
        matchups = []
        for _, row in validated_df.iterrows():
            home_team = str(row['home_team'])
            away_team = str(row['away_team'])
            date = str(row['date'])
            matchups.append((home_team, away_team, date))
        
        logger.info(f"Loaded {len(matchups)} schedule matchups successfully")
        return matchups, report
    
    def validate_power_rankings_dict(self, power_rankings: Dict[str, float]) -> DataValidationReport:
        """
        Validate an existing power rankings dictionary.
        
        Args:
            power_rankings: Dictionary of team -> power score
            
        Returns:
            Validation report
        """
        report = DataValidationReport(is_valid=True, total_records=len(power_rankings))
        
        for team, score in power_rankings.items():
            try:
                # Create a mock record for validation
                record_data = {
                    'team_name': team,
                    'power_score': score,
                    'rank': 1,  # Will be ignored for this validation
                    'games_played': 10  # Mock value
                }
                
                validated_record, individual_report = self.validator.validate_power_ranking_record(record_data)
                
                if validated_record is None:
                    report.invalid_records += 1
                    report.add_error(f"Team {team}: {'; '.join(individual_report.validation_errors)}")
                else:
                    report.valid_records += 1
                    
                    # Add any warnings
                    for warning in individual_report.warnings:
                        report.add_warning(f"Team {team}: {warning}")
                
            except Exception as e:
                report.invalid_records += 1
                report.add_error(f"Team {team}: {str(e)}")
        
        report.is_valid = report.invalid_records == 0
        return report
    
    def validate_matchups_list(self, matchups: List[Tuple[str, str, str]], week: Optional[int] = None) -> DataValidationReport:
        """
        Validate a list of matchups.
        
        Args:
            matchups: List of (home_team, away_team, date) tuples
            week: Optional week number for validation
            
        Returns:
            Validation report
        """
        report = DataValidationReport(is_valid=True, total_records=len(matchups))
        
        for i, (home_team, away_team, date) in enumerate(matchups):
            try:
                # Create a record for validation
                record_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': date,
                    'week': week or 1  # Default week if not provided
                }
                
                validated_record, individual_report = self.validator.validate_schedule_record(record_data)
                
                if validated_record is None:
                    report.invalid_records += 1
                    report.add_error(f"Matchup {i}: {'; '.join(individual_report.validation_errors)}")
                else:
                    report.valid_records += 1
                    
                    # Add any warnings
                    for warning in individual_report.warnings:
                        report.add_warning(f"Matchup {i} ({home_team} vs {away_team}): {warning}")
                
            except Exception as e:
                report.invalid_records += 1
                report.add_error(f"Matchup {i}: {str(e)}")
        
        report.is_valid = report.invalid_records == 0
        return report
    
    def load_and_validate_model_config(self, config_data: Dict[str, Any]) -> Tuple[Optional[ModelConfiguration], DataValidationReport]:
        """
        Load and validate model configuration.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Tuple of (validated_config, validation_report)
        """
        logger.info("Validating model configuration")
        
        config_for_validation = {
            'home_field_advantage': config_data.get('home_field_advantage', 2.0),
            'confidence_level': config_data.get('confidence_level', 0.95),
            'tolerance_points': config_data.get('tolerance_points', 3.0),
            'min_sample_size': config_data.get('min_sample_size', 10),
            'accuracy_threshold': config_data.get('accuracy_threshold', 0.55),
            'rmse_target': config_data.get('rmse_target', 10.0),
            'week_range': config_data.get('week_range', (1, 22)),
            'power_score_range': config_data.get('power_score_range', (-50.0, 50.0))
        }
        
        validated_config, report = self.validator.validate_model_configuration(config_for_validation)
        self.validation_reports.append(report)
        
        if self.log_validation_details:
            if validated_config:
                logger.info("Model configuration validation passed")
            else:
                logger.error("Model configuration validation failed")
            
            if report.has_warnings:
                for warning in report.warnings:
                    logger.warning(f"Config: {warning}")
        
        return validated_config, report
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation operations performed.
        
        Returns:
            Dictionary with validation statistics
        """
        total_reports = len(self.validation_reports)
        successful_reports = sum(1 for r in self.validation_reports if r.is_valid)
        total_records_processed = sum(r.total_records for r in self.validation_reports)
        total_valid_records = sum(r.valid_records for r in self.validation_reports)
        total_warnings = sum(len(r.warnings) for r in self.validation_reports)
        total_errors = sum(len(r.validation_errors) for r in self.validation_reports)
        
        return {
            'total_validation_operations': total_reports,
            'successful_operations': successful_reports,
            'failed_operations': total_reports - successful_reports,
            'total_records_processed': total_records_processed,
            'valid_records': total_valid_records,
            'invalid_records': total_records_processed - total_valid_records,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'overall_success_rate': successful_reports / total_reports if total_reports > 0 else 0.0,
            'record_validation_rate': total_valid_records / total_records_processed if total_records_processed > 0 else 0.0,
            'validator_stats': self.validator.get_validation_stats()
        }
    
    def clear_validation_history(self) -> None:
        """Clear validation report history."""
        self.validation_reports.clear()
        self.validator.reset_stats()

class DataLoader:
    """
    Legacy data loader class for backward compatibility.
    Internally uses ValidatedDataLoader but maintains the original interface.
    """
    
    def __init__(self, power_rankings_path: str, schedule_path: str):
        """Initialize with file paths."""
        self.power_rankings_path = power_rankings_path
        self.schedule_path = schedule_path
        self.validated_loader = ValidatedDataLoader(strict_mode=False)  # Non-strict for compatibility
    
    def load_power_rankings(self) -> Dict[str, float]:
        """Load power rankings using legacy interface."""
        power_rankings, report = self.validated_loader.load_power_rankings(self.power_rankings_path)
        
        # Log any validation issues for backward compatibility
        if report.has_errors:
            logger.warning(f"Power rankings validation issues: {len(report.validation_errors)} errors")
        if report.has_warnings:
            logger.info(f"Power rankings validation warnings: {len(report.warnings)} warnings")
        
        return power_rankings
    
    def get_weekly_matchups(self, week: int) -> List[Tuple[str, str]]:
        """Get weekly matchups using legacy interface."""
        matchups, report = self.validated_loader.load_schedule(self.schedule_path, week)
        
        # Log any validation issues
        if report.has_errors:
            logger.warning(f"Schedule validation issues: {len(report.validation_errors)} errors")
        if report.has_warnings:
            logger.info(f"Schedule validation warnings: {len(report.warnings)} warnings")
        
        # Convert to legacy format (remove date)
        legacy_matchups = [(home, away) for home, away, _ in matchups]
        return legacy_matchups

# Convenience functions for direct use
def load_validated_power_rankings(csv_path: Union[str, Path], strict_mode: bool = True) -> Dict[str, float]:
    """
    Convenience function to load power rankings with validation.
    
    Args:
        csv_path: Path to power rankings CSV
        strict_mode: Whether to use strict validation
        
    Returns:
        Dictionary of team -> power score
        
    Raises:
        ValueError: If validation fails in strict mode
    """
    loader = ValidatedDataLoader(strict_mode=strict_mode)
    power_rankings, _ = loader.load_power_rankings(csv_path)
    return power_rankings

def load_validated_schedule(csv_path: Union[str, Path], week: Optional[int] = None, strict_mode: bool = True) -> List[Tuple[str, str, str]]:
    """
    Convenience function to load schedule with validation.
    
    Args:
        csv_path: Path to schedule CSV
        week: Optional week filter
        strict_mode: Whether to use strict validation
        
    Returns:
        List of (home_team, away_team, date) tuples
        
    Raises:
        ValueError: If validation fails in strict mode
    """
    loader = ValidatedDataLoader(strict_mode=strict_mode)
    matchups, _ = loader.load_schedule(csv_path, week)
    return matchups