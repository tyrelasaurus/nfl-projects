"""
Enhanced data validation using Pydantic schemas for NCAA Spread Model.
Provides comprehensive validation for all input data sources.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Type
import logging
from pathlib import Path
from datetime import datetime
import json

from pydantic import ValidationError, BaseModel
from .schemas import (
    PowerRankingRecord, ScheduleRecord, ModelConfiguration, 
    DataValidationReport, MatchupStatus
)

logger = logging.getLogger(__name__)

class NCAAModelDataValidator:
    """
    Comprehensive data validator for NCAA Spread Model system using Pydantic schemas.
    """
    
    def __init__(self, strict_mode: bool = True, log_warnings: bool = True):
        """
        Initialize the data validator.
        
        Args:
            strict_mode: If True, validation errors will raise exceptions
            log_warnings: If True, warnings will be logged
        """
        self.strict_mode = strict_mode
        self.log_warnings = log_warnings
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'warnings_generated': 0
        }
    
    def validate_power_ranking_record(self, ranking_data: Dict[str, Any]) -> Tuple[Optional[PowerRankingRecord], DataValidationReport]:
        """
        Validate power ranking input data.
        
        Args:
            ranking_data: Raw power ranking dictionary
            
        Returns:
            Tuple of (validated_ranking, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_ranking = PowerRankingRecord(**ranking_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Additional business logic validation
            if validated_ranking.power_score > 25:
                report.add_warning("Exceptionally high power score", "power_score")
            elif validated_ranking.power_score < -25:
                report.add_warning("Exceptionally low power score", "power_score")
            
            # Validate rank vs power score correlation
            if validated_ranking.rank <= 5 and validated_ranking.power_score < 5:
                report.add_warning("High rank with relatively low power score", "rank")
            elif validated_ranking.rank >= 60 and validated_ranking.power_score > -5:
                report.add_warning("Low rank with relatively high power score", "rank")
            
            return validated_ranking, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Power ranking validation failed: {report.validation_errors}")
            
            return None, report
    
    def validate_schedule_record(self, schedule_data: Dict[str, Any]) -> Tuple[Optional[ScheduleRecord], DataValidationReport]:
        """
        Validate schedule/matchup data.
        
        Args:
            schedule_data: Raw schedule dictionary
            
        Returns:
            Tuple of (validated_schedule, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_schedule = ScheduleRecord(**schedule_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Additional validation checks
            if validated_schedule.week > 16 and validated_schedule.status == MatchupStatus.SCHEDULED:
                report.add_warning("Playoff game marked as scheduled", "week")
            
            # Check for reasonable scores
            if validated_schedule.home_score and validated_schedule.away_score:
                total_points = validated_schedule.home_score + validated_schedule.away_score
                if total_points > 80:
                    report.add_warning(f"Unusually high-scoring game: {total_points} points", "scores")
                elif total_points < 15:
                    report.add_warning(f"Unusually low-scoring game: {total_points} points", "scores")
                
                # Check for blowouts
                if validated_schedule.margin and abs(validated_schedule.margin) > 35:
                    report.add_warning(f"Unusual margin: {validated_schedule.margin} points", "scores")
            
            return validated_schedule, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Schedule validation failed: {report.validation_errors}")
            
            return None, report
    
    def validate_spread_prediction(self, prediction_data: Dict[str, Any]) -> Tuple[Optional[SpreadPrediction], DataValidationReport]:
        """
        Validate spread prediction data.
        
        Args:
            prediction_data: Raw prediction dictionary
            
        Returns:
            Tuple of (validated_prediction, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_prediction = SpreadPrediction(**prediction_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Additional validation checks
            if abs(validated_prediction.projected_spread) > 35:
                report.add_warning(f"Large spread: {validated_prediction.projected_spread:.1f} points", "projected_spread")
            
            if abs(validated_prediction.home_field_adjustment) > 8:
                report.add_warning(f"Unusual home field adjustment: {validated_prediction.home_field_adjustment}", "home_field_adjustment")
            
            # Check power rating differential reasonableness
            power_diff = abs(validated_prediction.home_power - validated_prediction.away_power)
            if power_diff > 50:
                report.add_warning(f"Large power rating differential: {power_diff:.1f}", "power_ratings")
            
            return validated_prediction, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Spread prediction validation failed: {report.validation_errors}")
            
            return None, report
    
    def validate_weekly_output(self, output_data: Dict[str, Any]) -> Tuple[Optional[WeeklySpreadOutput], DataValidationReport]:
        """
        Validate complete weekly spread output.
        
        Args:
            output_data: Raw weekly output dictionary
            
        Returns:
            Tuple of (validated_output, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_output = WeeklySpreadOutput(**output_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Validate week-appropriate number of games
            expected_games = self._get_expected_games_for_week(validated_output.week)
            actual_games = len(validated_output.predictions)
            
            if abs(actual_games - expected_games) > 5:  # Allow some tolerance for cancellations/bye weeks
                report.add_warning(f"Week {validated_output.week}: Expected ~{expected_games} games, got {actual_games}", "predictions")
            
            # Check spread distribution
            spreads = [abs(p.projected_spread) for p in validated_output.predictions]
            avg_spread = np.mean(spreads) if spreads else 0
            
            if avg_spread > 14:
                report.add_warning(f"High average spread: {avg_spread:.1f} points", "spreads")
            elif avg_spread < 4:
                report.add_warning(f"Low average spread: {avg_spread:.1f} points", "spreads")
            
            # Check home/road favorite balance
            if validated_output.home_favorites and validated_output.road_favorites:
                total_games = validated_output.home_favorites + validated_output.road_favorites
                home_pct = validated_output.home_favorites / total_games
                
                if home_pct > 0.8:
                    report.add_warning(f"High percentage of home favorites: {home_pct:.1%}", "favorites")
                elif home_pct < 0.4:
                    report.add_warning(f"Low percentage of home favorites: {home_pct:.1%}", "favorites")
            
            return validated_output, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Weekly output validation failed: {report.validation_errors}")
            
            return None, report
    
    def validate_csv_data(self, csv_path: Union[str, Path], 
                         schema_class: Type[BaseModel],
                         required_columns: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], DataValidationReport]:
        """
        Validate CSV data against a Pydantic schema.
        
        Args:
            csv_path: Path to CSV file
            schema_class: Pydantic model class for validation
            required_columns: List of required column names
            
        Returns:
            Tuple of (validated_dataframe, validation_report)
        """
        csv_path = Path(csv_path)
        report = DataValidationReport(is_valid=True, total_records=0)
        
        try:
            # Check file exists
            if not csv_path.exists():
                report.add_error(f"CSV file not found: {csv_path}")
                return None, report
            
            # Load CSV
            df = pd.read_csv(csv_path)
            report.total_records = len(df)
            
            # Check for empty file
            if df.empty:
                report.add_error("CSV file is empty")
                return None, report
            
            # Validate required columns
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    report.add_error(f"Missing required columns: {list(missing_cols)}")
                    return None, report
            
            # Validate each row against schema
            valid_rows = []
            
            for idx, row in df.iterrows():
                try:
                    # Convert row to dict and validate
                    row_dict = row.to_dict()
                    
                    # Handle NaN values
                    row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                    
                    validated_item = schema_class(**row_dict)
                    valid_rows.append(validated_item.model_dump())
                    
                except ValidationError as e:
                    report.invalid_records += 1
                    error_details = []
                    for error in e.errors():
                        field = '.'.join(str(loc) for loc in error['loc'])
                        error_details.append(f"{field}: {error['msg']}")
                    
                    report.add_error(f"Row {idx}: {'; '.join(error_details)}")
            
            report.valid_records = len(valid_rows)
            
            if report.invalid_records > 0:
                report.add_warning(f"Validation failed for {report.invalid_records}/{report.total_records} rows")
            
            if report.valid_records == 0:
                report.add_error("No valid rows found in CSV")
                return None, report
            
            # Return DataFrame with valid rows only
            validated_df = pd.DataFrame(valid_rows)
            
            if report.invalid_records > 0 and not self.strict_mode:
                report.add_warning(f"Returning {report.valid_records} valid rows, discarded {report.invalid_records} invalid rows")
            elif report.invalid_records > 0 and self.strict_mode:
                report.add_error(f"Strict mode: Cannot process file with {report.invalid_records} invalid rows")
                return None, report
            
            self._update_stats(success=True)
            return validated_df, report
            
        except Exception as e:
            report.add_error(f"CSV validation error: {str(e)}")
            self._update_stats(success=False)
            return None, report
    
    def validate_model_configuration(self, config_data: Dict[str, Any]) -> Tuple[Optional[ModelConfiguration], DataValidationReport]:
        """
        Validate model configuration data.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Tuple of (validated_config, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_config = ModelConfiguration(**config_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Additional validation checks
            if abs(validated_config.home_field_advantage) > 5:
                report.add_warning(f"Unusual home field advantage: {validated_config.home_field_advantage}", "home_field_advantage")
            
            if validated_config.tolerance_points > 7:
                report.add_warning(f"High tolerance threshold: {validated_config.tolerance_points}", "tolerance_points")
            
            return validated_config, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Config field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Model configuration validation failed: {report.validation_errors}")
            
            return None, report
    
    def validate_prediction_accuracy(self, accuracy_data: Dict[str, Any]) -> Tuple[Optional[PredictionAccuracy], DataValidationReport]:
        """
        Validate prediction accuracy data.
        
        Args:
            accuracy_data: Accuracy analysis dictionary
            
        Returns:
            Tuple of (validated_accuracy, validation_report)
        """
        report = DataValidationReport(is_valid=True, total_records=1)
        
        try:
            validated_accuracy = PredictionAccuracy(**accuracy_data)
            report.valid_records = 1
            report.is_valid = True
            self._update_stats(success=True)
            
            # Additional validation checks
            if validated_accuracy.accuracy_rate > 0.7:
                report.add_warning(f"Exceptionally high accuracy: {validated_accuracy.accuracy_rate:.1%}", "accuracy_rate")
            elif validated_accuracy.accuracy_rate < 0.45:
                report.add_warning(f"Low accuracy rate: {validated_accuracy.accuracy_rate:.1%}", "accuracy_rate")
            
            if validated_accuracy.rmse > 20:
                report.add_warning(f"High RMSE: {validated_accuracy.rmse:.1f}", "rmse")
            elif validated_accuracy.rmse < 8:
                report.add_warning(f"Unusually low RMSE: {validated_accuracy.rmse:.1f}", "rmse")
            
            return validated_accuracy, report
            
        except ValidationError as e:
            report.invalid_records = 1
            report.valid_records = 0
            self._update_stats(success=False)
            
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                report.add_error(message, field)
            
            if self.strict_mode:
                raise ValueError(f"Prediction accuracy validation failed: {report.validation_errors}")
            
            return None, report
    
    def _get_expected_games_for_week(self, week: int) -> int:
        """Get expected number of games for a given week."""
        if 1 <= week <= 13:
            return 60  # Full FBS slate
        elif week in (14, 15):
            return 20  # Conference championships and limited regular season makeups
        elif week >= 16:
            return 10  # Bowl season / playoffs
        else:
            return 60  # Default to full slate
    
    def _update_stats(self, success: bool) -> None:
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        if success:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        stats = self.validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['successful_validations'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'warnings_generated': 0
        }

def validate_power_rankings_csv(csv_path: Union[str, Path], 
                              strict_mode: bool = True) -> Tuple[Optional[pd.DataFrame], DataValidationReport]:
    """
    Convenience function to validate power rankings CSV file.
    
    Args:
        csv_path: Path to power rankings CSV
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of (validated_dataframe, validation_report)
    """
    validator = NCAAModelDataValidator(strict_mode=strict_mode)
    required_columns = ['team_name', 'power_score', 'rank']
    
    return validator.validate_csv_data(
        csv_path=csv_path,
        schema_class=PowerRankingRecord,
        required_columns=required_columns
    )

def validate_schedule_csv(csv_path: Union[str, Path], 
                         strict_mode: bool = True) -> Tuple[Optional[pd.DataFrame], DataValidationReport]:
    """
    Convenience function to validate schedule CSV file.
    
    Args:
        csv_path: Path to schedule CSV
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of (validated_dataframe, validation_report)
    """
    validator = NCAAModelDataValidator(strict_mode=strict_mode)
    required_columns = ['home_team', 'away_team', 'week', 'date']
    
    return validator.validate_csv_data(
        csv_path=csv_path,
        schema_class=ScheduleRecord,
        required_columns=required_columns
    )
