"""
Enhanced data validation using Pydantic schemas for Power Rankings.
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
from schemas import (
    ESPNGameData, TeamRanking, PowerRankingOutput, 
    ESPNAPIResponse, TeamStatsInput, ValidationResult,
    ConfigurationSchema, NFLTeam
)

logger = logging.getLogger(__name__)

class PowerRankingDataValidator:
    """
    Comprehensive data validator for Power Rankings system using Pydantic schemas.
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
    
    def validate_espn_game_data(self, game_data: Dict[str, Any]) -> Tuple[Optional[ESPNGameData], ValidationResult]:
        """
        Validate ESPN game data against schema.
        
        Args:
            game_data: Raw game data dictionary
            
        Returns:
            Tuple of (validated_data, validation_result)
        """
        result = ValidationResult(is_valid=True)
        
        try:
            validated_game = ESPNGameData(**game_data)
            result.is_valid = True
            self._update_stats(success=True)
            
            # Additional business logic validation
            if validated_game.total_points and validated_game.total_points > 80:
                result.add_warning(f"Unusually high total points: {validated_game.total_points}")
            
            if validated_game.margin and abs(validated_game.margin) > 40:
                result.add_warning(f"Unusually large margin: {validated_game.margin}")
            
            return validated_game, result
            
        except ValidationError as e:
            self._update_stats(success=False)
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                result.add_error(message)
            
            if self.strict_mode:
                raise ValueError(f"Game data validation failed: {result.errors}")
            
            return None, result
    
    def validate_team_ranking(self, ranking_data: Dict[str, Any]) -> Tuple[Optional[TeamRanking], ValidationResult]:
        """
        Validate team ranking data against schema.
        
        Args:
            ranking_data: Raw team ranking dictionary
            
        Returns:
            Tuple of (validated_ranking, validation_result)
        """
        result = ValidationResult(is_valid=True)
        
        try:
            validated_ranking = TeamRanking(**ranking_data)
            result.is_valid = True
            self._update_stats(success=True)
            
            # Additional validation checks
            if validated_ranking.power_score > 30:
                result.add_warning(f"Exceptionally high power score: {validated_ranking.power_score}")
            elif validated_ranking.power_score < -30:
                result.add_warning(f"Exceptionally low power score: {validated_ranking.power_score}")
            
            # Check win percentage consistency
            if validated_ranking.games_played > 0:
                win_pct = validated_ranking.wins / validated_ranking.games_played
                if validated_ranking.rank <= 5 and win_pct < 0.6:
                    result.add_warning(f"High rank ({validated_ranking.rank}) with low win rate ({win_pct:.2f})")
                elif validated_ranking.rank >= 28 and win_pct > 0.4:
                    result.add_warning(f"Low rank ({validated_ranking.rank}) with high win rate ({win_pct:.2f})")
            
            return validated_ranking, result
            
        except ValidationError as e:
            self._update_stats(success=False)
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                result.add_error(message)
            
            if self.strict_mode:
                raise ValueError(f"Team ranking validation failed: {result.errors}")
            
            return None, result
    
    def validate_power_ranking_output(self, output_data: Dict[str, Any]) -> Tuple[Optional[PowerRankingOutput], ValidationResult]:
        """
        Validate complete power ranking output.
        
        Args:
            output_data: Raw power ranking output dictionary
            
        Returns:
            Tuple of (validated_output, validation_result)
        """
        result = ValidationResult(is_valid=True)
        
        try:
            validated_output = PowerRankingOutput(**output_data)
            result.is_valid = True
            self._update_stats(success=True)
            
            # Validate ranking distribution
            power_scores = [r.power_score for r in validated_output.rankings]
            score_std = np.std(power_scores)
            
            if score_std < 5.0:
                result.add_warning(f"Low power score variance (std={score_std:.2f}), rankings may be too compressed")
            elif score_std > 20.0:
                result.add_warning(f"High power score variance (std={score_std:.2f}), rankings may be too spread out")
            
            # Check for reasonable score distribution
            top_5_avg = np.mean([r.power_score for r in validated_output.rankings if r.rank <= 5])
            bottom_5_avg = np.mean([r.power_score for r in validated_output.rankings if r.rank >= 28])
            
            if (top_5_avg - bottom_5_avg) < 10:
                result.add_warning("Small difference between top and bottom teams may indicate insufficient differentiation")
            
            return validated_output, result
            
        except ValidationError as e:
            self._update_stats(success=False)
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Field '{field}': {error['msg']}"
                result.add_error(message)
            
            if self.strict_mode:
                raise ValueError(f"Power ranking output validation failed: {result.errors}")
            
            return None, result
    
    def validate_csv_data(self, csv_path: Union[str, Path], 
                         schema_class: Type[BaseModel],
                         required_columns: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
        """
        Validate CSV data against a Pydantic schema.
        
        Args:
            csv_path: Path to CSV file
            schema_class: Pydantic model class for validation
            required_columns: List of required column names
            
        Returns:
            Tuple of (validated_dataframe, validation_result)
        """
        result = ValidationResult(is_valid=True)
        csv_path = Path(csv_path)
        
        try:
            # Check file exists
            if not csv_path.exists():
                result.add_error(f"CSV file not found: {csv_path}")
                return None, result
            
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Check for empty file
            if df.empty:
                result.add_error("CSV file is empty")
                return None, result
            
            # Validate required columns
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    result.add_error(f"Missing required columns: {list(missing_cols)}")
                    return None, result
            
            # Validate each row against schema
            valid_rows = []
            invalid_rows = []
            
            for idx, row in df.iterrows():
                try:
                    # Convert row to dict and validate
                    row_dict = row.to_dict()
                    validated_item = schema_class(**row_dict)
                    valid_rows.append(validated_item.model_dump())
                except ValidationError as e:
                    invalid_rows.append({
                        'row_index': idx,
                        'errors': [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" 
                                 for err in e.errors()]
                    })
            
            # Report validation results
            total_rows = len(df)
            valid_count = len(valid_rows)
            invalid_count = len(invalid_rows)
            
            if invalid_count > 0:
                result.add_warning(f"Validation failed for {invalid_count}/{total_rows} rows")
                
                # Log details of first few invalid rows
                for invalid_row in invalid_rows[:5]:
                    result.add_error(f"Row {invalid_row['row_index']}: {'; '.join(invalid_row['errors'])}")
                
                if invalid_count > 5:
                    result.add_error(f"... and {invalid_count - 5} more invalid rows")
            
            if valid_count == 0:
                result.add_error("No valid rows found in CSV")
                return None, result
            
            # Return DataFrame with valid rows only
            validated_df = pd.DataFrame(valid_rows)
            
            if invalid_count > 0 and not self.strict_mode:
                result.add_warning(f"Returning {valid_count} valid rows, discarded {invalid_count} invalid rows")
            elif invalid_count > 0 and self.strict_mode:
                result.add_error(f"Strict mode: Cannot process file with {invalid_count} invalid rows")
                return None, result
            
            self._update_stats(success=True)
            return validated_df, result
            
        except Exception as e:
            result.add_error(f"CSV validation error: {str(e)}")
            self._update_stats(success=False)
            return None, result
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> Tuple[Optional[ConfigurationSchema], ValidationResult]:
        """
        Validate configuration data.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Tuple of (validated_config, validation_result)
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Extract relevant configuration parts for validation
            config_for_validation = {
                'model_weights': config_data.get('model', {}).get('weights', {}),
                'rolling_window': config_data.get('model', {}).get('rolling_window', 5),
                'week18_weight': config_data.get('model', {}).get('week18_weight', 0.3),
                'api_timeout': config_data.get('api', {}).get('timeout_seconds', 30),
                'max_retries': config_data.get('api', {}).get('max_retries', 3)
            }
            
            validated_config = ConfigurationSchema(**config_for_validation)
            result.is_valid = True
            self._update_stats(success=True)
            
            return validated_config, result
            
        except ValidationError as e:
            self._update_stats(success=False)
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = f"Config field '{field}': {error['msg']}"
                result.add_error(message)
            
            if self.strict_mode:
                raise ValueError(f"Configuration validation failed: {result.errors}")
            
            return None, result
    
    def validate_team_stats_batch(self, stats_list: List[Dict[str, Any]]) -> Tuple[List[TeamStatsInput], ValidationResult]:
        """
        Validate a batch of team statistics.
        
        Args:
            stats_list: List of team statistics dictionaries
            
        Returns:
            Tuple of (validated_stats_list, validation_result)
        """
        result = ValidationResult(is_valid=True)
        validated_stats = []
        
        for idx, stats_data in enumerate(stats_list):
            try:
                validated_stat = TeamStatsInput(**stats_data)
                validated_stats.append(validated_stat)
                
                # Additional validation checks
                if validated_stat.games_played > 18:
                    result.add_warning(f"Team {validated_stat.team}: Unusually high games played ({validated_stat.games_played})")
                
                if validated_stat.point_differential > 200 or validated_stat.point_differential < -200:
                    result.add_warning(f"Team {validated_stat.team}: Extreme point differential ({validated_stat.point_differential})")
                
            except ValidationError as e:
                self._update_stats(success=False)
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    message = f"Stats[{idx}] field '{field}': {error['msg']}"
                    result.add_error(message)
        
        if validated_stats:
            self._update_stats(success=True)
        
        if self.strict_mode and result.has_errors:
            raise ValueError(f"Team stats validation failed: {result.errors}")
        
        return validated_stats, result
    
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
                              strict_mode: bool = True) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """
    Convenience function to validate power rankings CSV file.
    
    Args:
        csv_path: Path to power rankings CSV
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of (validated_dataframe, validation_result)
    """
    validator = PowerRankingDataValidator(strict_mode=strict_mode)
    required_columns = ['team_name', 'power_score', 'rank']
    
    return validator.validate_csv_data(
        csv_path=csv_path,
        schema_class=TeamRanking,
        required_columns=required_columns
    )

def validate_espn_api_response(response_data: Dict[str, Any], 
                             strict_mode: bool = True) -> Tuple[Optional[ESPNAPIResponse], ValidationResult]:
    """
    Convenience function to validate ESPN API response.
    
    Args:
        response_data: ESPN API response dictionary
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of (validated_response, validation_result)
    """
    validator = PowerRankingDataValidator(strict_mode=strict_mode)
    result = ValidationResult(is_valid=True)
    
    try:
        validated_response = ESPNAPIResponse(**response_data)
        return validated_response, result
    except ValidationError as e:
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            message = f"API response field '{field}': {error['msg']}"
            result.add_error(message)
        
        if strict_mode:
            raise ValueError(f"ESPN API response validation failed: {result.errors}")
        
        return None, result