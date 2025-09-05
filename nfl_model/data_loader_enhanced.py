"""
Enhanced data loader with comprehensive error handling and recovery.
Extends the original data loader with structured exception management.
"""

import pandas as pd
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging

# Import custom exceptions
# Import from nfl_model exceptions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .exceptions import (
    PowerRankingsLoadError, ScheduleLoadError, DataFormatError,
    ValidationError, PowerRatingValidationError, ScheduleValidationError,
    handle_nfl_model_exception, log_model_error
)

logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """Enhanced data loader with comprehensive error handling."""
    
    def __init__(self, validate_data: bool = True, use_fallback: bool = True):
        """
        Initialize enhanced data loader.
        
        Args:
            validate_data: Whether to perform data validation
            use_fallback: Whether to use fallback values for missing data
        """
        self.validate_data = validate_data
        self.use_fallback = use_fallback
        logger.info(f"Enhanced data loader initialized (validation: {validate_data}, fallback: {use_fallback})")
    
    @handle_nfl_model_exception
    def load_power_rankings(self, csv_path: str) -> Dict[str, float]:
        """
        Load power rankings from CSV file with enhanced error handling.
        
        Args:
            csv_path: Path to power rankings CSV file
            
        Returns:
            Dictionary mapping team names to power scores
            
        Raises:
            PowerRankingsLoadError: For file loading errors
            DataFormatError: For format validation errors
            PowerRatingValidationError: For data validation errors
        """
        logger.info(f"Loading power rankings from: {csv_path}")
        
        # Validate file exists and is readable
        if not os.path.exists(csv_path):
            raise PowerRankingsLoadError(
                f"Power rankings file not found: {csv_path}",
                file_path=csv_path,
                context={'operation': 'file_existence_check'}
            )
        
        if not os.access(csv_path, os.R_OK):
            raise PowerRankingsLoadError(
                f"Power rankings file not readable: {csv_path}",
                file_path=csv_path,
                context={'operation': 'file_permission_check'}
            )
        
        try:
            # Load CSV with error handling
            try:
                df = pd.read_csv(csv_path)
                logger.debug(f"Loaded {len(df)} rows from power rankings file")
            except pd.errors.EmptyDataError:
                raise PowerRankingsLoadError(
                    f"Power rankings file is empty: {csv_path}",
                    file_path=csv_path,
                    context={'error_type': 'empty_file'}
                )
            except pd.errors.ParserError as e:
                raise DataFormatError(
                    f"Error parsing power rankings CSV: {str(e)}",
                    expected_format="CSV with headers",
                    found_format="invalid CSV",
                    context={'file_path': csv_path, 'parser_error': str(e)}
                )
            
            # Validate DataFrame is not empty
            if df.empty:
                raise PowerRankingsLoadError(
                    "Power rankings file contains no data",
                    file_path=csv_path,
                    context={'rows_loaded': 0}
                )
            
            # Validate required columns
            required_columns = ['team_name', 'power_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                available_columns = list(df.columns)
                raise DataFormatError(
                    f"Power rankings file missing required columns: {missing_columns}",
                    expected_format=f"CSV with columns: {required_columns}",
                    found_format=f"CSV with columns: {available_columns}",
                    context={
                        'file_path': csv_path,
                        'missing_columns': missing_columns,
                        'available_columns': available_columns
                    }
                )
            
            # Validate and clean data if enabled
            if self.validate_data:
                df = self._validate_power_rankings_data(df, csv_path)
            
            # Convert to dictionary
            power_rankings = self._convert_to_power_dict(df, csv_path)
            
            logger.info(f"Successfully loaded {len(power_rankings)} team power rankings")
            return power_rankings
            
        except Exception as e:
            context = {'file_path': csv_path, 'operation': 'load_power_rankings'}
            log_model_error(e, logger, context=context)
            
            # Try fallback if enabled
            if self.use_fallback:
                logger.warning("Attempting to use fallback power rankings")
                return self._get_fallback_power_rankings()
            
            raise
    
    @handle_nfl_model_exception
    def load_schedule(self, csv_path: str, week: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """
        Load NFL schedule from CSV file with enhanced error handling.
        
        Args:
            csv_path: Path to schedule CSV file
            week: Week number to filter by (optional)
            
        Returns:
            List of tuples (home_team, away_team, game_date)
            
        Raises:
            ScheduleLoadError: For file loading errors
            DataFormatError: For format validation errors
            ScheduleValidationError: For data validation errors
        """
        logger.info(f"Loading schedule from: {csv_path} (week: {week})")
        
        # Validate file exists and is readable
        if not os.path.exists(csv_path):
            raise ScheduleLoadError(
                f"Schedule file not found: {csv_path}",
                file_path=csv_path,
                week=week,
                context={'operation': 'file_existence_check'}
            )
        
        try:
            # Load CSV
            try:
                df = pd.read_csv(csv_path)
                logger.debug(f"Loaded {len(df)} rows from schedule file")
            except pd.errors.EmptyDataError:
                raise ScheduleLoadError(
                    f"Schedule file is empty: {csv_path}",
                    file_path=csv_path,
                    week=week,
                    context={'error_type': 'empty_file'}
                )
            except pd.errors.ParserError as e:
                raise DataFormatError(
                    f"Error parsing schedule CSV: {str(e)}",
                    expected_format="CSV with headers",
                    found_format="invalid CSV",
                    context={'file_path': csv_path, 'parser_error': str(e)}
                )
            
            # Validate required columns
            required_columns = ['week', 'home_team', 'away_team']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                available_columns = list(df.columns)
                raise DataFormatError(
                    f"Schedule file missing required columns: {missing_columns}",
                    expected_format=f"CSV with columns: {required_columns}",
                    found_format=f"CSV with columns: {available_columns}",
                    context={
                        'file_path': csv_path,
                        'missing_columns': missing_columns,
                        'available_columns': available_columns
                    }
                )
            
            # Filter by week if specified
            if week is not None:
                if not isinstance(week, int) or week < 1 or week > 22:
                    raise ScheduleValidationError(
                        f"Invalid week number: {week}",
                        week=week,
                        context={'valid_range': '1-22'}
                    )
                
                original_count = len(df)
                df = df[df['week'] == week]
                
                if df.empty:
                    available_weeks = sorted(df['week'].unique()) if original_count > 0 else []
                    raise ScheduleLoadError(
                        f"No games found for week {week}",
                        file_path=csv_path,
                        week=week,
                        context={
                            'available_weeks': available_weeks,
                            'total_games_in_file': original_count
                        }
                    )
                
                logger.debug(f"Filtered to {len(df)} games for week {week}")
            
            # Validate data if enabled
            if self.validate_data:
                df = self._validate_schedule_data(df, csv_path, week)
            
            # Convert to tuples
            schedule = self._convert_to_schedule_tuples(df, csv_path)
            
            logger.info(f"Successfully loaded {len(schedule)} scheduled games")
            return schedule
            
        except Exception as e:
            context = {'file_path': csv_path, 'week': week, 'operation': 'load_schedule'}
            log_model_error(e, logger, context=context)
            
            # Try fallback if enabled
            if self.use_fallback and week is not None:
                logger.warning(f"Attempting to use fallback schedule for week {week}")
                return self._get_fallback_schedule(week)
            
            raise
    
    def _validate_power_rankings_data(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """
        Validate power rankings data quality.
        
        Args:
            df: DataFrame to validate
            file_path: Original file path for context
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            PowerRatingValidationError: For validation failures
        """
        validation_errors = []
        invalid_teams = []
        
        # Check for duplicate team names
        duplicates = df[df.duplicated(['team_name'], keep=False)]
        if not duplicates.empty:
            duplicate_teams = duplicates['team_name'].unique().tolist()
            validation_errors.append(f"Duplicate team names: {duplicate_teams}")
            # Keep last occurrence
            df = df.drop_duplicates(['team_name'], keep='last')
        
        # Validate power scores are numeric
        try:
            df['power_score'] = pd.to_numeric(df['power_score'], errors='raise')
        except (ValueError, TypeError) as e:
            raise PowerRatingValidationError(
                f"Non-numeric power scores found: {str(e)}",
                context={'file_path': file_path, 'validation_step': 'numeric_conversion'}
            )
        
        # Check for missing power scores
        missing_scores = df['power_score'].isna()
        if missing_scores.any():
            missing_teams = df.loc[missing_scores, 'team_name'].tolist()
            validation_errors.append(f"Missing power scores for teams: {missing_teams}")
            invalid_teams.extend(missing_teams)
            
            if self.use_fallback:
                # Fill with average power score
                avg_score = df['power_score'].mean()
                df.loc[missing_scores, 'power_score'] = avg_score
                logger.warning(f"Filled missing power scores with average: {avg_score:.2f}")
            else:
                df = df.dropna(subset=['power_score'])
        
        # Check for extreme power score values
        if len(df) > 0:
            mean_score = df['power_score'].mean()
            std_score = df['power_score'].std()
            
            if std_score > 0:  # Avoid division by zero
                z_scores = abs((df['power_score'] - mean_score) / std_score)
                extreme_values = df[z_scores > 3]  # More than 3 standard deviations
                
                if not extreme_values.empty:
                    extreme_teams = extreme_values['team_name'].tolist()
                    extreme_scores = extreme_values['power_score'].tolist()
                    validation_errors.append(f"Extreme power scores detected: {dict(zip(extreme_teams, extreme_scores))}")
        
        # Check for empty team names
        empty_names = df['team_name'].isna() | (df['team_name'].str.strip() == '')
        if empty_names.any():
            validation_errors.append("Empty team names found")
            df = df[~empty_names]
        
        # Log validation results
        if validation_errors:
            logger.warning(f"Power rankings validation issues: {validation_errors}")
            
            if invalid_teams:
                raise PowerRatingValidationError(
                    "Power rankings data validation failed",
                    invalid_teams=invalid_teams,
                    validation_failures=validation_errors,
                    context={'file_path': file_path, 'total_teams': len(df)}
                )
        
        logger.debug(f"Power rankings validation completed: {len(df)} valid teams")
        return df
    
    def _validate_schedule_data(self, df: pd.DataFrame, file_path: str, week: Optional[int]) -> pd.DataFrame:
        """
        Validate schedule data quality.
        
        Args:
            df: DataFrame to validate
            file_path: Original file path for context
            week: Week number for context
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ScheduleValidationError: For validation failures
        """
        validation_errors = []
        invalid_games = 0
        
        # Check for teams playing themselves
        self_games = df[df['home_team'] == df['away_team']]
        if not self_games.empty:
            invalid_games += len(self_games)
            validation_errors.append(f"{len(self_games)} games where teams play themselves")
            df = df[df['home_team'] != df['away_team']]
        
        # Check for empty team names
        empty_home = df['home_team'].isna() | (df['home_team'].str.strip() == '')
        empty_away = df['away_team'].isna() | (df['away_team'].str.strip() == '')
        empty_teams = empty_home | empty_away
        
        if empty_teams.any():
            invalid_games += empty_teams.sum()
            validation_errors.append(f"{empty_teams.sum()} games with empty team names")
            df = df[~empty_teams]
        
        # Check week numbers are valid
        if 'week' in df.columns:
            invalid_weeks = ~df['week'].between(1, 22, inclusive='both')
            if invalid_weeks.any():
                invalid_games += invalid_weeks.sum()
                validation_errors.append(f"{invalid_weeks.sum()} games with invalid week numbers")
                df = df[~invalid_weeks]
        
        # Log validation results
        if validation_errors:
            logger.warning(f"Schedule validation issues: {validation_errors}")
            
            if invalid_games > len(df) * 0.1:  # More than 10% invalid
                raise ScheduleValidationError(
                    "Schedule data validation failed - too many invalid games",
                    week=week,
                    invalid_games=invalid_games,
                    context={
                        'file_path': file_path,
                        'validation_failures': validation_errors,
                        'remaining_games': len(df)
                    }
                )
        
        logger.debug(f"Schedule validation completed: {len(df)} valid games")
        return df
    
    def _convert_to_power_dict(self, df: pd.DataFrame, file_path: str) -> Dict[str, float]:
        """Convert DataFrame to power rankings dictionary."""
        try:
            power_dict = dict(zip(df['team_name'], df['power_score']))
            
            # Validate no team has NaN power score
            nan_teams = [team for team, score in power_dict.items() if pd.isna(score)]
            if nan_teams:
                raise PowerRatingValidationError(
                    f"Teams with NaN power scores: {nan_teams}",
                    invalid_teams=nan_teams,
                    context={'file_path': file_path}
                )
            
            return power_dict
            
        except Exception as e:
            raise PowerRatingValidationError(
                f"Error converting power rankings to dictionary: {str(e)}",
                context={'file_path': file_path, 'dataframe_shape': df.shape}
            ) from e
    
    def _convert_to_schedule_tuples(self, df: pd.DataFrame, file_path: str) -> List[Tuple[str, str, str]]:
        """Convert DataFrame to schedule tuples."""
        try:
            schedule = []
            
            for _, row in df.iterrows():
                home_team = str(row['home_team']).strip()
                away_team = str(row['away_team']).strip()
                game_date = str(row.get('game_date', 'TBD')).strip()
                
                schedule.append((home_team, away_team, game_date))
            
            return schedule
            
        except Exception as e:
            raise ScheduleValidationError(
                f"Error converting schedule to tuples: {str(e)}",
                context={'file_path': file_path, 'dataframe_shape': df.shape}
            ) from e
    
    def _get_fallback_power_rankings(self) -> Dict[str, float]:
        """Get fallback power rankings with average values."""
        logger.warning("Using fallback power rankings with neutral values")
        
        # Standard NFL teams with neutral power ratings
        teams = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
        ]
        
        # Neutral power rating (league average)
        neutral_rating = 85.0
        
        return {team: neutral_rating for team in teams}
    
    def _get_fallback_schedule(self, week: int) -> List[Tuple[str, str, str]]:
        """Get fallback schedule for a week (empty schedule)."""
        logger.warning(f"Using fallback schedule for week {week} (empty)")
        return []

# Convenience functions for backward compatibility
@handle_nfl_model_exception 
def load_power_rankings(csv_path: str, validate_data: bool = True, use_fallback: bool = True) -> Dict[str, float]:
    """
    Load power rankings with enhanced error handling.
    
    Args:
        csv_path: Path to power rankings CSV file
        validate_data: Whether to perform data validation
        use_fallback: Whether to use fallback values for missing data
        
    Returns:
        Dictionary mapping team names to power scores
    """
    loader = EnhancedDataLoader(validate_data=validate_data, use_fallback=use_fallback)
    return loader.load_power_rankings(csv_path)

@handle_nfl_model_exception
def load_schedule(csv_path: str, week: Optional[int] = None, validate_data: bool = True, use_fallback: bool = True) -> List[Tuple[str, str, str]]:
    """
    Load schedule with enhanced error handling.
    
    Args:
        csv_path: Path to schedule CSV file
        week: Week number to filter by
        validate_data: Whether to perform data validation  
        use_fallback: Whether to use fallback values for missing data
        
    Returns:
        List of tuples (home_team, away_team, game_date)
    """
    loader = EnhancedDataLoader(validate_data=validate_data, use_fallback=use_fallback)
    return loader.load_schedule(csv_path, week)