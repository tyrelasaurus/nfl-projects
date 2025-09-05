import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_power_rankings(csv_path: str) -> Dict[str, float]:
    """
    Load power rankings from CSV file.
    
    Args:
        csv_path: Path to power rankings CSV file
        
    Returns:
        Dictionary mapping team names to power scores
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['team_name', 'power_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Validate power_score is numeric and has no NaN values
    try:
        df['power_score'] = pd.to_numeric(df['power_score'], errors='raise')
        if df['power_score'].isna().any():
            raise ValueError("power_score column contains NaN/missing values")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid power_score values: {e}")
    
    # Convert to dictionary (handles duplicates by keeping last occurrence)
    return dict(zip(df['team_name'], df['power_score']))


def load_schedule(csv_path: str, week: int) -> List[Tuple[str, str, str]]:
    """
    Load NFL schedule from CSV file for a specific week.
    
    Args:
        csv_path: Path to schedule CSV file
        week: Week number to filter by
        
    Returns:
        List of tuples (home_team, away_team, game_date)
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['week', 'home_team', 'away_team']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Filter by week
    week_df = df[df['week'] == week].copy()
    
    # Convert to tuples
    matchups = []
    for _, row in week_df.iterrows():
        game_date = row.get('game_date', '')  # Use empty string if missing
        matchups.append((row['home_team'], row['away_team'], game_date))
    
    return matchups


class PowerRankingsLoader:
    """Class-based power rankings loader for more advanced functionality."""
    
    def __init__(self):
        self.required_columns = ['week', 'rank', 'team_id', 'team_name', 'power_score']
    
    def load_from_file(self, csv_path: str) -> Dict[str, float]:
        """Load power rankings from CSV file."""
        df = pd.read_csv(csv_path)
        self._validate_required_columns(df)
        return self._convert_to_dict(df)
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has all required columns."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _convert_to_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Convert DataFrame to dictionary mapping team names to power scores."""
        return dict(zip(df['team_name'], df['power_score']))


class ScheduleLoader:
    """Class-based schedule loader for more advanced functionality."""
    
    def __init__(self):
        self.required_columns = ['week', 'home_team', 'away_team']
    
    def load_from_file(self, csv_path: str, week: int) -> List[Tuple[str, str, str]]:
        """Load schedule from CSV file for a specific week."""
        df = pd.read_csv(csv_path)
        self._validate_required_columns(df)
        filtered_df = self._filter_by_week(df, week)
        return self._convert_to_tuples(filtered_df)
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has all required columns."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _filter_by_week(self, df: pd.DataFrame, week: int) -> pd.DataFrame:
        """Filter DataFrame to specific week."""
        return df[df['week'] == week].copy()
    
    def _convert_to_tuples(self, df: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """Convert DataFrame to list of tuples."""
        matchups = []
        for _, row in df.iterrows():
            game_date = row.get('game_date', '')  # Use empty string if missing
            matchups.append((row['home_team'], row['away_team'], game_date))
        return matchups


class DataLoader:
    """Loads power rankings and NFL schedule data from CSV files."""
    
    def __init__(self, power_rankings_path: str, schedule_path: str):
        """
        Initialize DataLoader with file paths.
        
        Args:
            power_rankings_path: Path to power rankings CSV file
            schedule_path: Path to NFL schedule CSV file
        """
        self.power_rankings_path = power_rankings_path
        self.schedule_path = schedule_path
        self._power_ratings = None
        self._schedule = None
    
    def load_power_rankings(self) -> Dict[str, float]:
        """
        Load power rankings from CSV and return as team_name -> power_score mapping.
        
        Returns:
            Dictionary mapping team names to their power scores
        """
        if self._power_ratings is None:
            df = pd.read_csv(self.power_rankings_path)
            self._power_ratings = dict(zip(df['team_name'], df['power_score']))
        return self._power_ratings
    
    def load_schedule(self, week: Optional[int] = None) -> pd.DataFrame:
        """
        Load NFL schedule from CSV, optionally filtered by week.
        
        Args:
            week: If provided, filter schedule to only this week
            
        Returns:
            DataFrame with schedule data
        """
        if self._schedule is None:
            self._schedule = pd.read_csv(self.schedule_path)
        
        if week is not None:
            return self._schedule[self._schedule['week'] == week].copy()
        return self._schedule.copy()
    
    def get_team_power_score(self, team_name: str) -> float:
        """
        Get power score for a specific team.
        
        Args:
            team_name: Name of the team
            
        Returns:
            Power score for the team
            
        Raises:
            KeyError: If team not found in power rankings
        """
        power_ratings = self.load_power_rankings()
        if team_name not in power_ratings:
            raise KeyError(f"Team '{team_name}' not found in power rankings")
        return power_ratings[team_name]
    
    def get_weekly_matchups(self, week: int) -> List[Tuple[str, str, str]]:
        """
        Get all matchups for a specific week.
        
        Args:
            week: Week number
            
        Returns:
            List of tuples (home_team, away_team, game_date)
        """
        week_schedule = self.load_schedule(week)
        matchups = []
        
        for _, game in week_schedule.iterrows():
            matchups.append((
                game['home_team_name'],
                game['away_team_name'], 
                game['game_date']
            ))
        
        return matchups