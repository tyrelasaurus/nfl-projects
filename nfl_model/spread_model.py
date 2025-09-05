from typing import Dict, List, Tuple, NamedTuple, Optional
from dataclasses import dataclass
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from config_manager import get_nfl_config
except ImportError:
    # Fallback for testing - create a mock function
    def get_nfl_config():
        class MockConfig:
            class MockModel:
                home_field_advantage = 2.0
            model = MockModel()
        return MockConfig()


@dataclass
class MatchupResult:
    """Result of a spread calculation for a single matchup."""
    week: int
    home_team: str
    away_team: str
    home_power: float
    away_power: float
    neutral_diff: float
    home_field_adj: float
    projected_spread: float
    game_date: str


class SpreadCalculator:
    """
    NFL point spread prediction engine implementing Billy Walters methodology.
    
    This calculator transforms power rankings into point spread predictions using
    the proven Billy Walters approach, which emphasizes power rating differentials
    with home field advantage adjustments.
    
    Core Formula:
        Spread = (Home_Power - Away_Power) + Home_Field_Advantage
        
    Key Principles:
    - Negative spreads indicate home team favored (e.g., -7.0 = home team by 7)
    - Positive spreads indicate away team favored (e.g., +3.5 = away team by 3.5)
    - Power ratings should be normalized to ±15 range for realistic spreads
    - Home field advantage typically ranges from 1.5 to 3.5 points
    
    Statistical Validation:
    The model targets >52.4% against-the-spread accuracy, which represents
    break-even performance accounting for standard sportsbook juice (-110).
    
    Attributes:
        home_field_advantage (float): Points added for home team advantage
        config: Configuration manager with model parameters
        
    Example:
        >>> calculator = SpreadCalculator(home_field_advantage=2.5)
        >>> spread = calculator.calculate_spread("KC", "BUF", power_rankings)
        >>> print(f"Spread: KC {spread:+.1f}")  # e.g., "Spread: KC -3.5"
    """
    
    def __init__(self, home_field_advantage: Optional[float] = None, config=None):
        """
        Initialize the spread calculator with home field advantage settings.
        
        Args:
            home_field_advantage: Points to add for home team. If None, uses
                                config value or NFL average of 2.0 points.
            config: Configuration manager instance. If None, loads default config.
                   
        The home field advantage represents the average point differential
        attributable to playing at home, accounting for factors like:
        - Crowd noise and energy
        - Travel fatigue for away team  
        - Referee bias tendencies
        - Familiarity with venue conditions
        
        Typical NFL home field advantages by venue type:
        - Outdoor stadiums: 2.0-2.5 points
        - Dome stadiums: 1.5-2.0 points  
        - High-altitude venues (Denver): 3.0+ points
        - Extreme weather venues: 2.5-3.0 points
        """
        self.config = config or get_nfl_config()
        self.home_field_advantage = (home_field_advantage if home_field_advantage is not None 
                                   else self.config.model.home_field_advantage)
    
    def calculate_neutral_spread(self, team_a_power: float, team_b_power: float) -> float:
        """
        Calculate spread on neutral field.
        
        Args:
            team_a_power: Power rating of team A
            team_b_power: Power rating of team B
            
        Returns:
            Neutral field spread (positive = A favored, negative = B favored)
        """
        return team_a_power - team_b_power
    
    def calculate_matchup_spread(self, 
                               home_team: str,
                               away_team: str, 
                               home_power: float,
                               away_power: float,
                               week: int,
                               game_date: str = "") -> MatchupResult:
        """
        Calculate projected spread for a single matchup.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team  
            home_power: Home team's power rating
            away_power: Away team's power rating
            week: Week number
            game_date: Date of the game
            
        Returns:
            MatchupResult with all calculated values
        """
        # Calculate neutral field difference (home - away)
        neutral_diff = self.calculate_neutral_spread(home_power, away_power)
        
        # Apply home field advantage
        projected_spread = neutral_diff + self.home_field_advantage
        
        return MatchupResult(
            week=week,
            home_team=home_team,
            away_team=away_team,
            home_power=home_power,
            away_power=away_power,
            neutral_diff=neutral_diff,
            home_field_adj=self.home_field_advantage,
            projected_spread=projected_spread,
            game_date=game_date
        )
    
    def calculate_week_spreads(self, 
                             matchups: List[Tuple[str, str, str]],
                             power_ratings: Dict[str, float],
                             week: int) -> List[MatchupResult]:
        """
        Calculate spreads for all matchups in a week.
        
        Args:
            matchups: List of (home_team, away_team, game_date) tuples
            power_ratings: Dictionary mapping team names to power scores
            week: Week number
            
        Returns:
            List of MatchupResult objects
            
        Raises:
            KeyError: If a team is not found in power_ratings
        """
        results = []
        
        for home_team, away_team, game_date in matchups:
            try:
                home_power = power_ratings[home_team]
                away_power = power_ratings[away_team]
                
                result = self.calculate_matchup_spread(
                    home_team=home_team,
                    away_team=away_team,
                    home_power=home_power,
                    away_power=away_power,
                    week=week,
                    game_date=game_date
                )
                
                results.append(result)
                
            except KeyError as e:
                raise KeyError(f"Team not found in power ratings: {e}")
        
        return results
    
    def format_spread_as_betting_line(self, projected_spread: float, home_team: str) -> str:
        """
        Format the projected spread as a betting line.
        
        Args:
            projected_spread: The calculated spread
            home_team: Name of the home team
            
        Returns:
            Formatted betting line (e.g., "Chiefs -7.5")
        """
        if projected_spread > 0:
            # Home team favored
            return f"{home_team} -{abs(projected_spread):.1f}"
        elif projected_spread < 0:
            # Away team favored  
            return f"{home_team} +{abs(projected_spread):.1f}"
        else:
            # Pick 'em
            return f"{home_team} PK"