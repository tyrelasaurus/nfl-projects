"""
Comprehensive data validation schemas for Power Rankings using Pydantic.
Provides type-safe validation for all data structures in the power ranking system.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Dict, Optional, Union, Literal, Any
from datetime import datetime, date
from enum import Enum
import re

class NFLTeam(str, Enum):
    """Enumeration of all NFL teams with their standard abbreviations."""
    ARIZONA_CARDINALS = "ARI"
    ATLANTA_FALCONS = "ATL"
    BALTIMORE_RAVENS = "BAL"
    BUFFALO_BILLS = "BUF"
    CAROLINA_PANTHERS = "CAR"
    CHICAGO_BEARS = "CHI"
    CINCINNATI_BENGALS = "CIN"
    CLEVELAND_BROWNS = "CLE"
    DALLAS_COWBOYS = "DAL"
    DENVER_BRONCOS = "DEN"
    DETROIT_LIONS = "DET"
    GREEN_BAY_PACKERS = "GB"
    HOUSTON_TEXANS = "HOU"
    INDIANAPOLIS_COLTS = "IND"
    JACKSONVILLE_JAGUARS = "JAX"
    KANSAS_CITY_CHIEFS = "KC"
    LAS_VEGAS_RAIDERS = "LV"
    LOS_ANGELES_CHARGERS = "LAC"
    LOS_ANGELES_RAMS = "LAR"
    MIAMI_DOLPHINS = "MIA"
    MINNESOTA_VIKINGS = "MIN"
    NEW_ENGLAND_PATRIOTS = "NE"
    NEW_ORLEANS_SAINTS = "NO"
    NEW_YORK_GIANTS = "NYG"
    NEW_YORK_JETS = "NYJ"
    PHILADELPHIA_EAGLES = "PHI"
    PITTSBURGH_STEELERS = "PIT"
    SAN_FRANCISCO_49ERS = "SF"
    SEATTLE_SEAHAWKS = "SEA"
    TAMPA_BAY_BUCCANEERS = "TB"
    TENNESSEE_TITANS = "TEN"
    WASHINGTON_COMMANDERS = "WAS"

class GameStatus(str, Enum):
    """Game status enumeration."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"

class ESPNGameData(BaseModel):
    """Schema for individual game data from ESPN API."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    game_id: str = Field(..., description="Unique game identifier")
    week: int = Field(..., ge=1, le=22, description="NFL week number")
    season: int = Field(..., ge=2020, le=2030, description="NFL season year")
    date: datetime = Field(..., description="Game date and time")
    
    home_team: str = Field(..., min_length=2, max_length=5, description="Home team abbreviation")
    away_team: str = Field(..., min_length=2, max_length=5, description="Away team abbreviation")
    home_score: Optional[int] = Field(None, ge=0, le=100, description="Home team score")
    away_score: Optional[int] = Field(None, ge=0, le=100, description="Away team score")
    
    status: GameStatus = Field(default=GameStatus.SCHEDULED, description="Game status")
    is_playoff: bool = Field(default=False, description="Whether this is a playoff game")
    
    # ESPN-specific fields
    espn_game_id: Optional[str] = Field(None, description="ESPN internal game ID")
    venue: Optional[str] = Field(None, description="Stadium/venue name")
    attendance: Optional[int] = Field(None, ge=0, description="Game attendance")
    
    @field_validator('home_team', 'away_team')
    @classmethod
    def validate_team_format(cls, v: str) -> str:
        """Validate team abbreviation format."""
        if not re.match(r'^[A-Z]{2,5}$', v.upper()):
            raise ValueError('Team abbreviation must be 2-5 uppercase letters')
        return v.upper()
    
    @model_validator(mode='after')
    def validate_game_logic(self) -> 'ESPNGameData':
        """Validate game logic constraints."""
        # Teams cannot play themselves
        if self.home_team == self.away_team:
            raise ValueError('Home team and away team cannot be the same')
        
        # Scores must be consistent with status
        if self.status == GameStatus.COMPLETED:
            if self.home_score is None or self.away_score is None:
                raise ValueError('Completed games must have both scores')
        
        # Playoff games are typically weeks 19-22
        if self.is_playoff and self.week < 19:
            raise ValueError('Playoff games typically occur in weeks 19-22')
        
        return self
    
    @property
    def margin(self) -> Optional[int]:
        """Calculate point margin (home team perspective)."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None
    
    @property
    def total_points(self) -> Optional[int]:
        """Calculate total points scored."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None

class TeamRanking(BaseModel):
    """Schema for individual team power ranking."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    team_id: str = Field(..., description="Team identifier")
    team_name: str = Field(..., min_length=3, max_length=50, description="Full team name")
    team_abbreviation: str = Field(..., min_length=2, max_length=5, description="Team abbreviation")
    
    power_score: float = Field(..., ge=-50.0, le=50.0, description="Power ranking score")
    rank: int = Field(..., ge=1, le=32, description="Team rank (1-32)")
    
    # Statistical components
    season_avg_margin: float = Field(..., description="Season average point margin")
    rolling_avg_margin: float = Field(..., description="Rolling average point margin")
    strength_of_schedule: float = Field(..., description="Strength of schedule rating")
    recency_factor: float = Field(default=0.0, description="Recent performance factor")
    
    # Confidence metrics
    confidence_lower: Optional[float] = Field(None, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, description="Upper confidence bound")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")
    
    # Additional metadata
    games_played: int = Field(..., ge=0, le=22, description="Number of games played")
    wins: int = Field(default=0, ge=0, description="Number of wins")
    losses: int = Field(default=0, ge=0, description="Number of losses")
    ties: int = Field(default=0, ge=0, description="Number of ties")
    
    @field_validator('team_abbreviation')
    @classmethod
    def validate_team_abbreviation(cls, v: str) -> str:
        """Validate team abbreviation format."""
        return v.upper()
    
    @model_validator(mode='after')
    def validate_win_loss_logic(self) -> 'TeamRanking':
        """Validate win-loss logic."""
        total_games = self.wins + self.losses + self.ties
        if total_games != self.games_played:
            raise ValueError(f'Wins({self.wins}) + Losses({self.losses}) + Ties({self.ties}) must equal games played({self.games_played})')
        
        # Confidence bounds validation
        if self.confidence_lower is not None and self.confidence_upper is not None:
            if self.confidence_lower > self.power_score or self.confidence_upper < self.power_score:
                raise ValueError('Power score must be within confidence bounds')
            if self.confidence_lower >= self.confidence_upper:
                raise ValueError('Lower confidence bound must be less than upper bound')
        
        return self

class PowerRankingOutput(BaseModel):
    """Schema for complete power ranking output."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    rankings: List[TeamRanking] = Field(..., min_length=1, max_length=32, description="Team rankings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Output metadata")
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    week: Optional[int] = Field(None, ge=1, le=22, description="Target week")
    season: int = Field(default=2025, ge=2020, le=2030, description="NFL season")
    
    # Model configuration snapshot
    model_weights: Dict[str, float] = Field(..., description="Model weights used")
    rolling_window: int = Field(default=5, ge=1, le=10, description="Rolling window size")
    week18_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Week 18 weight factor")
    
    @model_validator(mode='after')
    def validate_rankings_consistency(self) -> 'PowerRankingOutput':
        """Validate ranking consistency."""
        # Check for duplicate ranks
        ranks = [r.rank for r in self.rankings]
        if len(set(ranks)) != len(ranks):
            raise ValueError('Duplicate ranks found in rankings')
        
        # Check for duplicate teams
        teams = [r.team_abbreviation for r in self.rankings]
        if len(set(teams)) != len(teams):
            raise ValueError('Duplicate teams found in rankings')
        
        # Validate rank sequence (should be 1, 2, 3, ... n)
        expected_ranks = list(range(1, len(self.rankings) + 1))
        if sorted(ranks) != expected_ranks:
            raise ValueError(f'Ranks must be sequential from 1 to {len(self.rankings)}')
        
        # Validate model weights sum to approximately 1.0
        weight_sum = sum(self.model_weights.values())
        if not (0.95 <= weight_sum <= 1.05):
            raise ValueError(f'Model weights sum to {weight_sum:.3f}, expected ~1.0')
        
        return self

class ESPNAPIResponse(BaseModel):
    """Schema for ESPN API response validation."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    events: List[ESPNGameData] = Field(default_factory=list, description="List of game events")
    season: Dict[str, Any] = Field(default_factory=dict, description="Season information")
    week: Dict[str, Any] = Field(default_factory=dict, description="Week information")
    
    # Response metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    status_code: int = Field(default=200, ge=200, le=299, description="HTTP status code")
    
    @model_validator(mode='after')
    def validate_response_consistency(self) -> 'ESPNAPIResponse':
        """Validate API response consistency."""
        if not self.events:
            # Empty response is valid but should be logged
            pass
        else:
            # Check for consistent week/season across events
            weeks = {event.week for event in self.events}
            seasons = {event.season for event in self.events}
            
            if len(weeks) > 1:
                raise ValueError(f'Mixed weeks in response: {weeks}')
            if len(seasons) > 1:
                raise ValueError(f'Mixed seasons in response: {seasons}')
        
        return self

class TeamStatsInput(BaseModel):
    """Schema for team statistics input validation."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    team: str = Field(..., description="Team identifier")
    wins: int = Field(..., ge=0, le=22, description="Wins")
    losses: int = Field(..., ge=0, le=22, description="Losses")
    ties: int = Field(default=0, ge=0, le=2, description="Ties")
    
    points_for: int = Field(..., ge=0, description="Total points scored")
    points_against: int = Field(..., ge=0, description="Total points allowed")
    
    # Advanced metrics (optional)
    offensive_yards: Optional[int] = Field(None, ge=0, description="Total offensive yards")
    defensive_yards: Optional[int] = Field(None, ge=0, description="Total defensive yards allowed")
    turnovers_forced: Optional[int] = Field(None, ge=0, description="Turnovers forced")
    turnovers_committed: Optional[int] = Field(None, ge=0, description="Turnovers committed")
    
    @property
    def games_played(self) -> int:
        """Calculate total games played."""
        return self.wins + self.losses + self.ties
    
    @property
    def point_differential(self) -> int:
        """Calculate point differential."""
        return self.points_for - self.points_against
    
    @property
    def avg_points_per_game(self) -> float:
        """Calculate average points per game."""
        return self.points_for / max(self.games_played, 1)

class ConfigurationSchema(BaseModel):
    """Schema for configuration validation."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='allow')
    
    # Model weights validation
    model_weights: Dict[str, float] = Field(..., description="Model weights configuration")
    rolling_window: int = Field(..., ge=1, le=10, description="Rolling window size")
    week18_weight: float = Field(..., ge=0.0, le=1.0, description="Week 18 weight factor")
    
    # API configuration
    api_timeout: int = Field(default=30, ge=5, le=300, description="API timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum API retries")
    
    @field_validator('model_weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate model weights sum to approximately 1.0."""
        required_keys = {'season_avg_margin', 'rolling_avg_margin', 'sos', 'recency_factor'}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f'Missing required weight keys: {required_keys - v.keys()}')
        
        weight_sum = sum(v.values())
        if not (0.95 <= weight_sum <= 1.05):
            raise ValueError(f'Weights sum to {weight_sum:.3f}, expected ~1.0')
        
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f'Weight {key}={value} must be between 0.0 and 1.0')
        
        return v

class ValidationResult(BaseModel):
    """Schema for validation result reporting."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    
    # Metadata
    validated_at: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    validator_version: str = Field(default="1.0.0", description="Schema version")
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0

# Export main schemas for easy importing
__all__ = [
    'NFLTeam',
    'GameStatus', 
    'ESPNGameData',
    'TeamRanking',
    'PowerRankingOutput',
    'ESPNAPIResponse',
    'TeamStatsInput',
    'ConfigurationSchema',
    'ValidationResult'
]