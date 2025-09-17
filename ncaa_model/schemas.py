"""
Fixed NCAA Spread Model schemas without circular references.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Optional, Any
from datetime import datetime
from enum import Enum

class MatchupStatus(str, Enum):
    """Status of a matchup."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"

class PowerRankingRecord(BaseModel):
    """Schema for power ranking input data."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    team_name: str = Field(..., min_length=3, max_length=50, description="Full team name")
    team_abbreviation: Optional[str] = Field(None, min_length=2, max_length=5, description="Team abbreviation")
    power_score: float = Field(..., ge=-50.0, le=50.0, description="Power ranking score")
    rank: int = Field(..., ge=1, le=150, description="Team rank")
    
    # Optional statistical components
    season_avg_margin: Optional[float] = Field(None, description="Season average margin")
    rolling_avg_margin: Optional[float] = Field(None, description="Rolling average margin")
    strength_of_schedule: Optional[float] = Field(None, description="Strength of schedule")
    
    # Record information
    wins: Optional[int] = Field(None, ge=0, le=16, description="Wins")
    losses: Optional[int] = Field(None, ge=0, le=16, description="Losses")
    ties: Optional[int] = Field(None, ge=0, le=2, description="Ties")

class ScheduleRecord(BaseModel):
    """Schema for schedule/matchup data."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    home_team: str = Field(..., min_length=2, max_length=50, description="Home team name")
    away_team: str = Field(..., min_length=2, max_length=50, description="Away team name") 
    week: int = Field(..., ge=1, le=16, description="NCAA week number")
    date: str = Field(..., description="Game date")
    
    # Optional fields
    game_id: Optional[str] = Field(None, description="Unique game identifier")
    season: Optional[int] = Field(None, ge=2020, le=2030, description="Season year")
    venue: Optional[str] = Field(None, description="Stadium/venue")
    
    # Actual results (if completed)
    home_score: Optional[int] = Field(None, ge=0, le=150, description="Home team final score")
    away_score: Optional[int] = Field(None, ge=0, le=150, description="Away team final score")
    status: MatchupStatus = Field(default=MatchupStatus.SCHEDULED, description="Game status")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format."""
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(f'Invalid date format: {v}')
    
    @model_validator(mode='after')
    def validate_matchup_logic(self):
        """Validate matchup logic."""
        # Teams cannot play themselves
        if self.home_team.lower() == self.away_team.lower():
            raise ValueError('Home and away teams cannot be the same')
        
        # Score validation
        if self.status == MatchupStatus.COMPLETED:
            if self.home_score is None or self.away_score is None:
                raise ValueError('Completed games must have scores')
        
        return self
    
    @property
    def margin(self) -> Optional[int]:
        """Calculate margin (home team perspective)."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None

class ModelConfiguration(BaseModel):
    """Schema for model configuration validation."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    # Core model parameters
    home_field_advantage: float = Field(..., ge=-10.0, le=10.0, description="Home field advantage")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")
    
    # Validation parameters
    tolerance_points: float = Field(default=3.0, gt=0.0, le=10.0, description="Accuracy tolerance")
    min_sample_size: int = Field(default=10, ge=1, description="Minimum sample size")
    
    # Performance thresholds
    accuracy_threshold: float = Field(default=0.55, ge=0.5, le=1.0, description="Accuracy threshold")
    rmse_target: float = Field(default=15.0, gt=0.0, description="RMSE target")

class DataValidationReport(BaseModel):
    """Schema for data validation reporting."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Validation summary
    is_valid: bool = Field(..., description="Overall validation status")
    total_records: int = Field(..., ge=0, description="Total records validated")
    valid_records: int = Field(..., ge=0, description="Valid records count")
    invalid_records: int = Field(..., ge=0, description="Invalid records count")
    
    # Error details
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    # Metadata
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    schema_version: str = Field(default="1.0.0", description="Schema version")
    
    @property
    def validation_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_records == 0:
            return 1.0
        return self.valid_records / self.total_records
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.validation_errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.validation_errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

# Export main schemas
__all__ = [
    'MatchupStatus',
    'PowerRankingRecord',
    'ScheduleRecord', 
    'ModelConfiguration',
    'DataValidationReport'
]
