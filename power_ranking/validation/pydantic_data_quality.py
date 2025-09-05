"""
Enhanced data quality system using Pydantic schemas for NFL power rankings.
Integrates comprehensive validation with the existing quality assurance framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from data_validator import PowerRankingDataValidator, ValidationResult
from schemas import ESPNGameData, TeamRanking, PowerRankingOutput, ESPNAPIResponse

logger = logging.getLogger(__name__)

class DataQualityLevel(str, Enum):
    """Data quality severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class DataQualityIssue(BaseModel):
    """Pydantic model for data quality issues."""
    level: DataQualityLevel
    category: str = Field(..., min_length=1, description="Issue category")
    description: str = Field(..., min_length=1, description="Issue description")
    affected_records: int = Field(..., ge=0, description="Number of affected records")
    field_name: Optional[str] = Field(None, description="Specific field affected")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Issue timestamp")
    
    # Suggested remediation
    recommended_action: Optional[str] = Field(None, description="Recommended action")
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Severity score")

class DataQualityReport(BaseModel):
    """Comprehensive data quality assessment report using Pydantic."""
    dataset_name: str = Field(..., min_length=1, description="Name of the dataset")
    total_records: int = Field(..., ge=0, description="Total number of records")
    valid_records: int = Field(..., ge=0, description="Number of valid records")
    invalid_records: int = Field(..., ge=0, description="Number of invalid records")
    
    # Quality scores
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Data completeness score")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Data accuracy score")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Data consistency score")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    
    # Issues and recommendations
    issues: List[DataQualityIssue] = Field(default_factory=list, description="List of quality issues")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    
    # Metadata
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    validation_duration_seconds: Optional[float] = Field(None, description="Validation duration")
    schema_version: str = Field(default="1.0.0", description="Schema version used")
    
    @property
    def validation_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_records == 0:
            return 1.0
        return self.valid_records / self.total_records
    
    @property
    def critical_issues(self) -> List[DataQualityIssue]:
        """Get critical issues only."""
        return [issue for issue in self.issues if issue.level == DataQualityLevel.CRITICAL]
    
    @property
    def high_issues(self) -> List[DataQualityIssue]:
        """Get high priority issues only."""
        return [issue for issue in self.issues if issue.level == DataQualityLevel.HIGH]
    
    def add_issue(self, level: DataQualityLevel, category: str, description: str, 
                  affected_records: int, field_name: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None, 
                  recommended_action: Optional[str] = None) -> None:
        """Add a quality issue to the report."""
        severity_scores = {
            DataQualityLevel.CRITICAL: 1.0,
            DataQualityLevel.HIGH: 0.8,
            DataQualityLevel.MEDIUM: 0.6,
            DataQualityLevel.LOW: 0.4,
            DataQualityLevel.INFO: 0.2
        }
        
        issue = DataQualityIssue(
            level=level,
            category=category,
            description=description,
            affected_records=affected_records,
            field_name=field_name,
            details=details or {},
            recommended_action=recommended_action,
            severity_score=severity_scores.get(level, 0.5)
        )
        self.issues.append(issue)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a quality improvement recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)

class PydanticDataQualityValidator:
    """
    Enhanced data quality validator using Pydantic schemas.
    Combines schema validation with business logic quality checks.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the quality validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.schema_validator = PowerRankingDataValidator(strict_mode=False)  # Always non-strict for quality analysis
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness_critical': 1.0,      # 100% for critical fields
            'completeness_important': 0.95,    # 95% for important fields
            'completeness_optional': 0.7,      # 70% for optional fields
            'accuracy_threshold': 0.9,         # 90% accuracy minimum
            'consistency_threshold': 0.85,     # 85% consistency minimum
            'overall_threshold': 0.8           # 80% overall quality minimum
        }
    
    def validate_espn_game_data(self, games_data: List[Dict[str, Any]]) -> DataQualityReport:
        """
        Validate ESPN game data with comprehensive quality analysis.
        
        Args:
            games_data: List of game data dictionaries
            
        Returns:
            Comprehensive quality report
        """
        start_time = datetime.now()
        
        report = DataQualityReport(
            dataset_name="ESPN Game Data",
            total_records=len(games_data),
            valid_records=0,
            invalid_records=0,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            overall_score=0.0
        )
        
        if not games_data:
            report.add_issue(
                DataQualityLevel.CRITICAL,
                "data_availability",
                "No game data provided",
                0,
                recommended_action="Verify data source and API connectivity"
            )
            return report
        
        # Validate each game record
        valid_games = []
        validation_errors = []
        
        for i, game_data in enumerate(games_data):
            try:
                validated_game, validation_result = self.schema_validator.validate_espn_game_data(game_data)
                if validated_game:
                    valid_games.append(validated_game)
                    report.valid_records += 1
                else:
                    report.invalid_records += 1
                    validation_errors.extend(validation_result.errors)
            except Exception as e:
                report.invalid_records += 1
                validation_errors.append(f"Game {i}: {str(e)}")
        
        # Analyze validation errors
        if validation_errors:
            report.add_issue(
                DataQualityLevel.HIGH,
                "schema_validation",
                f"Schema validation failed for {report.invalid_records} records",
                report.invalid_records,
                details={'sample_errors': validation_errors[:5]},
                recommended_action="Review data format and fix validation errors"
            )
        
        # Quality analysis on valid games
        if valid_games:
            self._analyze_game_data_quality(valid_games, report)
        
        # Calculate final scores
        report.completeness_score = self._calculate_completeness_score(valid_games)
        report.accuracy_score = self._calculate_accuracy_score(valid_games, report)
        report.consistency_score = self._calculate_consistency_score(valid_games, report)
        report.overall_score = (report.completeness_score + report.accuracy_score + report.consistency_score) / 3
        
        # Add recommendations based on scores
        self._generate_recommendations(report)
        
        # Set validation duration
        report.validation_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return report
    
    def validate_team_rankings(self, rankings_data: List[Dict[str, Any]]) -> DataQualityReport:
        """
        Validate team rankings data with quality analysis.
        
        Args:
            rankings_data: List of team ranking dictionaries
            
        Returns:
            Comprehensive quality report
        """
        start_time = datetime.now()
        
        report = DataQualityReport(
            dataset_name="Team Rankings Data",
            total_records=len(rankings_data),
            valid_records=0,
            invalid_records=0,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            overall_score=0.0
        )
        
        if not rankings_data:
            report.add_issue(
                DataQualityLevel.CRITICAL,
                "data_availability", 
                "No ranking data provided",
                0,
                recommended_action="Verify power ranking calculation and data availability"
            )
            return report
        
        # Validate each ranking record
        valid_rankings = []
        validation_errors = []
        
        for i, ranking_data in enumerate(rankings_data):
            try:
                validated_ranking, validation_result = self.schema_validator.validate_team_ranking(ranking_data)
                if validated_ranking:
                    valid_rankings.append(validated_ranking)
                    report.valid_records += 1
                else:
                    report.invalid_records += 1
                    validation_errors.extend(validation_result.errors)
            except Exception as e:
                report.invalid_records += 1
                validation_errors.append(f"Ranking {i}: {str(e)}")
        
        # Analyze validation errors
        if validation_errors:
            report.add_issue(
                DataQualityLevel.HIGH,
                "schema_validation",
                f"Schema validation failed for {report.invalid_records} records",
                report.invalid_records,
                details={'sample_errors': validation_errors[:5]},
                recommended_action="Review ranking data format and calculation logic"
            )
        
        # Quality analysis on valid rankings
        if valid_rankings:
            self._analyze_rankings_quality(valid_rankings, report)
        
        # Calculate final scores
        report.completeness_score = self._calculate_rankings_completeness(valid_rankings)
        report.accuracy_score = self._calculate_rankings_accuracy(valid_rankings, report)
        report.consistency_score = self._calculate_rankings_consistency(valid_rankings, report)
        report.overall_score = (report.completeness_score + report.accuracy_score + report.consistency_score) / 3
        
        # Add recommendations
        self._generate_rankings_recommendations(report, valid_rankings)
        
        # Set validation duration
        report.validation_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return report
    
    def _analyze_game_data_quality(self, games: List[ESPNGameData], report: DataQualityReport) -> None:
        """Analyze game data for quality issues."""
        
        # Check for unusual score patterns
        completed_games = [g for g in games if g.status.value == "completed" and g.home_score is not None and g.away_score is not None]
        
        if completed_games:
            total_points = [g.total_points for g in completed_games if g.total_points]
            margins = [abs(g.margin) for g in completed_games if g.margin is not None]
            
            # Unusual scoring patterns
            if total_points:
                avg_total = np.mean(total_points)
                high_scoring = sum(1 for pts in total_points if pts > 60)
                low_scoring = sum(1 for pts in total_points if pts < 20)
                
                if high_scoring > len(total_points) * 0.1:  # More than 10% high-scoring
                    report.add_issue(
                        DataQualityLevel.MEDIUM,
                        "scoring_patterns",
                        f"{high_scoring} games with unusually high scores (>60 points)",
                        high_scoring,
                        field_name="scores",
                        details={'avg_total_points': avg_total},
                        recommended_action="Verify scoring accuracy for high-scoring games"
                    )
                
                if low_scoring > len(total_points) * 0.05:  # More than 5% low-scoring  
                    report.add_issue(
                        DataQualityLevel.MEDIUM,
                        "scoring_patterns",
                        f"{low_scoring} games with unusually low scores (<20 points)",
                        low_scoring,
                        field_name="scores",
                        recommended_action="Verify scoring accuracy for low-scoring games"
                    )
            
            # Large margins
            if margins:
                large_margins = sum(1 for margin in margins if margin > 30)
                if large_margins > 0:
                    report.add_issue(
                        DataQualityLevel.LOW,
                        "competitive_balance",
                        f"{large_margins} games with margins >30 points",
                        large_margins,
                        field_name="margin",
                        details={'large_margin_count': large_margins}
                    )
        
        # Check for data completeness
        missing_scores = sum(1 for g in games if g.status.value == "completed" and (g.home_score is None or g.away_score is None))
        if missing_scores > 0:
            report.add_issue(
                DataQualityLevel.HIGH,
                "data_completeness",
                f"{missing_scores} completed games missing scores",
                missing_scores,
                field_name="scores",
                recommended_action="Update completed games with final scores"
            )
        
        # Check for date consistency
        week_dates = {}
        for game in games:
            if game.week not in week_dates:
                week_dates[game.week] = []
            week_dates[game.week].append(game.date)
        
        for week, dates in week_dates.items():
            date_range = max(dates) - min(dates)
            if date_range.days > 7:
                report.add_issue(
                    DataQualityLevel.MEDIUM,
                    "temporal_consistency",
                    f"Week {week} games span {date_range.days} days",
                    len(dates),
                    field_name="date",
                    recommended_action="Verify week assignments for games with unusual dates"
                )
    
    def _analyze_rankings_quality(self, rankings: List[TeamRanking], report: DataQualityReport) -> None:
        """Analyze team rankings for quality issues."""
        
        # Check ranking sequence
        ranks = [r.rank for r in rankings]
        expected_ranks = list(range(1, len(rankings) + 1))
        
        if sorted(ranks) != expected_ranks:
            missing_ranks = set(expected_ranks) - set(ranks)
            duplicate_ranks = [r for r in set(ranks) if ranks.count(r) > 1]
            
            if missing_ranks:
                report.add_issue(
                    DataQualityLevel.CRITICAL,
                    "ranking_integrity",
                    f"Missing ranks: {sorted(missing_ranks)}",
                    len(missing_ranks),
                    field_name="rank",
                    recommended_action="Ensure all ranks from 1-32 are assigned exactly once"
                )
            
            if duplicate_ranks:
                report.add_issue(
                    DataQualityLevel.CRITICAL,
                    "ranking_integrity", 
                    f"Duplicate ranks: {sorted(duplicate_ranks)}",
                    len(duplicate_ranks),
                    field_name="rank",
                    recommended_action="Resolve duplicate rank assignments"
                )
        
        # Check power score distribution
        power_scores = [r.power_score for r in rankings]
        power_std = np.std(power_scores)
        
        if power_std < 5:
            report.add_issue(
                DataQualityLevel.MEDIUM,
                "score_distribution",
                f"Low power score variance (std={power_std:.1f})",
                len(rankings),
                field_name="power_score",
                details={'std_deviation': power_std},
                recommended_action="Review model parameters to ensure adequate team differentiation"
            )
        elif power_std > 25:
            report.add_issue(
                DataQualityLevel.MEDIUM,
                "score_distribution",
                f"High power score variance (std={power_std:.1f})",
                len(rankings),
                field_name="power_score",
                details={'std_deviation': power_std},
                recommended_action="Review for potential outliers or calculation errors"
            )
        
        # Check rank-score correlation
        rank_score_correlation = np.corrcoef(ranks, power_scores)[0, 1]
        if rank_score_correlation > -0.9:  # Should be strongly negative
            report.add_issue(
                DataQualityLevel.HIGH,
                "rank_consistency",
                f"Weak rank-score correlation: {rank_score_correlation:.3f}",
                len(rankings),
                field_name="rank",
                details={'correlation': rank_score_correlation},
                recommended_action="Verify ranking algorithm - scores and ranks should be inversely correlated"
            )
    
    def _calculate_completeness_score(self, games: List[ESPNGameData]) -> float:
        """Calculate data completeness score."""
        if not games:
            return 0.0
        
        # Check completeness of key fields
        completeness_checks = []
        
        for game in games:
            checks = [
                game.game_id is not None,
                game.home_team is not None,
                game.away_team is not None,
                game.week is not None,
                game.date is not None,
            ]
            
            # Add score completeness for completed games
            if game.status.value == "completed":
                checks.extend([
                    game.home_score is not None,
                    game.away_score is not None
                ])
            
            completeness_checks.append(sum(checks) / len(checks))
        
        return np.mean(completeness_checks)
    
    def _calculate_accuracy_score(self, games: List[ESPNGameData], report: DataQualityReport) -> float:
        """Calculate data accuracy score based on issues found."""
        if not games:
            return 0.0
        
        # Start with perfect score and deduct for issues
        accuracy_score = 1.0
        
        # Deduct for critical and high issues
        for issue in report.issues:
            if issue.level == DataQualityLevel.CRITICAL:
                accuracy_score -= 0.2
            elif issue.level == DataQualityLevel.HIGH:
                accuracy_score -= 0.1
            elif issue.level == DataQualityLevel.MEDIUM:
                accuracy_score -= 0.05
        
        return max(0.0, accuracy_score)
    
    def _calculate_consistency_score(self, games: List[ESPNGameData], report: DataQualityReport) -> float:
        """Calculate data consistency score."""
        if not games:
            return 0.0
        
        consistency_score = 1.0
        
        # Check temporal consistency
        temporal_issues = [i for i in report.issues if i.category == "temporal_consistency"]
        consistency_score -= len(temporal_issues) * 0.1
        
        # Check format consistency
        format_issues = [i for i in report.issues if i.category == "schema_validation"]
        consistency_score -= len(format_issues) * 0.2
        
        return max(0.0, consistency_score)
    
    def _calculate_rankings_completeness(self, rankings: List[TeamRanking]) -> float:
        """Calculate rankings completeness score."""
        if not rankings:
            return 0.0
        
        # Expected to have 32 teams
        expected_teams = 32
        actual_teams = len(rankings)
        
        completeness = min(1.0, actual_teams / expected_teams)
        return completeness
    
    def _calculate_rankings_accuracy(self, rankings: List[TeamRanking], report: DataQualityReport) -> float:
        """Calculate rankings accuracy score."""
        if not rankings:
            return 0.0
        
        accuracy_score = 1.0
        
        # Deduct for issues
        for issue in report.issues:
            if issue.level == DataQualityLevel.CRITICAL:
                accuracy_score -= 0.3
            elif issue.level == DataQualityLevel.HIGH:
                accuracy_score -= 0.15
            elif issue.level == DataQualityLevel.MEDIUM:
                accuracy_score -= 0.1
        
        return max(0.0, accuracy_score)
    
    def _calculate_rankings_consistency(self, rankings: List[TeamRanking], report: DataQualityReport) -> float:
        """Calculate rankings consistency score."""
        if not rankings:
            return 0.0
        
        consistency_score = 1.0
        
        # Check for ranking integrity issues
        integrity_issues = [i for i in report.issues if i.category == "ranking_integrity"]
        consistency_score -= len(integrity_issues) * 0.4
        
        return max(0.0, consistency_score)
    
    def _generate_recommendations(self, report: DataQualityReport) -> None:
        """Generate quality improvement recommendations."""
        if report.overall_score < 0.8:
            report.add_recommendation("Overall data quality is below threshold - implement systematic quality improvements")
        
        if report.completeness_score < 0.9:
            report.add_recommendation("Improve data collection processes to reduce missing data")
        
        if report.accuracy_score < 0.9:
            report.add_recommendation("Implement additional data validation checks")
        
        if report.consistency_score < 0.9:
            report.add_recommendation("Standardize data formats and validation rules")
        
        critical_issues = report.critical_issues
        if critical_issues:
            report.add_recommendation(f"Address {len(critical_issues)} critical issues immediately")
    
    def _generate_rankings_recommendations(self, report: DataQualityReport, rankings: List[TeamRanking]) -> None:
        """Generate rankings-specific recommendations."""
        if len(rankings) != 32:
            report.add_recommendation(f"Ensure all 32 NFL teams are included (currently have {len(rankings)})")
        
        if report.overall_score < 0.8:
            report.add_recommendation("Review power ranking calculation methodology")
        
        # Check for unrealistic power score ranges
        power_scores = [r.power_score for r in rankings]
        if power_scores:
            min_score, max_score = min(power_scores), max(power_scores)
            if max_score - min_score < 10:
                report.add_recommendation("Consider adjusting model parameters to increase team differentiation")
            elif max_score - min_score > 40:
                report.add_recommendation("Review for potential outliers in power score calculations")

def validate_data_quality(data: Union[List[Dict[str, Any]], Dict[str, Any]], 
                         data_type: str = "games",
                         strict_mode: bool = False) -> DataQualityReport:
    """
    Convenience function for data quality validation.
    
    Args:
        data: Data to validate (games or rankings)
        data_type: Type of data ("games" or "rankings")
        strict_mode: Whether to use strict validation
        
    Returns:
        Data quality report
    """
    validator = PydanticDataQualityValidator(strict_mode=strict_mode)
    
    if data_type == "games":
        if isinstance(data, dict):
            data = [data]
        return validator.validate_espn_game_data(data)
    elif data_type == "rankings":
        if isinstance(data, dict):
            data = [data]
        return validator.validate_team_rankings(data)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")