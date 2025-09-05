"""
Comprehensive data quality assurance system for NFL power rankings.
Validates data completeness, accuracy, and detects anomalies in ESPN API data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import warnings

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Data quality severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    level: DataQualityLevel
    category: str
    description: str
    affected_records: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    overall_score: float
    issues: List[DataQualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)

class DataQualityValidator:
    """Comprehensive data quality validation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data quality validator.
        
        Args:
            config: Configuration dictionary for validation rules
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'completeness_thresholds': {
                'critical_fields': 1.0,  # 100% required
                'important_fields': 0.95,  # 95% required
                'optional_fields': 0.7   # 70% acceptable
            },
            'score_ranges': {
                'home_score': (0, 70),
                'away_score': (0, 70),
                'margin': (-50, 50)
            },
            'date_ranges': {
                'season_start': '2024-09-01',
                'season_end': '2025-02-15'
            },
            'expected_teams': 32,
            'expected_weeks': 18,
            'games_per_week_range': (14, 16),
            'anomaly_detection': {
                'score_outlier_threshold': 3.0,  # Standard deviations
                'margin_outlier_threshold': 2.5,
                'enable_statistical_outliers': True
            }
        }
    
    def validate_game_data(self, game_data: pd.DataFrame, 
                          dataset_name: str = "NFL Game Data") -> DataQualityReport:
        """
        Validate NFL game data comprehensively.
        
        Args:
            game_data: DataFrame with game data
            dataset_name: Name of the dataset being validated
            
        Returns:
            DataQualityReport with validation results
        """
        self.logger.info(f"Starting data quality validation for {dataset_name}")
        
        issues = []
        recommendations = []
        
        # Basic data structure validation
        issues.extend(self._validate_data_structure(game_data))
        
        # Completeness validation
        completeness_issues, completeness_score = self._validate_completeness(game_data)
        issues.extend(completeness_issues)
        
        # Accuracy validation
        accuracy_issues, accuracy_score = self._validate_accuracy(game_data)
        issues.extend(accuracy_issues)
        
        # Consistency validation
        consistency_issues, consistency_score = self._validate_consistency(game_data)
        issues.extend(consistency_issues)
        
        # Anomaly detection
        anomaly_issues = self._detect_anomalies(game_data)
        issues.extend(anomaly_issues)
        
        # Business rule validation
        business_issues = self._validate_business_rules(game_data)
        issues.extend(business_issues)
        
        # Calculate metrics
        total_records = len(game_data)
        critical_issues = [i for i in issues if i.level == DataQualityLevel.CRITICAL]
        invalid_records = sum(i.affected_records for i in critical_issues)
        valid_records = total_records - invalid_records
        
        # Calculate overall score
        overall_score = (completeness_score + accuracy_score + consistency_score) / 3
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, overall_score)
        
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
        
        self.logger.info(f"Data quality validation completed. Overall score: {overall_score:.2f}")
        return report
    
    def _validate_data_structure(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate basic data structure and required columns."""
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append(DataQualityIssue(
                level=DataQualityLevel.CRITICAL,
                category="Structure",
                description="Dataset is empty",
                affected_records=0,
                details={"expected_columns": ["game_id", "week", "home_team", "away_team"]}
            ))
            return issues
        
        # Required columns
        required_columns = {
            'critical': ['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score'],
            'important': ['date', 'home_win', 'margin'],
            'optional': ['total_points', 'home_team_id', 'away_team_id']
        }
        
        for importance, columns in required_columns.items():
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                level = DataQualityLevel.CRITICAL if importance == 'critical' else DataQualityLevel.HIGH
                issues.append(DataQualityIssue(
                    level=level,
                    category="Structure",
                    description=f"Missing {importance} columns: {missing_columns}",
                    affected_records=len(df),
                    details={"missing_columns": missing_columns, "importance": importance}
                ))
        
        return issues
    
    def _validate_completeness(self, df: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate data completeness."""
        issues = []
        field_scores = []
        
        thresholds = self.config['completeness_thresholds']
        
        # Critical fields (must be 100% complete)
        critical_fields = ['game_id', 'week', 'home_team', 'away_team']
        for field in critical_fields:
            if field in df.columns:
                completeness = 1 - (df[field].isna().sum() / len(df))
                field_scores.append(completeness)
                
                if completeness < thresholds['critical_fields']:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.CRITICAL,
                        category="Completeness",
                        description=f"Critical field '{field}' has {completeness:.1%} completeness (required: 100%)",
                        affected_records=df[field].isna().sum(),
                        details={"field": field, "completeness": completeness, "missing_count": df[field].isna().sum()}
                    ))
        
        # Important fields (95% complete)
        important_fields = ['home_score', 'away_score', 'date', 'margin']
        for field in important_fields:
            if field in df.columns:
                completeness = 1 - (df[field].isna().sum() / len(df))
                field_scores.append(completeness)
                
                if completeness < thresholds['important_fields']:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.HIGH,
                        category="Completeness",
                        description=f"Important field '{field}' has {completeness:.1%} completeness (required: 95%)",
                        affected_records=df[field].isna().sum(),
                        details={"field": field, "completeness": completeness}
                    ))
        
        completeness_score = np.mean(field_scores) if field_scores else 0.0
        return issues, completeness_score
    
    def _validate_accuracy(self, df: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate data accuracy against expected ranges and formats."""
        issues = []
        accuracy_checks = []
        
        # Score range validation
        score_ranges = self.config['score_ranges']
        for field, (min_val, max_val) in score_ranges.items():
            if field in df.columns:
                invalid_scores = df[(df[field] < min_val) | (df[field] > max_val)]
                if not invalid_scores.empty:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.HIGH,
                        category="Accuracy",
                        description=f"Invalid {field} values outside range [{min_val}, {max_val}]",
                        affected_records=len(invalid_scores),
                        details={"field": field, "range": [min_val, max_val], "invalid_values": invalid_scores[field].tolist()[:10]}
                    ))
                    accuracy_checks.append(1 - len(invalid_scores) / len(df))
                else:
                    accuracy_checks.append(1.0)
        
        # Week validation
        if 'week' in df.columns:
            invalid_weeks = df[(df['week'] < 1) | (df['week'] > self.config['expected_weeks'])]
            if not invalid_weeks.empty:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.HIGH,
                    category="Accuracy",
                    description=f"Invalid week numbers (expected 1-{self.config['expected_weeks']})",
                    affected_records=len(invalid_weeks),
                    details={"invalid_weeks": invalid_weeks['week'].unique().tolist()}
                ))
            accuracy_checks.append(1 - len(invalid_weeks) / len(df))
        
        # Date validation
        if 'date' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df[df['date_parsed'].isna() & df['date'].notna()]
                
                season_start = pd.to_datetime(self.config['date_ranges']['season_start'])
                season_end = pd.to_datetime(self.config['date_ranges']['season_end'])
                
                out_of_season = df[
                    (df['date_parsed'] < season_start) | 
                    (df['date_parsed'] > season_end)
                ]
                
                if not invalid_dates.empty:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.MEDIUM,
                        category="Accuracy",
                        description="Invalid date formats detected",
                        affected_records=len(invalid_dates),
                        details={"sample_invalid_dates": invalid_dates['date'].head(5).tolist()}
                    ))
                
                if not out_of_season.empty:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.MEDIUM,
                        category="Accuracy",
                        description="Games outside expected season date range",
                        affected_records=len(out_of_season),
                        details={"season_range": [str(season_start.date()), str(season_end.date())]}
                    ))
                
                accuracy_checks.append(1 - (len(invalid_dates) + len(out_of_season)) / len(df))
                
            except Exception as e:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.MEDIUM,
                    category="Accuracy",
                    description=f"Date validation failed: {str(e)}",
                    affected_records=0,
                    details={"error": str(e)}
                ))
        
        accuracy_score = np.mean(accuracy_checks) if accuracy_checks else 1.0
        return issues, accuracy_score
    
    def _validate_consistency(self, df: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate data consistency and logical relationships."""
        issues = []
        consistency_checks = []
        
        # Check margin calculation consistency
        if all(col in df.columns for col in ['home_score', 'away_score', 'margin']):
            calculated_margin = df['home_score'] - df['away_score']
            inconsistent_margins = df[abs(df['margin'] - calculated_margin) > 0.1]
            
            if not inconsistent_margins.empty:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.HIGH,
                    category="Consistency",
                    description="Margin values inconsistent with score difference",
                    affected_records=len(inconsistent_margins),
                    details={"sample_inconsistencies": inconsistent_margins[['home_score', 'away_score', 'margin']].head(5).to_dict('records')}
                ))
            
            consistency_checks.append(1 - len(inconsistent_margins) / len(df))
        
        # Check home_win consistency
        if all(col in df.columns for col in ['margin', 'home_win']):
            # home_win should be True when margin > 0
            inconsistent_wins = df[
                ((df['margin'] > 0) & (~df['home_win'])) |
                ((df['margin'] < 0) & (df['home_win']))
            ]
            
            if not inconsistent_wins.empty:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.HIGH,
                    category="Consistency",
                    description="home_win values inconsistent with margin",
                    affected_records=len(inconsistent_wins),
                    details={"inconsistent_count": len(inconsistent_wins)}
                ))
            
            consistency_checks.append(1 - len(inconsistent_wins) / len(df))
        
        # Check duplicate game IDs
        if 'game_id' in df.columns:
            duplicate_games = df[df.duplicated(['game_id'], keep=False)]
            if not duplicate_games.empty:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.HIGH,
                    category="Consistency",
                    description="Duplicate game IDs detected",
                    affected_records=len(duplicate_games),
                    details={"duplicate_game_ids": duplicate_games['game_id'].unique().tolist()[:10]}
                ))
            
            consistency_checks.append(1 - len(duplicate_games) / len(df))
        
        # Check team name consistency
        if all(col in df.columns for col in ['home_team', 'away_team']):
            # Teams shouldn't play themselves
            self_games = df[df['home_team'] == df['away_team']]
            if not self_games.empty:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.CRITICAL,
                    category="Consistency",
                    description="Teams playing against themselves detected",
                    affected_records=len(self_games),
                    details={"self_games": self_games[['home_team', 'away_team', 'game_id']].head(5).to_dict('records')}
                ))
            
            consistency_checks.append(1 - len(self_games) / len(df))
        
        consistency_score = np.mean(consistency_checks) if consistency_checks else 1.0
        return issues, consistency_score
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect statistical anomalies in the data."""
        issues = []
        
        if not self.config['anomaly_detection']['enable_statistical_outliers']:
            return issues
        
        # Score outliers
        for score_field in ['home_score', 'away_score']:
            if score_field in df.columns:
                scores = df[score_field].dropna()
                if len(scores) > 10:  # Need sufficient data for outlier detection
                    z_scores = np.abs((scores - scores.mean()) / scores.std())
                    outliers = scores[z_scores > self.config['anomaly_detection']['score_outlier_threshold']]
                    
                    if len(outliers) > 0:
                        issues.append(DataQualityIssue(
                            level=DataQualityLevel.MEDIUM,
                            category="Anomaly",
                            description=f"Statistical outliers detected in {score_field}",
                            affected_records=len(outliers),
                            details={
                                "field": score_field,
                                "outlier_values": outliers.tolist()[:10],
                                "threshold": self.config['anomaly_detection']['score_outlier_threshold'],
                                "mean": scores.mean(),
                                "std": scores.std()
                            }
                        ))
        
        # Margin outliers
        if 'margin' in df.columns:
            margins = df['margin'].dropna()
            if len(margins) > 10:
                z_scores = np.abs((margins - margins.mean()) / margins.std())
                outliers = margins[z_scores > self.config['anomaly_detection']['margin_outlier_threshold']]
                
                if len(outliers) > 0:
                    issues.append(DataQualityIssue(
                        level=DataQualityLevel.MEDIUM,
                        category="Anomaly",
                        description="Statistical outliers detected in margin",
                        affected_records=len(outliers),
                        details={
                            "outlier_values": outliers.tolist()[:10],
                            "threshold": self.config['anomaly_detection']['margin_outlier_threshold']
                        }
                    ))
        
        return issues
    
    def _validate_business_rules(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate NFL-specific business rules."""
        issues = []
        
        # Check expected number of teams
        if 'home_team' in df.columns and 'away_team' in df.columns:
            all_teams = set(df['home_team'].dropna()) | set(df['away_team'].dropna())
            if len(all_teams) != self.config['expected_teams']:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.MEDIUM,
                    category="Business Rule",
                    description=f"Expected {self.config['expected_teams']} teams, found {len(all_teams)}",
                    affected_records=0,
                    details={"teams_found": len(all_teams), "team_names": sorted(list(all_teams))}
                ))
        
        # Check games per week distribution
        if 'week' in df.columns:
            games_per_week = df.groupby('week').size()
            expected_range = self.config['games_per_week_range']
            
            unusual_weeks = games_per_week[
                (games_per_week < expected_range[0]) | 
                (games_per_week > expected_range[1])
            ]
            
            if len(unusual_weeks) > 0:
                issues.append(DataQualityIssue(
                    level=DataQualityLevel.LOW,
                    category="Business Rule",
                    description="Unusual number of games per week detected",
                    affected_records=0,
                    details={
                        "expected_range": expected_range,
                        "unusual_weeks": unusual_weeks.to_dict()
                    }
                ))
        
        return issues
    
    def _generate_recommendations(self, issues: List[DataQualityIssue], 
                                overall_score: float) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        
        # Critical issues recommendations
        critical_issues = [i for i in issues if i.level == DataQualityLevel.CRITICAL]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address critical data quality issues immediately before using data for analysis")
            for issue in critical_issues:
                recommendations.append(f"   - Fix: {issue.description}")
        
        # High priority recommendations
        high_issues = [i for i in issues if i.level == DataQualityLevel.HIGH]
        if high_issues:
            recommendations.append("⚠️ HIGH PRIORITY: Address these issues to improve data reliability")
            for issue in high_issues:
                recommendations.append(f"   - {issue.description}")
        
        # Overall score recommendations
        if overall_score < 0.7:
            recommendations.append("📊 Data quality score is below acceptable threshold (70%)")
            recommendations.append("   - Implement automated data validation in your data pipeline")
            recommendations.append("   - Add data quality monitoring and alerting")
        elif overall_score < 0.9:
            recommendations.append("📈 Data quality is acceptable but has room for improvement")
            recommendations.append("   - Consider implementing additional validation rules")
        else:
            recommendations.append("✅ Excellent data quality! Consider this validation framework for ongoing monitoring")
        
        # Specific category recommendations
        categories = {issue.category for issue in issues}
        if 'Completeness' in categories:
            recommendations.append("🔍 Improve data collection processes to reduce missing values")
        if 'Accuracy' in categories:
            recommendations.append("🎯 Review data source accuracy and implement range validations")
        if 'Consistency' in categories:
            recommendations.append("🔄 Add data consistency checks in your ETL pipeline")
        if 'Anomaly' in categories:
            recommendations.append("📈 Investigate anomalous values - they may indicate data collection issues")
        
        return recommendations
    
    def generate_data_quality_report(self, report: DataQualityReport, 
                                   save_path: Optional[str] = None) -> str:
        """Generate a comprehensive data quality report in Markdown format."""
        lines = []
        
        # Header
        lines.append(f"# Data Quality Assessment Report")
        lines.append(f"**Dataset:** {report.dataset_name}")
        lines.append(f"**Generated:** {report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append(f"- **Total Records:** {report.total_records:,}")
        lines.append(f"- **Valid Records:** {report.valid_records:,}")
        lines.append(f"- **Invalid Records:** {report.invalid_records:,}")
        lines.append(f"- **Overall Quality Score:** {report.overall_score:.1%}")
        lines.append("")
        
        # Quality Metrics
        lines.append("## Quality Metrics")
        lines.append("| Metric | Score | Status |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| Completeness | {report.completeness_score:.1%} | {'✅' if report.completeness_score >= 0.9 else '⚠️' if report.completeness_score >= 0.7 else '❌'} |")
        lines.append(f"| Accuracy | {report.accuracy_score:.1%} | {'✅' if report.accuracy_score >= 0.9 else '⚠️' if report.accuracy_score >= 0.7 else '❌'} |")
        lines.append(f"| Consistency | {report.consistency_score:.1%} | {'✅' if report.consistency_score >= 0.9 else '⚠️' if report.consistency_score >= 0.7 else '❌'} |")
        lines.append(f"| **Overall** | **{report.overall_score:.1%}** | **{'✅' if report.overall_score >= 0.9 else '⚠️' if report.overall_score >= 0.7 else '❌'}** |")
        lines.append("")
        
        # Issues by Category
        if report.issues:
            lines.append("## Issues Identified")
            
            # Group issues by category and level
            by_category = {}
            for issue in report.issues:
                if issue.category not in by_category:
                    by_category[issue.category] = []
                by_category[issue.category].append(issue)
            
            for category, issues in by_category.items():
                lines.append(f"### {category} Issues")
                for issue in issues:
                    emoji = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "🔵", "info": "ℹ️"}[issue.level.value]
                    lines.append(f"- **{emoji} {issue.level.value.upper()}:** {issue.description}")
                    lines.append(f"  - Records affected: {issue.affected_records:,}")
                    if issue.details:
                        lines.append(f"  - Details: {json.dumps(issue.details, indent=2, default=str)[:200]}{'...' if len(str(issue.details)) > 200 else ''}")
                lines.append("")
        else:
            lines.append("## Issues Identified")
            lines.append("✅ **No data quality issues detected!**")
            lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Summary
        lines.append("## Summary")
        if report.overall_score >= 0.9:
            lines.append("🎯 **Excellent Data Quality** - This dataset meets high quality standards and is suitable for analysis and modeling.")
        elif report.overall_score >= 0.7:
            lines.append("✅ **Good Data Quality** - This dataset is suitable for analysis with minor improvements recommended.")
        elif report.overall_score >= 0.5:
            lines.append("⚠️ **Moderate Data Quality** - This dataset has quality issues that should be addressed before critical analysis.")
        else:
            lines.append("❌ **Poor Data Quality** - This dataset has significant quality issues and should not be used for analysis without major corrections.")
        
        report_text = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Data quality report saved to: {save_path}")
        
        return report_text