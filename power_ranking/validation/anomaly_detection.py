"""
Advanced anomaly detection system for NFL data.
Detects statistical, temporal, and business rule anomalies in real-time.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    BUSINESS_RULE = "business_rule"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"

class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    affected_records: List[Dict[str, Any]]
    detection_method: str
    confidence_score: float  # 0-1, higher means more confident
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyDetectionResult:
    """Results from anomaly detection analysis."""
    dataset_name: str
    total_records: int
    anomalies_detected: int
    anomalies: List[Anomaly]
    detection_summary: Dict[str, int]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

class AnomalyDetectionEngine:
    """Advanced anomaly detection engine for NFL data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize anomaly detection engine.
        
        Args:
            config: Configuration for detection parameters
        """
        self.config = config or self._get_default_config()
        self.historical_data = deque(maxlen=1000)  # Store recent data for baseline
        self.learned_patterns = {}
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default anomaly detection configuration."""
        return {
            'statistical_detection': {
                'z_score_threshold': 3.0,
                'iqr_multiplier': 1.5,
                'modified_z_score_threshold': 3.5,
                'enable_multivariate': True
            },
            'temporal_detection': {
                'seasonal_detection': True,
                'trend_detection': True,
                'sudden_change_threshold': 2.0,
                'week_comparison_enabled': True
            },
            'business_rules': {
                'max_score_nfl': 70,
                'min_score_nfl': 0,
                'max_margin': 50,
                'min_margin': -50,
                'reasonable_total_range': (6, 100),
                'overtime_score_patterns': True
            },
            'contextual_detection': {
                'team_performance_baseline': True,
                'home_away_patterns': True,
                'division_rivalry_patterns': True,
                'weather_impact_detection': False  # Would need weather data
            },
            'confidence_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
        }
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        dataset_name: str = "NFL Dataset") -> AnomalyDetectionResult:
        """
        Detect anomalies in the provided dataset.
        
        Args:
            data: DataFrame containing NFL game data
            dataset_name: Name of the dataset
            
        Returns:
            AnomalyDetectionResult with all detected anomalies
        """
        start_time = datetime.now()
        self.logger.info(f"Starting anomaly detection on {len(data)} records")
        
        all_anomalies = []
        
        # Statistical anomaly detection
        if self.config['statistical_detection']['enable_multivariate']:
            statistical_anomalies = self._detect_statistical_anomalies(data)
            all_anomalies.extend(statistical_anomalies)
        
        # Temporal anomaly detection
        if self.config['temporal_detection']['seasonal_detection']:
            temporal_anomalies = self._detect_temporal_anomalies(data)
            all_anomalies.extend(temporal_anomalies)
        
        # Business rule anomaly detection
        business_anomalies = self._detect_business_rule_anomalies(data)
        all_anomalies.extend(business_anomalies)
        
        # Contextual anomaly detection
        contextual_anomalies = self._detect_contextual_anomalies(data)
        all_anomalies.extend(contextual_anomalies)
        
        # Collective anomaly detection
        collective_anomalies = self._detect_collective_anomalies(data)
        all_anomalies.extend(collective_anomalies)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create summary
        detection_summary = defaultdict(int)
        for anomaly in all_anomalies:
            detection_summary[f"{anomaly.anomaly_type.value}_{anomaly.severity.value}"] += 1
        
        result = AnomalyDetectionResult(
            dataset_name=dataset_name,
            total_records=len(data),
            anomalies_detected=len(all_anomalies),
            anomalies=all_anomalies,
            detection_summary=dict(detection_summary),
            processing_time=processing_time
        )
        
        self.logger.info(f"Anomaly detection completed: {len(all_anomalies)} anomalies found in {processing_time:.2f}s")
        return result
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect statistical outliers using multiple methods."""
        anomalies = []
        
        numerical_columns = ['home_score', 'away_score', 'margin', 'total_points']
        available_columns = [col for col in numerical_columns if col in data.columns]
        
        for column in available_columns:
            values = data[column].dropna()
            if len(values) < 10:  # Need sufficient data
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            outlier_mask = z_scores > self.config['statistical_detection']['z_score_threshold']
            z_outliers = data.loc[values.index[outlier_mask]]
            
            if not z_outliers.empty:
                confidence = min(0.95, np.max(z_scores[z_scores > self.config['statistical_detection']['z_score_threshold']]) / 5)
                severity = self._determine_severity(confidence)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    description=f"Statistical outliers in {column} (Z-score method)",
                    affected_records=z_outliers.to_dict('records')[:10],  # Limit to 10 records
                    detection_method="z_score",
                    confidence_score=confidence,
                    context={
                        "column": column,
                        "threshold": self.config['statistical_detection']['z_score_threshold'],
                        "outlier_count": len(z_outliers),
                        "mean": values.mean(),
                        "std": values.std()
                    }
                ))
            
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.config['statistical_detection']['iqr_multiplier']
            
            iqr_outliers = data[
                (data[column] < Q1 - multiplier * IQR) | 
                (data[column] > Q3 + multiplier * IQR)
            ]
            
            if not iqr_outliers.empty:
                confidence = 0.8  # IQR is generally reliable
                severity = self._determine_severity(confidence)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    description=f"IQR outliers in {column}",
                    affected_records=iqr_outliers.to_dict('records')[:10],
                    detection_method="iqr",
                    confidence_score=confidence,
                    context={
                        "column": column,
                        "Q1": Q1,
                        "Q3": Q3,
                        "IQR": IQR,
                        "multiplier": multiplier,
                        "outlier_count": len(iqr_outliers)
                    }
                ))
        
        return anomalies
    
    def _detect_temporal_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect temporal anomalies and unusual patterns."""
        anomalies = []
        
        if 'week' not in data.columns:
            return anomalies
        
        # Week-by-week analysis
        weekly_stats = data.groupby('week').agg({
            'home_score': ['mean', 'std', 'count'],
            'away_score': ['mean', 'std', 'count'],
            'margin': ['mean', 'std'],
            'total_points': ['mean', 'std'] if 'total_points' in data.columns else None
        }).dropna()
        
        # Detect sudden changes between weeks
        for metric in ['home_score', 'away_score', 'margin']:
            if (metric, 'mean') in weekly_stats.columns:
                means = weekly_stats[(metric, 'mean')]
                
                # Calculate week-to-week changes
                week_changes = means.diff().abs()
                threshold = self.config['temporal_detection']['sudden_change_threshold'] * means.std()
                
                sudden_changes = week_changes[week_changes > threshold]
                
                if not sudden_changes.empty:
                    confidence = min(0.9, (sudden_changes.max() / threshold) / 3)
                    severity = self._determine_severity(confidence)
                    
                    affected_weeks = sudden_changes.index.tolist()
                    affected_records = data[data['week'].isin(affected_weeks)]
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.TEMPORAL,
                        severity=severity,
                        description=f"Sudden changes in {metric} between weeks",
                        affected_records=affected_records.to_dict('records')[:10],
                        detection_method="week_to_week_change",
                        confidence_score=confidence,
                        context={
                            "metric": metric,
                            "affected_weeks": affected_weeks,
                            "threshold": threshold,
                            "max_change": sudden_changes.max()
                        }
                    ))
        
        return anomalies
    
    def _detect_business_rule_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect violations of NFL business rules."""
        anomalies = []
        
        # Score range violations
        score_columns = ['home_score', 'away_score']
        for col in score_columns:
            if col in data.columns:
                invalid_scores = data[
                    (data[col] < self.config['business_rules']['min_score_nfl']) |
                    (data[col] > self.config['business_rules']['max_score_nfl'])
                ]
                
                if not invalid_scores.empty:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.BUSINESS_RULE,
                        severity=AnomalySeverity.HIGH,
                        description=f"Invalid NFL scores in {col}",
                        affected_records=invalid_scores.to_dict('records'),
                        detection_method="score_range_validation",
                        confidence_score=1.0,  # Business rules are definitive
                        context={
                            "column": col,
                            "valid_range": [
                                self.config['business_rules']['min_score_nfl'],
                                self.config['business_rules']['max_score_nfl']
                            ],
                            "violation_count": len(invalid_scores)
                        }
                    ))
        
        # Margin validation
        if 'margin' in data.columns:
            invalid_margins = data[
                (data['margin'] < self.config['business_rules']['min_margin']) |
                (data['margin'] > self.config['business_rules']['max_margin'])
            ]
            
            if not invalid_margins.empty:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BUSINESS_RULE,
                    severity=AnomalySeverity.MEDIUM,
                    description="Extreme margin values detected",
                    affected_records=invalid_margins.to_dict('records'),
                    detection_method="margin_validation",
                    confidence_score=0.9,
                    context={
                        "valid_range": [
                            self.config['business_rules']['min_margin'],
                            self.config['business_rules']['max_margin']
                        ]
                    }
                ))
        
        # Total points validation
        if 'total_points' in data.columns:
            min_total, max_total = self.config['business_rules']['reasonable_total_range']
            invalid_totals = data[
                (data['total_points'] < min_total) |
                (data['total_points'] > max_total)
            ]
            
            if not invalid_totals.empty:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BUSINESS_RULE,
                    severity=AnomalySeverity.MEDIUM,
                    description="Unusual total points detected",
                    affected_records=invalid_totals.to_dict('records'),
                    detection_method="total_points_validation",
                    confidence_score=0.8,
                    context={"reasonable_range": [min_total, max_total]}
                ))
        
        # Check for impossible combinations
        if all(col in data.columns for col in ['home_team', 'away_team']):
            self_games = data[data['home_team'] == data['away_team']]
            if not self_games.empty:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BUSINESS_RULE,
                    severity=AnomalySeverity.CRITICAL,
                    description="Teams playing against themselves",
                    affected_records=self_games.to_dict('records'),
                    detection_method="self_game_validation",
                    confidence_score=1.0,
                    context={"impossible_games": len(self_games)}
                ))
        
        return anomalies
    
    def _detect_contextual_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect contextual anomalies based on team performance patterns."""
        anomalies = []
        
        # Home field advantage anomalies
        if self.config['contextual_detection']['home_away_patterns']:
            if all(col in data.columns for col in ['margin', 'home_win']):
                # Calculate typical home advantage
                home_wins = data[data['home_win'] == True]
                away_wins = data[data['home_win'] == False]
                
                if len(home_wins) > 0 and len(away_wins) > 0:
                    avg_home_margin = home_wins['margin'].mean()
                    avg_away_margin = away_wins['margin'].mean()
                    
                    # Look for games where home team lost by unusually large margin
                    unusual_home_losses = data[
                        (data['home_win'] == False) & 
                        (data['margin'] < -20)  # Home team lost by more than 20
                    ]
                    
                    if not unusual_home_losses.empty:
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.CONTEXTUAL,
                            severity=AnomalySeverity.MEDIUM,
                            description="Unusually large home team losses",
                            affected_records=unusual_home_losses.to_dict('records'),
                            detection_method="home_advantage_violation",
                            confidence_score=0.7,
                            context={
                                "typical_home_advantage": avg_home_margin,
                                "unusual_loss_count": len(unusual_home_losses)
                            }
                        ))
        
        return anomalies
    
    def _detect_collective_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect collective anomalies in groups of records."""
        anomalies = []
        
        # Weekly scoring anomalies
        if 'week' in data.columns and 'total_points' in data.columns:
            weekly_totals = data.groupby('week')['total_points'].mean()
            
            if len(weekly_totals) > 3:  # Need multiple weeks for comparison
                # Detect weeks with unusually high or low scoring
                z_scores = np.abs(stats.zscore(weekly_totals))
                unusual_weeks = weekly_totals[z_scores > 2.0]
                
                if not unusual_weeks.empty:
                    affected_records = data[data['week'].isin(unusual_weeks.index)]
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.COLLECTIVE,
                        severity=AnomalySeverity.LOW,
                        description="Weeks with unusual scoring patterns",
                        affected_records=affected_records.to_dict('records')[:20],
                        detection_method="weekly_scoring_pattern",
                        confidence_score=0.6,
                        context={
                            "unusual_weeks": unusual_weeks.to_dict(),
                            "overall_mean": weekly_totals.mean(),
                            "overall_std": weekly_totals.std()
                        }
                    ))
        
        return anomalies
    
    def _determine_severity(self, confidence_score: float) -> AnomalySeverity:
        """Determine anomaly severity based on confidence score."""
        thresholds = self.config['confidence_thresholds']
        
        if confidence_score >= thresholds['critical']:
            return AnomalySeverity.CRITICAL
        elif confidence_score >= thresholds['high']:
            return AnomalySeverity.HIGH
        elif confidence_score >= thresholds['medium']:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def generate_anomaly_report(self, result: AnomalyDetectionResult, 
                               save_path: Optional[str] = None) -> str:
        """Generate comprehensive anomaly detection report."""
        lines = []
        
        # Header
        lines.append(f"# Anomaly Detection Report")
        lines.append(f"**Dataset:** {result.dataset_name}")
        lines.append(f"**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Processing Time:** {result.processing_time:.2f} seconds")
        lines.append("")
        
        # Summary
        lines.append("## Executive Summary")
        lines.append(f"- **Total Records Analyzed:** {result.total_records:,}")
        lines.append(f"- **Anomalies Detected:** {result.anomalies_detected}")
        lines.append(f"- **Detection Rate:** {result.anomalies_detected / result.total_records:.1%}")
        lines.append("")
        
        # Detection Summary
        if result.detection_summary:
            lines.append("## Detection Summary")
            lines.append("| Type | Severity | Count |")
            lines.append("|------|----------|-------|")
            
            for key, count in sorted(result.detection_summary.items()):
                anomaly_type, severity = key.split('_')
                lines.append(f"| {anomaly_type.title()} | {severity.title()} | {count} |")
            lines.append("")
        
        # Detailed Anomalies
        if result.anomalies:
            lines.append("## Detailed Anomaly Analysis")
            
            # Group by type
            by_type = defaultdict(list)
            for anomaly in result.anomalies:
                by_type[anomaly.anomaly_type].append(anomaly)
            
            for anomaly_type, anomalies in by_type.items():
                lines.append(f"### {anomaly_type.value.title()} Anomalies")
                
                for i, anomaly in enumerate(anomalies, 1):
                    emoji = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "🔵"}[anomaly.severity.value]
                    lines.append(f"#### {i}. {emoji} {anomaly.description}")
                    lines.append(f"- **Severity:** {anomaly.severity.value.title()}")
                    lines.append(f"- **Confidence:** {anomaly.confidence_score:.1%}")
                    lines.append(f"- **Detection Method:** {anomaly.detection_method}")
                    lines.append(f"- **Affected Records:** {len(anomaly.affected_records)}")
                    
                    if anomaly.context:
                        lines.append(f"- **Context:** {json.dumps(anomaly.context, indent=2, default=str)[:300]}{'...' if len(str(anomaly.context)) > 300 else ''}")
                    
                    lines.append("")
                
                lines.append("")
        else:
            lines.append("## Detailed Anomaly Analysis")
            lines.append("✅ **No anomalies detected in this dataset!**")
            lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        
        critical_anomalies = [a for a in result.anomalies if a.severity == AnomalySeverity.CRITICAL]
        high_anomalies = [a for a in result.anomalies if a.severity == AnomalySeverity.HIGH]
        
        if critical_anomalies:
            lines.append("🚨 **CRITICAL:** Immediate action required:")
            for anomaly in critical_anomalies:
                lines.append(f"   - {anomaly.description}")
        
        if high_anomalies:
            lines.append("⚠️ **HIGH PRIORITY:** Review and investigate:")
            for anomaly in high_anomalies:
                lines.append(f"   - {anomaly.description}")
        
        if not critical_anomalies and not high_anomalies:
            lines.append("✅ No critical or high-priority anomalies requiring immediate attention.")
        
        lines.append("")
        lines.append("💡 **General Recommendations:**")
        lines.append("- Implement automated anomaly monitoring for ongoing data quality assurance")
        lines.append("- Review data collection processes for any systemic issues")
        lines.append("- Consider setting up alerts for critical anomaly detection")
        
        report_text = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Anomaly detection report saved to: {save_path}")
        
        return report_text