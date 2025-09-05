"""
Enhanced ESPN API client with integrated data quality validation.
Extends the base ESPN client with real-time data quality monitoring and validation.
"""

import requests
import time
import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import base ESPN client and validation systems
from .espn_client import ESPNClient
from ..validation.data_quality import DataQualityValidator, DataQualityReport
from ..validation.data_monitoring import DataQualityMonitor, MonitoringAlert
from ..validation.anomaly_detection import AnomalyDetectionEngine, AnomalyDetectionResult

logger = logging.getLogger(__name__)

class EnhancedESPNClient(ESPNClient):
    """Enhanced ESPN API client with integrated data quality assurance."""
    
    def __init__(self, base_url: str = "https://site.api.espn.com/apis/site/v2",
                 enable_monitoring: bool = True,
                 enable_validation: bool = True,
                 validation_config: Optional[Dict] = None):
        """
        Initialize enhanced ESPN client.
        
        Args:
            base_url: ESPN API base URL
            enable_monitoring: Enable real-time monitoring
            enable_validation: Enable data validation
            validation_config: Configuration for validation systems
        """
        super().__init__(base_url)
        
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        
        # Initialize validation systems
        if self.enable_validation:
            self.validator = DataQualityValidator(validation_config)
            self.anomaly_detector = AnomalyDetectionEngine(validation_config)
        
        if self.enable_monitoring:
            self.monitor = DataQualityMonitor(validation_config)
            self.monitor.add_alert_callback(self._handle_monitoring_alert)
            self.monitor.start_monitoring()
        
        # Track request metrics for monitoring
        self.request_history = []
        self.validation_reports = []
        
        logger.info(f"Enhanced ESPN client initialized (monitoring: {enable_monitoring}, validation: {enable_validation})")
    
    def _handle_monitoring_alert(self, alert: MonitoringAlert):
        """Handle monitoring alerts."""
        log_level = logging.CRITICAL if alert.level.value == "critical" else logging.WARNING
        logger.log(log_level, f"Data Quality Alert: {alert.message}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        """Enhanced request method with monitoring integration."""
        start_time = time.time()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        success = False
        error_type = None
        
        try:
            # Call parent method
            response_data = super()._make_request(endpoint, params, retries)
            success = True
            
            # Record successful request
            if self.enable_monitoring:
                response_time = time.time() - start_time
                self.monitor.record_api_request(endpoint, True, response_time)
            
            return response_data
            
        except Exception as e:
            error_type = type(e).__name__
            
            # Record failed request
            if self.enable_monitoring:
                response_time = time.time() - start_time
                self.monitor.record_api_request(endpoint, False, response_time, error_type)
            
            raise
    
    def get_validated_scoreboard(self, week: Optional[int] = None, 
                               season: Optional[int] = None,
                               validate_data: bool = True) -> Tuple[Dict, Optional[DataQualityReport]]:
        """
        Get scoreboard data with integrated validation.
        
        Args:
            week: Week number
            season: Season year
            validate_data: Whether to validate the returned data
            
        Returns:
            Tuple of (scoreboard_data, validation_report)
        """
        logger.info(f"Fetching validated scoreboard data for week {week}, season {season}")
        
        # Get the raw data
        raw_data = self.get_scoreboard(week, season)
        validation_report = None
        
        # Validate if requested
        if validate_data and self.enable_validation:
            try:
                # Convert to DataFrame for validation
                games_df = self._extract_games_dataframe(raw_data)
                
                if not games_df.empty:
                    # Run data quality validation
                    validation_report = self.validator.validate_game_data(
                        games_df, 
                        f"ESPN Scoreboard Week {week} Season {season}"
                    )
                    
                    # Run anomaly detection
                    anomaly_result = self.anomaly_detector.detect_anomalies(
                        games_df,
                        f"ESPN Scoreboard Week {week} Season {season}"
                    )
                    
                    # Store validation results
                    self.validation_reports.append({
                        'timestamp': datetime.now(),
                        'data_quality': validation_report,
                        'anomaly_detection': anomaly_result
                    })
                    
                    # Alert if critical issues found
                    critical_issues = [i for i in validation_report.issues if i.level.value == "critical"]
                    if critical_issues:
                        logger.critical(f"Critical data quality issues found: {len(critical_issues)} issues")
                    
                    # Update monitoring with validation results
                    if self.enable_monitoring:
                        self.monitor.validate_and_monitor(games_df, f"Week {week}")
                    
                else:
                    logger.warning("No game data extracted for validation")
                    
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
        
        return raw_data, validation_report
    
    def get_comprehensive_season_data(self, season: int = 2024,
                                    validate_data: bool = True) -> Tuple[Dict, Optional[DataQualityReport]]:
        """
        Get comprehensive season data with validation.
        
        Args:
            season: Season year
            validate_data: Whether to validate the data
            
        Returns:
            Tuple of (season_data, validation_report)
        """
        logger.info(f"Fetching comprehensive {season} season data with validation")
        
        # Get the raw data using parent method
        raw_data = self.get_last_season_final_rankings()
        validation_report = None
        
        # Validate if requested
        if validate_data and self.enable_validation:
            try:
                # Extract games from the complex ESPN data structure
                games_df = self._extract_games_dataframe(raw_data)
                
                if not games_df.empty:
                    logger.info(f"Extracted {len(games_df)} games for validation")
                    
                    # Run comprehensive data quality validation
                    validation_report = self.validator.validate_game_data(
                        games_df,
                        f"ESPN {season} Complete Season Data"
                    )
                    
                    # Run anomaly detection
                    anomaly_result = self.anomaly_detector.detect_anomalies(
                        games_df,
                        f"ESPN {season} Complete Season Data"
                    )
                    
                    # Store results
                    self.validation_reports.append({
                        'timestamp': datetime.now(),
                        'data_quality': validation_report,
                        'anomaly_detection': anomaly_result,
                        'dataset_type': 'complete_season'
                    })
                    
                    # Generate summary
                    logger.info(f"Season data validation completed:")
                    logger.info(f"  - Overall quality score: {validation_report.overall_score:.1%}")
                    logger.info(f"  - Issues found: {len(validation_report.issues)}")
                    logger.info(f"  - Anomalies detected: {anomaly_result.anomalies_detected}")
                    
                    # Alert on significant issues
                    if validation_report.overall_score < 0.8:
                        logger.warning(f"Season data quality below threshold: {validation_report.overall_score:.1%}")
                    
                else:
                    logger.error("Failed to extract game data for validation")
                    
            except Exception as e:
                logger.error(f"Season data validation failed: {e}")
                import traceback
                traceback.print_exc()
        
        return raw_data, validation_report
    
    def _extract_games_dataframe(self, espn_data: Dict) -> pd.DataFrame:
        """
        Extract game data from ESPN API response and convert to DataFrame.
        
        Args:
            espn_data: Raw ESPN API response
            
        Returns:
            DataFrame with standardized game data
        """
        try:
            events = espn_data.get('events', [])
            if not events:
                logger.warning("No events found in ESPN data")
                return pd.DataFrame()
            
            games_data = []
            
            for event in events:
                try:
                    # Extract basic game information
                    game_id = event.get('id')
                    date = event.get('date')
                    
                    # Extract week information
                    week = event.get('week', {}).get('number', 0)
                    if week == 0:
                        # Try to extract from season type or other sources
                        season_type = event.get('season', {}).get('type', 0)
                        if season_type == 2:  # Regular season
                            # Try to infer week from date or other fields
                            week = 1  # Default fallback
                    
                    # Extract team and score information
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue
                    
                    competition = competitions[0]
                    competitors = competition.get('competitors', [])
                    
                    if len(competitors) != 2:
                        continue
                    
                    # Identify home and away teams
                    home_team = None
                    away_team = None
                    
                    for competitor in competitors:
                        if competitor.get('homeAway') == 'home':
                            home_team = competitor
                        elif competitor.get('homeAway') == 'away':
                            away_team = competitor
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Extract scores
                    home_score = int(home_team.get('score', 0))
                    away_score = int(away_team.get('score', 0))
                    
                    # Extract team names and IDs
                    home_team_name = home_team.get('team', {}).get('displayName', 'Unknown')
                    away_team_name = away_team.get('team', {}).get('displayName', 'Unknown')
                    home_team_id = home_team.get('team', {}).get('id', '')
                    away_team_id = away_team.get('team', {}).get('id', '')
                    
                    # Calculate derived fields
                    margin = home_score - away_score
                    total_points = home_score + away_score
                    home_win = home_score > away_score
                    
                    # Check if game is completed
                    status = event.get('status', {}).get('type', {}).get('name', '')
                    if status != 'STATUS_FINAL':
                        continue  # Only include completed games
                    
                    game_record = {
                        'game_id': game_id,
                        'week': week,
                        'date': date,
                        'home_team_id': home_team_id,
                        'home_team': home_team_name,
                        'home_score': home_score,
                        'away_team_id': away_team_id,
                        'away_team': away_team_name,
                        'away_score': away_score,
                        'home_win': home_win,
                        'margin': margin,
                        'total_points': total_points
                    }
                    
                    games_data.append(game_record)
                    
                except Exception as e:
                    logger.warning(f"Error extracting game data from event {event.get('id', 'unknown')}: {e}")
                    continue
            
            df = pd.DataFrame(games_data)
            logger.info(f"Successfully extracted {len(df)} game records")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract games DataFrame: {e}")
            return pd.DataFrame()
    
    def get_data_quality_summary(self) -> Dict:
        """Get summary of data quality metrics from recent validations."""
        if not self.validation_reports:
            return {"status": "No validation data available"}
        
        recent_reports = self.validation_reports[-10:]  # Last 10 reports
        
        # Calculate averages
        quality_scores = [r['data_quality'].overall_score for r in recent_reports if r['data_quality']]
        anomaly_counts = [r['anomaly_detection'].anomalies_detected for r in recent_reports if r['anomaly_detection']]
        
        summary = {
            "total_validations": len(self.validation_reports),
            "recent_validations": len(recent_reports),
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_anomaly_count": sum(anomaly_counts) / len(anomaly_counts) if anomaly_counts else 0,
            "latest_validation": recent_reports[-1]['timestamp'].isoformat() if recent_reports else None
        }
        
        # Add monitoring data if available
        if self.enable_monitoring:
            monitoring_dashboard = self.monitor.get_monitoring_dashboard()
            summary.update({
                "monitoring_status": monitoring_dashboard.get("status"),
                "health_status": monitoring_dashboard.get("health_status")
            })
        
        return summary
    
    def export_validation_reports(self, save_path: str):
        """Export all validation reports to a comprehensive report."""
        if not self.validation_reports:
            logger.warning("No validation reports to export")
            return
        
        lines = []
        lines.append("# ESPN Data Quality Validation Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Validations: {len(self.validation_reports)}")
        lines.append("")
        
        # Summary statistics
        quality_scores = [r['data_quality'].overall_score for r in self.validation_reports if r['data_quality']]
        anomaly_counts = [r['anomaly_detection'].anomalies_detected for r in self.validation_reports if r['anomaly_detection']]
        
        if quality_scores:
            lines.append("## Summary Statistics")
            lines.append(f"- **Average Quality Score:** {sum(quality_scores) / len(quality_scores):.1%}")
            lines.append(f"- **Best Quality Score:** {max(quality_scores):.1%}")
            lines.append(f"- **Worst Quality Score:** {min(quality_scores):.1%}")
            lines.append(f"- **Total Anomalies Detected:** {sum(anomaly_counts)}")
            lines.append("")
        
        # Recent validations
        lines.append("## Recent Validation Results")
        for i, report in enumerate(self.validation_reports[-5:], 1):  # Last 5 reports
            timestamp = report['timestamp']
            quality_report = report.get('data_quality')
            anomaly_report = report.get('anomaly_detection')
            
            lines.append(f"### {i}. Validation at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if quality_report:
                lines.append(f"- **Quality Score:** {quality_report.overall_score:.1%}")
                lines.append(f"- **Issues Found:** {len(quality_report.issues)}")
            
            if anomaly_report:
                lines.append(f"- **Anomalies Detected:** {anomaly_report.anomalies_detected}")
                lines.append(f"- **Processing Time:** {anomaly_report.processing_time:.2f}s")
            
            lines.append("")
        
        # Monitoring summary
        if self.enable_monitoring:
            dashboard = self.monitor.get_monitoring_dashboard()
            lines.append("## Monitoring Status")
            lines.append(f"- **Status:** {dashboard.get('status', 'unknown')}")
            lines.append(f"- **Health:** {dashboard.get('health_status', 'unknown')}")
            
            current_metrics = dashboard.get('current_metrics', {})
            lines.append(f"- **Data Freshness:** {current_metrics.get('data_freshness_minutes', 0):.1f} minutes")
            lines.append(f"- **Error Rate:** {current_metrics.get('error_rate', 0):.1%}")
            
        report_text = "\n".join(lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Validation reports exported to: {save_path}")
    
    def __del__(self):
        """Cleanup monitoring when client is destroyed."""
        if hasattr(self, 'monitor') and self.monitor:
            try:
                self.monitor.stop_monitoring()
            except:
                pass