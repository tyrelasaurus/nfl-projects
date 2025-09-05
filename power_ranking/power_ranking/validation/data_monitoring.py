"""
Real-time data quality monitoring system for NFL data collection.
Monitors data quality during API collection and provides alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import threading
import time

from .data_quality import DataQualityValidator, DataQualityReport, DataQualityLevel

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring alert levels."""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class MonitoringAlert:
    """Real-time monitoring alert."""
    level: MonitoringLevel
    message: str
    metric_name: str
    current_value: Any
    expected_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics."""
    data_freshness: float  # Minutes since last update
    completeness_rate: float  # Percentage of expected data received
    error_rate: float  # Percentage of failed requests
    average_response_time: float  # Average API response time in seconds
    data_volume: int  # Number of records processed
    anomaly_count: int  # Number of anomalies detected
    timestamp: datetime = field(default_factory=datetime.now)

class DataQualityMonitor:
    """Real-time data quality monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the monitoring system."""
        self.config = config or self._get_default_config()
        self.validator = DataQualityValidator(config)
        self.alerts: deque = deque(maxlen=1000)  # Store last 1000 alerts
        self.metrics_history: deque = deque(maxlen=100)  # Store last 100 metric snapshots
        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []
        self.running = False
        self.monitor_thread = None
        
        # Monitoring state
        self.last_data_update = datetime.now()
        self.request_times = deque(maxlen=50)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'monitoring_interval': 60,  # seconds
            'freshness_threshold': 300,  # 5 minutes
            'completeness_threshold': 0.95,  # 95%
            'error_rate_threshold': 0.1,  # 10%
            'response_time_threshold': 10.0,  # 10 seconds
            'anomaly_threshold': 5,  # 5 anomalies
            'alert_cooldown': 300,  # 5 minutes between similar alerts
            'max_retries': 3,
            'enable_real_time_alerts': True
        }
    
    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the real-time monitoring system."""
        if self.running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Data quality monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Data quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._calculate_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                alerts = self._check_monitoring_thresholds(metrics)
                for alert in alerts:
                    self._handle_alert(alert)
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config['monitoring_interval'])
    
    def _calculate_current_metrics(self) -> MonitoringMetrics:
        """Calculate current monitoring metrics."""
        now = datetime.now()
        
        # Data freshness
        freshness = (now - self.last_data_update).total_seconds() / 60
        
        # Error rate
        total_requests = sum(self.success_counts.values()) + sum(self.error_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        # Average response time
        avg_response_time = np.mean(list(self.request_times)) if self.request_times else 0
        
        return MonitoringMetrics(
            data_freshness=freshness,
            completeness_rate=1.0,  # Will be updated during validation
            error_rate=error_rate,
            average_response_time=avg_response_time,
            data_volume=total_requests,
            anomaly_count=0,  # Will be updated during validation
            timestamp=now
        )
    
    def _check_monitoring_thresholds(self, metrics: MonitoringMetrics) -> List[MonitoringAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Data freshness alert
        if metrics.data_freshness > self.config['freshness_threshold'] / 60:
            alerts.append(MonitoringAlert(
                level=MonitoringLevel.WARNING,
                message=f"Data is stale (last update: {metrics.data_freshness:.1f} minutes ago)",
                metric_name="data_freshness",
                current_value=metrics.data_freshness,
                expected_value=self.config['freshness_threshold'] / 60,
                details={"threshold_minutes": self.config['freshness_threshold'] / 60}
            ))
        
        # Error rate alert
        if metrics.error_rate > self.config['error_rate_threshold']:
            level = MonitoringLevel.CRITICAL if metrics.error_rate > 0.3 else MonitoringLevel.WARNING
            alerts.append(MonitoringAlert(
                level=level,
                message=f"High error rate: {metrics.error_rate:.1%}",
                metric_name="error_rate",
                current_value=metrics.error_rate,
                expected_value=self.config['error_rate_threshold'],
                details={"error_counts": dict(self.error_counts)}
            ))
        
        # Response time alert
        if metrics.average_response_time > self.config.get('response_time_threshold', 10.0):
            alerts.append(MonitoringAlert(
                level=MonitoringLevel.WARNING,
                message=f"Slow API response times: {metrics.average_response_time:.1f}s",
                metric_name="response_time",
                current_value=metrics.average_response_time,
                expected_value=self.config['response_time_threshold']
            ))
        
        return alerts
    
    def _handle_alert(self, alert: MonitoringAlert):
        """Handle a monitoring alert."""
        # Add to alert history
        self.alerts.append(alert)
        
        # Log the alert
        log_level = logging.CRITICAL if alert.level == MonitoringLevel.CRITICAL else logging.WARNING
        self.logger.log(log_level, f"MONITORING ALERT [{alert.level.value}]: {alert.message}")
        
        # Call registered callbacks
        if self.config['enable_real_time_alerts']:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def record_api_request(self, endpoint: str, success: bool, 
                          response_time: float, error_type: Optional[str] = None):
        """Record an API request for monitoring."""
        if success:
            self.success_counts[endpoint] += 1
            self.request_times.append(response_time)
        else:
            self.error_counts[error_type or "unknown"] += 1
            
        self.last_data_update = datetime.now()
    
    def validate_and_monitor(self, data: pd.DataFrame, 
                           dataset_name: str = "Dataset") -> DataQualityReport:
        """Validate data and update monitoring metrics."""
        # Run data quality validation
        report = self.validator.validate_game_data(data, dataset_name)
        
        # Update monitoring metrics based on validation results
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            latest_metrics.completeness_rate = report.completeness_score
            latest_metrics.anomaly_count = len([i for i in report.issues if i.category == "Anomaly"])
        
        # Generate alerts for data quality issues
        quality_alerts = self._generate_quality_alerts(report)
        for alert in quality_alerts:
            self._handle_alert(alert)
        
        return report
    
    def _generate_quality_alerts(self, report: DataQualityReport) -> List[MonitoringAlert]:
        """Generate monitoring alerts from data quality report."""
        alerts = []
        
        # Critical data quality issues
        critical_issues = [i for i in report.issues if i.level == DataQualityLevel.CRITICAL]
        if critical_issues:
            alerts.append(MonitoringAlert(
                level=MonitoringLevel.CRITICAL,
                message=f"Critical data quality issues detected: {len(critical_issues)} issues",
                metric_name="data_quality",
                current_value=len(critical_issues),
                expected_value=0,
                details={"issues": [i.description for i in critical_issues]}
            ))
        
        # Low overall quality score
        if report.overall_score < 0.7:
            level = MonitoringLevel.CRITICAL if report.overall_score < 0.5 else MonitoringLevel.WARNING
            alerts.append(MonitoringAlert(
                level=level,
                message=f"Low data quality score: {report.overall_score:.1%}",
                metric_name="quality_score",
                current_value=report.overall_score,
                expected_value=0.9,
                details={"completeness": report.completeness_score, "accuracy": report.accuracy_score}
            ))
        
        # High number of invalid records
        if report.invalid_records > report.total_records * 0.1:  # More than 10% invalid
            alerts.append(MonitoringAlert(
                level=MonitoringLevel.WARNING,
                message=f"High number of invalid records: {report.invalid_records}/{report.total_records}",
                metric_name="invalid_records",
                current_value=report.invalid_records,
                expected_value=report.total_records * 0.05  # Expect less than 5%
            ))
        
        return alerts
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get current monitoring dashboard data."""
        if not self.metrics_history:
            return {"status": "No monitoring data available"}
        
        latest_metrics = self.metrics_history[-1]
        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).seconds < 3600]  # Last hour
        
        return {
            "status": "active" if self.running else "inactive",
            "last_update": latest_metrics.timestamp.isoformat(),
            "current_metrics": {
                "data_freshness_minutes": latest_metrics.data_freshness,
                "completeness_rate": latest_metrics.completeness_rate,
                "error_rate": latest_metrics.error_rate,
                "avg_response_time_seconds": latest_metrics.average_response_time,
                "data_volume": latest_metrics.data_volume,
                "anomaly_count": latest_metrics.anomaly_count
            },
            "recent_alerts": {
                "total": len(recent_alerts),
                "critical": len([a for a in recent_alerts if a.level == MonitoringLevel.CRITICAL]),
                "warning": len([a for a in recent_alerts if a.level == MonitoringLevel.WARNING]),
                "latest": [
                    {
                        "level": a.level.value,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat()
                    } for a in recent_alerts[-5:]  # Last 5 alerts
                ]
            },
            "health_status": self._calculate_health_status(latest_metrics, recent_alerts)
        }
    
    def _calculate_health_status(self, metrics: MonitoringMetrics, 
                               recent_alerts: List[MonitoringAlert]) -> str:
        """Calculate overall system health status."""
        critical_alerts = [a for a in recent_alerts if a.level == MonitoringLevel.CRITICAL]
        warning_alerts = [a for a in recent_alerts if a.level == MonitoringLevel.WARNING]
        
        if critical_alerts:
            return "critical"
        elif metrics.error_rate > 0.2 or metrics.data_freshness > 30:  # 30 minutes
            return "degraded"
        elif warning_alerts or metrics.error_rate > 0.05:
            return "warning"
        else:
            return "healthy"
    
    def export_monitoring_report(self, save_path: str, hours_back: int = 24):
        """Export monitoring report for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter data for time period
        relevant_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        relevant_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        
        report_lines = []
        report_lines.append(f"# Data Quality Monitoring Report")
        report_lines.append(f"**Time Period:** Last {hours_back} hours")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        if relevant_metrics:
            latest = relevant_metrics[-1]
            report_lines.append(f"- **System Status:** {self._calculate_health_status(latest, relevant_alerts).upper()}")
            report_lines.append(f"- **Data Freshness:** {latest.data_freshness:.1f} minutes")
            report_lines.append(f"- **Error Rate:** {latest.error_rate:.1%}")
            report_lines.append(f"- **Total Alerts:** {len(relevant_alerts)}")
            report_lines.append("")
        
        # Alerts
        if relevant_alerts:
            report_lines.append("## Alerts")
            for alert in relevant_alerts[-20:]:  # Last 20 alerts
                emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[alert.level.value]
                report_lines.append(f"- **{emoji} {alert.timestamp.strftime('%H:%M:%S')}:** {alert.message}")
            report_lines.append("")
        
        # Metrics trend
        if len(relevant_metrics) > 1:
            report_lines.append("## Metrics Trend")
            error_rates = [m.error_rate for m in relevant_metrics]
            response_times = [m.average_response_time for m in relevant_metrics]
            
            report_lines.append(f"- **Average Error Rate:** {np.mean(error_rates):.1%}")
            report_lines.append(f"- **Average Response Time:** {np.mean(response_times):.1f}s")
            report_lines.append(f"- **Data Points Collected:** {len(relevant_metrics)}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Monitoring report exported to: {save_path}")
        return report_text

# Utility functions for easy integration
def setup_basic_monitoring(config: Optional[Dict[str, Any]] = None) -> DataQualityMonitor:
    """Set up basic monitoring with default logging callback."""
    monitor = DataQualityMonitor(config)
    
    def log_alert_callback(alert: MonitoringAlert):
        level = logging.CRITICAL if alert.level == MonitoringLevel.CRITICAL else logging.WARNING
        logging.getLogger("DataQualityMonitor").log(
            level, f"[{alert.level.value.upper()}] {alert.message}"
        )
    
    monitor.add_alert_callback(log_alert_callback)
    return monitor

def validate_with_monitoring(data: pd.DataFrame, dataset_name: str = "Dataset",
                           monitor: Optional[DataQualityMonitor] = None) -> DataQualityReport:
    """Convenience function to validate data with monitoring."""
    if monitor is None:
        monitor = DataQualityMonitor()
    
    return monitor.validate_and_monitor(data, dataset_name)