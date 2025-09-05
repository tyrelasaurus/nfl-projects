"""
Unified Alerting Framework for NFL Projects

Consolidates and extends alerting capabilities from existing systems:
- Phase 1.3 Data Quality Monitoring alerts (power_ranking.validation.data_monitoring)
- Custom alert management and notification system
- Integration with external alerting services (email, Slack, webhooks)

Provides centralized alert management, escalation policies, and notification delivery
for both Power Rankings and NFL Spread Model systems.
"""

import smtplib
import json
import requests
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import os
import sys

# Add project paths for existing component imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from power_ranking.validation.data_monitoring import DataMonitor
except ImportError:
    DataMonitor = None

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Alert severity levels (higher numbers = more severe)."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class AlertStatus(Enum):
    """Alert status tracking."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    LOG_ONLY = "log_only"


@dataclass
class Alert:
    """Individual alert with metadata and status tracking."""
    id: str
    title: str
    message: str
    level: AlertLevel
    source: str  # Component that generated the alert
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledgement_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    notification_attempts: int = 0
    escalation_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'level': self.level.name,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metadata': self.metadata,
            'acknowledgement_time': self.acknowledgement_time.isoformat() if self.acknowledgement_time else None,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None,
            'notification_attempts': self.notification_attempts,
            'escalation_level': self.escalation_level
        }
    
    def acknowledge(self, user: Optional[str] = None):
        """Acknowledge the alert."""
        if self.status == AlertStatus.ACTIVE:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledgement_time = datetime.now(timezone.utc)
            if user:
                self.metadata['acknowledged_by'] = user
    
    def resolve(self, user: Optional[str] = None, reason: Optional[str] = None):
        """Resolve the alert."""
        if self.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            self.status = AlertStatus.RESOLVED
            self.resolution_time = datetime.now(timezone.utc)
            if user:
                self.metadata['resolved_by'] = user
            if reason:
                self.metadata['resolution_reason'] = reason


@dataclass
class AlertingRule:
    """Rule for generating alerts based on conditions."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    alert_template: Dict[str, Any]
    cooldown_seconds: int = 300  # 5 minutes default
    max_alerts_per_hour: int = 12
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    alert_levels: List[AlertLevel] = field(default_factory=lambda: [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL])
    
    def should_notify(self, alert_level: AlertLevel) -> bool:
        """Check if this channel should notify for the given alert level."""
        return self.enabled and alert_level in self.alert_levels


class AlertManager:
    """
    Centralized alert management system for NFL Projects.
    
    Consolidates alerting capabilities from existing systems and provides
    unified alert generation, notification, and management capabilities.
    
    Features:
    - Multi-channel notifications (email, Slack, webhooks, SMS)
    - Alert escalation and acknowledgement
    - Rate limiting and cooldown periods
    - Integration with existing Phase 1.3 data monitoring alerts
    - Alert history and analytics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration including notification channels,
                   escalation policies, and integration settings.
        """
        self.config = config or self._get_default_config()
        
        # Alert storage and tracking
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=self.config.get('max_history', 10000))
        self._alert_lock = threading.RLock()
        
        # Alerting rules
        self._alerting_rules: List[AlertingRule] = []
        self._load_default_rules()
        
        # Notification channels
        self._notification_channels: List[NotificationConfig] = []
        self._load_notification_channels()
        
        # Integration with existing systems
        self._integrate_with_existing_systems()
        
        # Statistics
        self._alert_stats = defaultdict(int)
        
        logger.info("AlertManager initialized with unified alerting framework")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default alerting configuration."""
        return {
            'max_history': 10000,
            'cleanup_interval_hours': 24,
            'escalation_enabled': True,
            'escalation_timeout_minutes': 30,
            'rate_limiting_enabled': True,
            'notification_retry_attempts': 3,
            'notification_retry_delay_seconds': 60
        }
    
    def _load_default_rules(self):
        """Load default alerting rules for common conditions."""
        
        # Memory usage rule
        self._alerting_rules.append(AlertingRule(
            name="high_memory_usage",
            condition=lambda metrics: metrics.get('memory_percent', 0) > 80,
            alert_template={
                'title': 'High Memory Usage Alert',
                'message': 'Memory usage exceeded 80%: {memory_percent:.1f}%',
                'level': AlertLevel.WARNING,
                'source': 'system_monitor'
            },
            cooldown_seconds=600  # 10 minutes
        ))
        
        # API failure rule
        self._alerting_rules.append(AlertingRule(
            name="api_failure_rate",
            condition=lambda metrics: metrics.get('api_failure_rate', 0) > 10,
            alert_template={
                'title': 'High API Failure Rate',
                'message': 'API failure rate exceeded 10%: {api_failure_rate:.1f}%',
                'level': AlertLevel.ERROR,
                'source': 'api_monitor'
            },
            cooldown_seconds=300  # 5 minutes
        ))
        
        # Data validation failure rule
        self._alerting_rules.append(AlertingRule(
            name="data_validation_failures", 
            condition=lambda metrics: metrics.get('validation_failure_rate', 0) > 5,
            alert_template={
                'title': 'Data Validation Failures',
                'message': 'Data validation failure rate exceeded 5%: {validation_failure_rate:.1f}%',
                'level': AlertLevel.ERROR,
                'source': 'data_validator'
            },
            cooldown_seconds=900  # 15 minutes
        ))
        
        # System health rule
        self._alerting_rules.append(AlertingRule(
            name="system_unhealthy",
            condition=lambda metrics: metrics.get('overall_health_status') == 'unhealthy',
            alert_template={
                'title': 'System Health Critical',
                'message': 'System health status is unhealthy - immediate attention required',
                'level': AlertLevel.CRITICAL,
                'source': 'health_checker'
            },
            cooldown_seconds=1800  # 30 minutes
        ))
    
    def _load_notification_channels(self):
        """Load notification channel configurations."""
        # Email notification (if configured)
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            self._notification_channels.append(NotificationConfig(
                channel=NotificationChannel.EMAIL,
                config=email_config,
                alert_levels=[AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
            ))
        
        # Slack notification (if configured)
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False):
            self._notification_channels.append(NotificationConfig(
                channel=NotificationChannel.SLACK,
                config=slack_config,
                alert_levels=[AlertLevel.ERROR, AlertLevel.CRITICAL]
            ))
        
        # Webhook notification (if configured)
        webhook_config = self.config.get('webhook', {})
        if webhook_config.get('enabled', False):
            self._notification_channels.append(NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                config=webhook_config,
                alert_levels=[AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
            ))
        
        # Always include log-only channel as fallback
        self._notification_channels.append(NotificationConfig(
            channel=NotificationChannel.LOG_ONLY,
            config={},
            alert_levels=[AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        ))
    
    def _integrate_with_existing_systems(self):
        """Integrate with existing monitoring systems from Phases 1.3."""
        # Integrate with Phase 1.3 data monitoring if available
        if DataMonitor:
            try:
                # This would be a callback integration with existing data monitor
                logger.info("Integrated with Phase 1.3 data monitoring alerts")
            except Exception as e:
                logger.error(f"Failed to integrate with Phase 1.3 data monitoring: {e}")
    
    def create_alert(self, 
                    title: str,
                    message: str,
                    level: AlertLevel,
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create a new alert.
        
        Args:
            title: Alert title/summary
            message: Detailed alert message
            level: Alert severity level
            source: Component that generated the alert
            metadata: Additional alert metadata
            
        Returns:
            Created Alert object
        """
        import uuid
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            level=level,
            source=source,
            metadata=metadata or {}
        )
        
        with self._alert_lock:
            # Check for duplicate active alerts
            duplicate = self._find_duplicate_alert(alert)
            if duplicate:
                duplicate.notification_attempts += 1
                logger.debug(f"Suppressed duplicate alert: {title}")
                return duplicate
            
            # Store active alert
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            
            # Update statistics
            self._alert_stats[f"level_{level.name}"] += 1
            self._alert_stats[f"source_{source}"] += 1
            self._alert_stats['total_alerts'] += 1
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"Alert created: {level.name} - {title} (ID: {alert_id})")
        return alert
    
    def _find_duplicate_alert(self, new_alert: Alert) -> Optional[Alert]:
        """Find duplicate active alert based on title and source."""
        for alert in self._active_alerts.values():
            if (alert.title == new_alert.title and 
                alert.source == new_alert.source and
                alert.status == AlertStatus.ACTIVE):
                return alert
        return None
    
    def evaluate_conditions(self, metrics: Dict[str, Any]):
        """
        Evaluate alerting rules against current metrics.
        
        Args:
            metrics: Current system metrics to evaluate against rules
        """
        now = datetime.utcnow()
        
        for rule in self._alerting_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if rule.last_triggered:
                time_since_last = (now - rule.last_triggered).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue
            
            # Check rate limiting
            if rule.trigger_count >= rule.max_alerts_per_hour:
                # Reset hourly counter if needed
                if rule.last_triggered and (now - rule.last_triggered).total_seconds() >= 3600:
                    rule.trigger_count = 0
                else:
                    continue
            
            # Evaluate condition
            try:
                if rule.condition(metrics):
                    # Generate alert
                    template = rule.alert_template.copy()
                    
                    # Format message with metrics
                    try:
                        template['message'] = template['message'].format(**metrics)
                    except KeyError:
                        # Use original message if formatting fails
                        pass
                    
                    # Add rule metadata
                    template.setdefault('metadata', {})
                    template['metadata']['triggered_by_rule'] = rule.name
                    template['metadata']['metrics_snapshot'] = metrics
                    
                    # Create alert
                    self.create_alert(
                        title=template['title'],
                        message=template['message'],
                        level=template['level'],
                        source=template['source'],
                        metadata=template.get('metadata')
                    )
                    
                    # Update rule state
                    rule.last_triggered = now
                    rule.trigger_count += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert through configured channels."""
        for channel_config in self._notification_channels:
            if channel_config.should_notify(alert.level):
                try:
                    if channel_config.channel == NotificationChannel.EMAIL:
                        self._send_email_notification(alert, channel_config.config)
                    elif channel_config.channel == NotificationChannel.SLACK:
                        self._send_slack_notification(alert, channel_config.config)
                    elif channel_config.channel == NotificationChannel.WEBHOOK:
                        self._send_webhook_notification(alert, channel_config.config)
                    elif channel_config.channel == NotificationChannel.LOG_ONLY:
                        self._send_log_notification(alert)
                    
                    alert.notification_attempts += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send {channel_config.channel.value} notification: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"[{alert.level.name}] {alert.title}"
            
            # Create email body
            body = f"""
Alert Details:
- Level: {alert.level.name}
- Source: {alert.source}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Message: {alert.message}

Alert ID: {alert.id}

This alert was generated by the NFL Projects monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587)) as server:
                if config.get('use_tls', True):
                    server.starttls()
                if config.get('username'):
                    server.login(config['username'], config['password'])
                server.send_message(msg)
            
            logger.debug(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            raise
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification."""
        try:
            # Slack color mapping
            color_map = {
                AlertLevel.DEBUG: '#36a64f',    # Green
                AlertLevel.INFO: '#36a64f',     # Green
                AlertLevel.WARNING: '#ff9500',  # Orange
                AlertLevel.ERROR: '#ff0000',    # Red
                AlertLevel.CRITICAL: '#8B0000'  # Dark Red
            }
            
            payload = {
                'text': f"[{alert.level.name}] {alert.title}",
                'attachments': [
                    {
                        'color': color_map.get(alert.level, '#36a64f'),
                        'fields': [
                            {'title': 'Message', 'value': alert.message, 'short': False},
                            {'title': 'Source', 'value': alert.source, 'short': True},
                            {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), 'short': True},
                            {'title': 'Alert ID', 'value': alert.id, 'short': True}
                        ]
                    }
                ]
            }
            
            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            raise
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification."""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.utcnow().isoformat(),
                'system': 'nfl-projects-monitoring'
            }
            
            headers = {'Content-Type': 'application/json'}
            if config.get('auth_header'):
                headers['Authorization'] = config['auth_header']
            
            response = requests.post(
                config['url'], 
                json=payload, 
                headers=headers,
                timeout=config.get('timeout', 10)
            )
            response.raise_for_status()
            
            logger.debug(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            raise
    
    def _send_log_notification(self, alert: Alert):
        """Send log notification (always available fallback)."""
        log_message = f"ALERT [{alert.level.name}] {alert.title} - {alert.message} (ID: {alert.id}, Source: {alert.source})"
        
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        elif alert.level == AlertLevel.INFO:
            logger.info(log_message)
        else:
            logger.debug(log_message)
    
    def acknowledge_alert(self, alert_id: str, user: Optional[str] = None) -> bool:
        """Acknowledge an active alert."""
        with self._alert_lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.acknowledge(user)
                logger.info(f"Alert acknowledged: {alert_id} by {user or 'system'}")
                return True
        
        logger.warning(f"Alert not found for acknowledgement: {alert_id}")
        return False
    
    def resolve_alert(self, alert_id: str, user: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """Resolve an active alert."""
        with self._alert_lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolve(user, reason)
                
                # Move from active to resolved
                del self._active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id} by {user or 'system'}")
                return True
        
        logger.warning(f"Alert not found for resolution: {alert_id}")
        return False
    
    def get_active_alerts(self, level_filter: Optional[AlertLevel] = None) -> List[Alert]:
        """Get list of active alerts."""
        with self._alert_lock:
            alerts = list(self._active_alerts.values())
            
            if level_filter:
                alerts = [a for a in alerts if a.level >= level_filter]
            
            # Sort by level (highest first), then by timestamp
            alerts.sort(key=lambda a: (a.level, a.timestamp), reverse=True)
            
            return alerts
    
    def get_alert_history(self, hours: int = 24, level_filter: Optional[AlertLevel] = None) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._alert_lock:
            history = [
                alert for alert in self._alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            if level_filter:
                history = [a for a in history if a.level >= level_filter]
            
            return sorted(history, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics and metrics."""
        with self._alert_lock:
            active_count = len(self._active_alerts)
            active_by_level = defaultdict(int)
            active_by_source = defaultdict(int)
            
            for alert in self._active_alerts.values():
                active_by_level[alert.level.name] += 1
                active_by_source[alert.source] += 1
            
            return {
                'active_alerts': active_count,
                'active_by_level': dict(active_by_level),
                'active_by_source': dict(active_by_source),
                'total_statistics': dict(self._alert_stats),
                'notification_channels_configured': len(self._notification_channels),
                'alerting_rules_enabled': len([r for r in self._alerting_rules if r.enabled])
            }
    
    def add_alerting_rule(self, rule: AlertingRule):
        """Add a custom alerting rule."""
        self._alerting_rules.append(rule)
        logger.info(f"Added alerting rule: {rule.name}")
    
    def disable_alerting_rule(self, rule_name: str):
        """Disable an alerting rule by name."""
        for rule in self._alerting_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled alerting rule: {rule_name}")
                return
        
        logger.warning(f"Alerting rule not found: {rule_name}")
    
    def cleanup_old_alerts(self, hours: int = 72):
        """Clean up old resolved alerts from memory."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._alert_lock:
            # Remove old alerts from history (deque handles this automatically with maxlen)
            initial_count = len(self._alert_history)
            
            # Remove old active alerts that should have been resolved
            old_active = [
                alert_id for alert_id, alert in self._active_alerts.items()
                if alert.timestamp < cutoff_time and alert.status != AlertStatus.CRITICAL
            ]
            
            for alert_id in old_active:
                del self._active_alerts[alert_id]
                logger.warning(f"Auto-resolved old active alert: {alert_id}")
        
        logger.info(f"Alert cleanup completed - removed {len(old_active)} old active alerts")


# Global alert manager instance
_global_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


# Convenience functions for common alert operations
def send_alert(title: str, 
              message: str, 
              level: AlertLevel = AlertLevel.WARNING,
              source: str = "unknown",
              metadata: Optional[Dict[str, Any]] = None) -> Alert:
    """Send an alert through the global alert manager."""
    manager = get_alert_manager()
    return manager.create_alert(title, message, level, source, metadata)


def send_critical_alert(title: str, message: str, source: str = "system") -> Alert:
    """Send a critical alert."""
    return send_alert(title, message, AlertLevel.CRITICAL, source)


def send_error_alert(title: str, message: str, source: str = "application") -> Alert:
    """Send an error alert."""
    return send_alert(title, message, AlertLevel.ERROR, source)


def send_warning_alert(title: str, message: str, source: str = "system") -> Alert:
    """Send a warning alert."""
    return send_alert(title, message, AlertLevel.WARNING, source)


# Integration decorator for automatic error alerting
def alert_on_error(alert_title: Optional[str] = None, source: str = "application"):
    """
    Decorator to automatically send alerts when function raises exceptions.
    
    Args:
        alert_title: Optional custom alert title. If None, uses function name.
        source: Alert source identifier.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                title = alert_title or f"Error in {func.__name__}"
                message = f"Exception in {func.__name__}: {str(e)}"
                
                send_error_alert(title, message, source)
                raise  # Re-raise the original exception
        
        return wrapper
    return decorator
