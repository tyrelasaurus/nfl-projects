"""
Monitoring Dashboard for NFL Projects

Provides a web-based monitoring interface that consolidates data from:
- Health check system (monitoring.health_checks)
- Performance metrics (monitoring.performance_metrics)
- Alert management (monitoring.alerts)
- Phase 3 memory monitoring integration
- Phase 1.3 data quality monitoring integration

Features:
- Real-time system status display
- Historical performance charts
- Alert management interface  
- Interactive monitoring controls
- REST API for external integrations
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from threading import Thread, Event
import time
import os
import sys

# Add project paths for monitoring components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .health_checks import HealthChecker, HealthStatus
    from .performance_metrics import PerformanceMonitor, get_performance_monitor
    from .alerts import AlertManager, AlertLevel, get_alert_manager
except ImportError:
    # Fallback imports for development
    from health_checks import HealthChecker, HealthStatus
    from performance_metrics import PerformanceMonitor, get_performance_monitor
    from alerts import AlertManager, AlertLevel, get_alert_manager

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Web-based monitoring dashboard for NFL Projects.
    
    Provides a centralized interface for monitoring system health,
    performance metrics, and alert management across both Power Rankings
    and NFL Spread Model systems.
    
    Features:
    - Real-time system status monitoring
    - Historical performance data visualization
    - Alert management and acknowledgement
    - System configuration and control
    - REST API endpoints for integration
    
    Example:
        >>> dashboard = MonitoringDashboard()
        >>> dashboard.start_monitoring()
        >>> status = dashboard.get_system_status()
        >>> print(f"System health: {status['health']['overall_status']}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize monitoring dashboard.
        
        Args:
            config: Optional dashboard configuration. If None, uses defaults.
                   Expected keys:
                   - refresh_interval: Dashboard refresh rate in seconds (default: 30)
                   - enable_historical_charts: Enable historical data collection (default: True)
                   - max_historical_points: Maximum historical data points (default: 288)
                   - api_enabled: Enable REST API endpoints (default: True)
        """
        self.config = config or self._get_default_config()
        
        # Initialize monitoring components
        self.health_checker = HealthChecker()
        self.performance_monitor = get_performance_monitor()
        self.alert_manager = get_alert_manager()
        
        # Dashboard state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._stop_event = Event()
        
        # Historical data for charts
        self._historical_data = {
            'timestamps': [],
            'memory_usage': [],
            'cpu_usage': [],
            'alert_counts': [],
            'health_status': []
        }
        
        logger.info("MonitoringDashboard initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default dashboard configuration."""
        return {
            'refresh_interval': 30,  # seconds
            'enable_historical_charts': True,
            'max_historical_points': 288,  # 24 hours at 5-minute intervals
            'api_enabled': True,
            'alert_auto_refresh': True,
            'performance_auto_collect': True
        }
    
    def start_monitoring(self):
        """Start the monitoring dashboard and data collection."""
        if self._monitoring_active:
            logger.warning("Monitoring dashboard already active")
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        # Start performance monitoring if not already running
        if self.config.get('performance_auto_collect', True):
            self.performance_monitor.start_monitoring()
        
        # Start dashboard monitoring thread
        if self.config.get('enable_historical_charts', True):
            self._monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
        
        logger.info("Monitoring dashboard started")
    
    def stop_monitoring(self):
        """Stop the monitoring dashboard."""
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        logger.info("Monitoring dashboard stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for data collection."""
        while not self._stop_event.is_set():
            try:
                # Collect current system data
                self._collect_historical_data()
                
                # Wait for next collection
                if self._stop_event.wait(self.config['refresh_interval']):
                    break
                    
            except Exception as e:
                logger.error(f"Dashboard monitoring loop error: {e}")
                time.sleep(self.config['refresh_interval'])
    
    def _collect_historical_data(self):
        """Collect historical data point for dashboard charts."""
        try:
            now = datetime.utcnow()
            
            # Get current system status
            health_status = self.health_checker.check_system_health()
            performance_data = self.performance_monitor.get_current_performance()
            active_alerts = len(self.alert_manager.get_active_alerts())
            
            # Extract metrics
            current_snapshot = performance_data.get('current_snapshot')
            memory_usage = current_snapshot.get('memory_usage_mb', 0) if current_snapshot else 0
            cpu_usage = current_snapshot.get('cpu_percent', 0) if current_snapshot else 0
            
            # Store historical data
            self._historical_data['timestamps'].append(now.isoformat())
            self._historical_data['memory_usage'].append(memory_usage)
            self._historical_data['cpu_usage'].append(cpu_usage)
            self._historical_data['alert_counts'].append(active_alerts)
            self._historical_data['health_status'].append(health_status.overall_status.value)
            
            # Maintain maximum data points
            max_points = self.config['max_historical_points']
            for key in self._historical_data:
                if len(self._historical_data[key]) > max_points:
                    self._historical_data[key] = self._historical_data[key][-max_points:]
            
        except Exception as e:
            logger.error(f"Historical data collection error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for dashboard display.
        
        Returns:
            Dict with system health, performance, alerts, and status information.
        """
        try:
            # Get health status
            health_status = self.health_checker.check_system_health()
            
            # Get performance metrics
            performance_data = self.performance_monitor.get_current_performance()
            
            # Get alert information
            active_alerts = self.alert_manager.get_active_alerts()
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Compile dashboard status
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'health': health_status.to_dict(),
                'performance': performance_data,
                'alerts': {
                    'active_count': len(active_alerts),
                    'active_alerts': [alert.to_dict() for alert in active_alerts[:10]],  # Limit to top 10
                    'statistics': alert_stats
                },
                'system_info': {
                    'monitoring_active': self._monitoring_active,
                    'uptime_hours': health_status.uptime_seconds / 3600,
                    'dashboard_version': '1.0.0'
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'error'
            }
    
    def get_historical_charts_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get historical data for dashboard charts.
        
        Args:
            hours: Number of hours of historical data to return.
            
        Returns:
            Dict with chart data arrays for visualization.
        """
        try:
            # Filter data by time range if requested
            if hours < 24:  # If less than full dataset
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                filtered_data = {}
                
                for key, values in self._historical_data.items():
                    if key == 'timestamps':
                        # Filter by timestamp
                        filtered_indices = [
                            i for i, ts in enumerate(values)
                            if datetime.fromisoformat(ts.replace('Z', '+00:00')) >= cutoff_time
                        ]
                        filtered_data[key] = [values[i] for i in filtered_indices]
                    else:
                        # Filter corresponding data points
                        if 'timestamps' in filtered_data:
                            filtered_data[key] = [values[i] for i in filtered_indices if i < len(values)]
                        else:
                            filtered_data[key] = values
                
                return filtered_data
            else:
                return self._historical_data.copy()
                
        except Exception as e:
            logger.error(f"Error getting historical charts data: {e}")
            return {'error': str(e)}
    
    def get_alert_management_data(self) -> Dict[str, Any]:
        """
        Get alert management data for dashboard alert panel.
        
        Returns:
            Dict with active alerts, history, and management options.
        """
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            recent_history = self.alert_manager.get_alert_history(hours=24)
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Categorize active alerts by level
            alerts_by_level = {level.name: [] for level in AlertLevel}
            for alert in active_alerts:
                alerts_by_level[alert.level.name].append(alert.to_dict())
            
            return {
                'active_alerts': {
                    'total': len(active_alerts),
                    'by_level': alerts_by_level,
                    'all_alerts': [alert.to_dict() for alert in active_alerts]
                },
                'recent_history': [alert.to_dict() for alert in recent_history[:50]],  # Limit to 50
                'statistics': alert_stats,
                'management_actions': {
                    'can_acknowledge': True,
                    'can_resolve': True,
                    'can_suppress': False  # Not implemented yet
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting alert management data: {e}")
            return {'error': str(e)}
    
    def acknowledge_alert(self, alert_id: str, user: Optional[str] = None) -> Dict[str, Any]:
        """
        Acknowledge an alert through the dashboard.
        
        Args:
            alert_id: ID of alert to acknowledge
            user: Optional user who is acknowledging the alert
            
        Returns:
            Dict with acknowledgement result.
        """
        try:
            success = self.alert_manager.acknowledge_alert(alert_id, user)
            
            return {
                'success': success,
                'alert_id': alert_id,
                'acknowledged_by': user,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'alert_id': alert_id
            }
    
    def resolve_alert(self, alert_id: str, user: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve an alert through the dashboard.
        
        Args:
            alert_id: ID of alert to resolve
            user: Optional user who is resolving the alert
            reason: Optional resolution reason
            
        Returns:
            Dict with resolution result.
        """
        try:
            success = self.alert_manager.resolve_alert(alert_id, user, reason)
            
            return {
                'success': success,
                'alert_id': alert_id,
                'resolved_by': user,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'alert_id': alert_id
            }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive performance report for dashboard.
        
        Args:
            hours: Hours of historical data to include in report.
            
        Returns:
            Dict with performance analysis and recommendations.
        """
        try:
            performance_report = self.performance_monitor.get_performance_report(hours)
            
            # Add dashboard-specific analysis
            dashboard_analysis = self._analyze_performance_trends()
            performance_report['dashboard_analysis'] = dashboard_analysis
            
            return performance_report
            
        except Exception as e:
            logger.error(f"Error getting performance report: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        try:
            if not self._historical_data['memory_usage']:
                return {'status': 'insufficient_data'}
            
            # Calculate trends
            memory_data = self._historical_data['memory_usage'][-50:]  # Last 50 points
            cpu_data = self._historical_data['cpu_usage'][-50:]
            
            memory_trend = 'stable'
            if len(memory_data) >= 10:
                recent_avg = sum(memory_data[-10:]) / 10
                older_avg = sum(memory_data[:10]) / 10
                
                if recent_avg > older_avg * 1.1:
                    memory_trend = 'increasing'
                elif recent_avg < older_avg * 0.9:
                    memory_trend = 'decreasing'
            
            cpu_trend = 'stable'
            if len(cpu_data) >= 10:
                recent_avg = sum(cpu_data[-10:]) / 10
                older_avg = sum(cpu_data[:10]) / 10
                
                if recent_avg > older_avg * 1.1:
                    cpu_trend = 'increasing'
                elif recent_avg < older_avg * 0.9:
                    cpu_trend = 'decreasing'
            
            return {
                'memory_trend': memory_trend,
                'cpu_trend': cpu_trend,
                'current_memory_mb': memory_data[-1] if memory_data else 0,
                'current_cpu_percent': cpu_data[-1] if cpu_data else 0,
                'data_points_analyzed': len(memory_data)
            }
            
        except Exception as e:
            logger.error(f"Performance trend analysis error: {e}")
            return {'error': str(e)}
    
    def trigger_system_test(self) -> Dict[str, Any]:
        """
        Trigger a comprehensive system test for dashboard validation.
        
        Returns:
            Dict with test results and system validation.
        """
        try:
            test_results = {
                'timestamp': datetime.utcnow().isoformat(),
                'tests': {}
            }
            
            # Health check test
            try:
                health_status = self.health_checker.check_system_health()
                test_results['tests']['health_check'] = {
                    'status': 'pass',
                    'overall_health': health_status.overall_status.value,
                    'components_tested': len(health_status.components)
                }
            except Exception as e:
                test_results['tests']['health_check'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Performance test
            try:
                perf_data = self.performance_monitor.get_current_performance()
                test_results['tests']['performance_monitoring'] = {
                    'status': 'pass',
                    'monitoring_active': self.performance_monitor._collection_thread and self.performance_monitor._collection_thread.is_alive(),
                    'current_snapshot_available': bool(perf_data.get('current_snapshot'))
                }
            except Exception as e:
                test_results['tests']['performance_monitoring'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Alert system test
            try:
                # Create a test alert
                test_alert = self.alert_manager.create_alert(
                    title="Dashboard System Test",
                    message="Test alert generated by monitoring dashboard",
                    level=AlertLevel.INFO,
                    source="dashboard_test"
                )
                
                # Immediately resolve it
                self.alert_manager.resolve_alert(test_alert.id, "system", "System test completed")
                
                test_results['tests']['alert_system'] = {
                    'status': 'pass',
                    'test_alert_created': test_alert.id,
                    'active_alerts': len(self.alert_manager.get_active_alerts())
                }
            except Exception as e:
                test_results['tests']['alert_system'] = {
                    'status': 'fail',
                    'error': str(e)
                }
            
            # Overall test status
            all_tests_passed = all(
                test.get('status') == 'pass' 
                for test in test_results['tests'].values()
            )
            test_results['overall_status'] = 'pass' if all_tests_passed else 'fail'
            
            return test_results
            
        except Exception as e:
            logger.error(f"System test error: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def export_dashboard_data(self, format: str = 'json') -> str:
        """
        Export current dashboard data for external use.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data as string.
        """
        try:
            dashboard_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'system_status': self.get_system_status(),
                'historical_data': self.get_historical_charts_data(),
                'alert_data': self.get_alert_management_data(),
                'performance_report': self.get_performance_report()
            }
            
            if format.lower() == 'json':
                return json.dumps(dashboard_data, indent=2, default=str)
            elif format.lower() == 'csv':
                # Export historical data as CSV
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write historical data
                historical = dashboard_data['historical_data']
                if historical.get('timestamps'):
                    writer.writerow(['timestamp', 'memory_mb', 'cpu_percent', 'active_alerts', 'health_status'])
                    for i in range(len(historical['timestamps'])):
                        writer.writerow([
                            historical['timestamps'][i],
                            historical.get('memory_usage', [0] * len(historical['timestamps']))[i],
                            historical.get('cpu_usage', [0] * len(historical['timestamps']))[i],
                            historical.get('alert_counts', [0] * len(historical['timestamps']))[i],
                            historical.get('health_status', ['unknown'] * len(historical['timestamps']))[i]
                        ])
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Dashboard data export error: {e}")
            return json.dumps({'error': str(e)})


# Flask/FastAPI integration helpers
def create_dashboard_api():
    """
    Create dashboard API endpoints for web framework integration.
    
    Returns a dictionary of endpoint functions that can be integrated
    with Flask, FastAPI, or other web frameworks.
    
    Example usage with Flask:
        from flask import Flask, jsonify
        from monitoring.dashboard import create_dashboard_api
        
        app = Flask(__name__)
        api = create_dashboard_api()
        
        @app.route('/api/status')
        def status():
            return jsonify(api['get_status']())
    """
    dashboard = MonitoringDashboard()
    dashboard.start_monitoring()
    
    return {
        'get_status': dashboard.get_system_status,
        'get_historical_data': dashboard.get_historical_charts_data,
        'get_alerts': dashboard.get_alert_management_data,
        'acknowledge_alert': dashboard.acknowledge_alert,
        'resolve_alert': dashboard.resolve_alert,
        'get_performance_report': dashboard.get_performance_report,
        'run_system_test': dashboard.trigger_system_test,
        'export_data': dashboard.export_dashboard_data
    }


# Standalone dashboard server (for development/testing)
def run_simple_dashboard_server(host: str = '0.0.0.0', port: int = 8080):
    """
    Run a simple dashboard server for development and testing.
    
    Args:
        host: Server host address
        port: Server port number
    """
    try:
        from flask import Flask, jsonify, render_template_string, request
        
        app = Flask(__name__)
        dashboard = MonitoringDashboard()
        dashboard.start_monitoring()
        
        # Simple HTML dashboard template
        DASHBOARD_HTML = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NFL Projects Monitoring Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status-healthy { color: green; }
                .status-degraded { color: orange; }
                .status-unhealthy { color: red; }
                .alert-critical { background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 5px 0; }
                .alert-error { background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin: 5px 0; }
                .alert-warning { background-color: #fffde7; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>NFL Projects Monitoring Dashboard</h1>
            <div id="status">Loading...</div>
            <script>
                function loadStatus() {
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('status').innerHTML = 
                                '<h2>System Status: <span class="status-' + data.health.overall_status + '">' + 
                                data.health.overall_status + '</span></h2>' +
                                '<p>Active Alerts: ' + data.alerts.active_count + '</p>' +
                                '<p>Memory Usage: ' + (data.performance.current_snapshot ? 
                                data.performance.current_snapshot.memory_usage_mb.toFixed(1) + ' MB' : 'N/A') + '</p>' +
                                '<p>Last Updated: ' + new Date(data.timestamp).toLocaleString() + '</p>';
                        });
                }
                loadStatus();
                setInterval(loadStatus, 30000);
            </script>
        </body>
        </html>
        """
        
        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @app.route('/api/status')
        def api_status():
            return jsonify(dashboard.get_system_status())
        
        @app.route('/api/alerts')
        def api_alerts():
            return jsonify(dashboard.get_alert_management_data())
        
        @app.route('/api/performance')
        def api_performance():
            hours = request.args.get('hours', 24, type=int)
            return jsonify(dashboard.get_performance_report(hours))
        
        @app.route('/api/test')
        def api_test():
            return jsonify(dashboard.trigger_system_test())
        
        print(f"Starting NFL Projects Monitoring Dashboard on http://{host}:{port}")
        app.run(host=host, port=port, debug=False)
        
    except ImportError:
        print("Flask is required to run the simple dashboard server")
        print("Install with: pip install flask")
    except Exception as e:
        print(f"Error starting dashboard server: {e}")


if __name__ == "__main__":
    # Run simple dashboard server when executed directly
    run_simple_dashboard_server()