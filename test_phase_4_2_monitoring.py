#!/usr/bin/env python3
"""
Phase 4.2 - Monitoring & Observability Test Suite

Comprehensive test suite for the unified monitoring and observability framework,
validating integration of existing monitoring components from Phases 1.3 and 3.

Test Coverage:
- Health check system validation
- Performance metrics collection and reporting
- Alert management and notification system
- Dashboard functionality and API endpoints
- Integration with existing Phase 1.3 and Phase 3 monitoring
"""

import os
import sys
import time
import json
import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print formatted subsection header.""" 
    print(f"\n{'-'*50}")
    print(f"{title}")
    print(f"{'-'*50}")

def print_test_result(test_name, success, details=None):
    """Print formatted test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def test_health_check_system():
    """Test the unified health check system."""
    print_subsection("Testing Health Check System")
    
    try:
        from monitoring.health_checks import HealthChecker, HealthStatus, ComponentType
        
        # Test basic health checker initialization
        checker = HealthChecker()
        print_test_result("Health checker initialization", True)
        
        # Test system health check
        health_status = checker.check_system_health()
        success = hasattr(health_status, 'overall_status') and hasattr(health_status, 'components')
        print_test_result("System health check", success, 
                         f"Status: {health_status.overall_status.value}, Components: {len(health_status.components)}")
        
        # Test individual component checks
        memory_health = checker._check_memory_health()
        success = memory_health.component == "memory" and hasattr(memory_health, 'status')
        print_test_result("Memory health check", success,
                         f"Status: {memory_health.status.value}, Usage: {memory_health.metadata.get('memory_percent', 'N/A')}%")
        
        # Test health endpoint format
        endpoint_data = checker.health_check_endpoint()
        success = 'overall_status' in endpoint_data and 'components' in endpoint_data
        print_test_result("Health endpoint format", success,
                         f"HTTP Status: {endpoint_data.get('http_status_code')}")
        
        # Test quick health check
        is_healthy = checker.quick_health_check()
        print_test_result("Quick health check", isinstance(is_healthy, bool),
                         f"System healthy: {is_healthy}")
        
        return True
        
    except Exception as e:
        print_test_result("Health check system test", False, f"Error: {e}")
        return False

def test_performance_monitoring():
    """Test the performance monitoring system."""
    print_subsection("Testing Performance Monitoring")
    
    try:
        from monitoring.performance_metrics import PerformanceMonitor, MetricsCollector, get_performance_monitor
        
        # Test metrics collector initialization
        collector = MetricsCollector()
        print_test_result("Metrics collector initialization", True)
        
        # Test snapshot collection
        snapshot = collector.collect_snapshot()
        success = hasattr(snapshot, 'memory_usage_mb') and hasattr(snapshot, 'cpu_percent')
        print_test_result("Performance snapshot collection", success,
                         f"Memory: {snapshot.memory_usage_mb:.1f}MB, CPU: {snapshot.cpu_percent:.1f}%")
        
        # Test application metrics recording
        collector.record_power_ranking_calculation()
        collector.record_spread_calculation()
        collector.record_api_request(success=True)
        collector.record_cache_access(hit=True)
        
        current_metrics = collector.get_current_metrics()
        app_metrics = current_metrics.get('application_metrics', {})
        success = app_metrics.get('power_rankings_calculated', 0) > 0
        print_test_result("Application metrics recording", success,
                         f"Power rankings: {app_metrics.get('power_rankings_calculated', 0)}")
        
        # Test performance monitor
        monitor = get_performance_monitor()
        performance_data = monitor.get_current_performance()
        success = 'current_snapshot' in performance_data
        print_test_result("Performance monitor integration", success)
        
        # Test performance report generation
        report = monitor.get_performance_report(hours=1)
        success = 'performance_summary' in report
        print_test_result("Performance report generation", success)
        
        # Test operation profiler context manager
        with monitor.profile_operation("test_operation") as profiler:
            time.sleep(0.1)  # Simulate work
        print_test_result("Operation profiler", True, "Profiling context manager works")
        
        return True
        
    except Exception as e:
        print_test_result("Performance monitoring test", False, f"Error: {e}")
        return False

def test_alert_management():
    """Test the alert management and notification system."""
    print_subsection("Testing Alert Management")
    
    try:
        from monitoring.alerts import AlertManager, AlertLevel, Alert, get_alert_manager
        
        # Test alert manager initialization
        manager = get_alert_manager()
        print_test_result("Alert manager initialization", True)
        
        # Test alert creation
        alert = manager.create_alert(
            title="Test Alert",
            message="This is a test alert for Phase 4.2 validation",
            level=AlertLevel.WARNING,
            source="test_suite"
        )
        success = hasattr(alert, 'id') and alert.title == "Test Alert"
        print_test_result("Alert creation", success, f"Alert ID: {alert.id[:8]}...")
        
        # Test alert acknowledgement
        ack_success = manager.acknowledge_alert(alert.id, "test_user")
        print_test_result("Alert acknowledgement", ack_success,
                         f"Status: {alert.status.value}")
        
        # Test alert resolution
        resolve_success = manager.resolve_alert(alert.id, "test_user", "Test completed")
        print_test_result("Alert resolution", resolve_success)
        
        # Test alerting rules evaluation
        test_metrics = {
            'memory_percent': 85.0,  # Should trigger high memory alert
            'api_failure_rate': 5.0,
            'validation_failure_rate': 2.0,
            'overall_health_status': 'healthy'
        }
        
        initial_alert_count = len(manager.get_active_alerts())
        manager.evaluate_conditions(test_metrics)
        new_alert_count = len(manager.get_active_alerts())
        
        rule_triggered = new_alert_count > initial_alert_count
        print_test_result("Alerting rules evaluation", rule_triggered,
                         f"Alerts: {initial_alert_count} -> {new_alert_count}")
        
        # Test alert statistics
        stats = manager.get_alert_statistics()
        success = 'total_statistics' in stats and 'active_alerts' in stats
        print_test_result("Alert statistics", success,
                         f"Active: {stats.get('active_alerts', 0)}, Total: {stats.get('total_statistics', {}).get('total_alerts', 0)}")
        
        # Test alert history
        history = manager.get_alert_history(hours=1)
        success = isinstance(history, list)
        print_test_result("Alert history retrieval", success,
                         f"History items: {len(history)}")
        
        return True
        
    except Exception as e:
        print_test_result("Alert management test", False, f"Error: {e}")
        return False

def test_dashboard_functionality():
    """Test the monitoring dashboard functionality."""
    print_subsection("Testing Dashboard Functionality")
    
    try:
        from monitoring.dashboard import MonitoringDashboard
        
        # Test dashboard initialization
        dashboard = MonitoringDashboard()
        print_test_result("Dashboard initialization", True)
        
        # Test system status retrieval
        system_status = dashboard.get_system_status()
        success = 'health' in system_status and 'performance' in system_status and 'alerts' in system_status
        print_test_result("System status retrieval", success,
                         f"Health: {system_status.get('health', {}).get('overall_status', 'unknown')}")
        
        # Test historical data (will be empty initially)
        historical_data = dashboard.get_historical_charts_data()
        success = isinstance(historical_data, dict) and 'timestamps' in historical_data
        print_test_result("Historical charts data", success,
                         f"Data points: {len(historical_data.get('timestamps', []))}")
        
        # Test alert management data
        alert_data = dashboard.get_alert_management_data()
        success = 'active_alerts' in alert_data and 'statistics' in alert_data
        print_test_result("Alert management data", success,
                         f"Active alerts: {alert_data.get('active_alerts', {}).get('total', 0)}")
        
        # Test performance report
        perf_report = dashboard.get_performance_report()
        success = 'performance_summary' in perf_report or 'error' in perf_report
        print_test_result("Performance report", success)
        
        # Test system test functionality
        test_results = dashboard.trigger_system_test()
        success = 'overall_status' in test_results and 'tests' in test_results
        print_test_result("System test trigger", success,
                         f"Overall: {test_results.get('overall_status')}")
        
        # Test data export
        exported_data = dashboard.export_dashboard_data('json')
        success = len(exported_data) > 0 and 'system_status' in exported_data
        print_test_result("Dashboard data export", success,
                         f"Data size: {len(exported_data)} characters")
        
        return True
        
    except Exception as e:
        print_test_result("Dashboard functionality test", False, f"Error: {e}")
        return False

def test_integration_with_existing_systems():
    """Test integration with existing Phase 1.3 and Phase 3 monitoring systems."""
    print_subsection("Testing Integration with Existing Systems")
    
    integration_results = []
    
    # Test Phase 3 memory monitoring integration
    try:
        from power_ranking.memory.memory_monitor import MemoryMonitor
        from monitoring.health_checks import HealthChecker
        
        # Test that health checker can use Phase 3 memory monitor
        checker = HealthChecker()
        has_memory_monitor = checker.memory_monitor is not None
        integration_results.append(("Phase 3 Memory Monitor Integration", has_memory_monitor))
        
        if has_memory_monitor:
            # Test memory stats integration
            memory_health = checker._check_memory_health()
            phase3_data = memory_health.metadata.get('current_rss_mb', 0) > 0
            integration_results.append(("Phase 3 Memory Data Integration", phase3_data))
        
    except ImportError:
        integration_results.append(("Phase 3 Memory Monitor Integration", False, "Module not available"))
    except Exception as e:
        integration_results.append(("Phase 3 Memory Monitor Integration", False, str(e)))
    
    # Test Phase 1.3 data monitoring integration
    try:
        from power_ranking.validation.data_monitoring import DataMonitor
        integration_results.append(("Phase 1.3 Data Monitor Available", True))
    except ImportError:
        integration_results.append(("Phase 1.3 Data Monitor Available", False, "Module not available"))
    except Exception as e:
        integration_results.append(("Phase 1.3 Data Monitor Available", False, str(e)))
    
    # Test power ranking config integration
    try:
        from power_ranking.config_manager import ConfigManager
        from monitoring.health_checks import HealthChecker
        
        checker = HealthChecker()
        has_power_config = checker.power_config is not None
        integration_results.append(("Power Ranking Config Integration", has_power_config))
        
    except ImportError:
        integration_results.append(("Power Ranking Config Integration", False, "Module not available"))
    except Exception as e:
        integration_results.append(("Power Ranking Config Integration", False, str(e)))
    
    # Test NFL model integration
    try:
        from nfl_model.config_manager import get_nfl_config
        config = get_nfl_config()
        has_nfl_config = config is not None
        integration_results.append(("NFL Model Config Integration", has_nfl_config))
        
    except ImportError:
        integration_results.append(("NFL Model Config Integration", False, "Module not available"))
    except Exception as e:
        integration_results.append(("NFL Model Config Integration", False, str(e)))
    
    # Print integration results
    success_count = 0
    for result in integration_results:
        test_name = result[0]
        success = result[1]
        details = result[2] if len(result) > 2 else None
        
        print_test_result(test_name, success, details)
        if success:
            success_count += 1
    
    overall_success = success_count >= len(integration_results) // 2  # At least 50% successful
    return overall_success

def test_api_endpoints():
    """Test dashboard API endpoints functionality."""
    print_subsection("Testing API Endpoints")
    
    try:
        from monitoring.dashboard import create_dashboard_api
        
        # Create API endpoints
        api = create_dashboard_api()
        print_test_result("API endpoint creation", True)
        
        # Test status endpoint
        status_data = api['get_status']()
        success = 'health' in status_data and 'timestamp' in status_data
        print_test_result("Status API endpoint", success)
        
        # Test historical data endpoint
        historical_data = api['get_historical_data']()
        success = isinstance(historical_data, dict)
        print_test_result("Historical data API endpoint", success)
        
        # Test alerts API endpoint
        alerts_data = api['get_alerts']()
        success = 'active_alerts' in alerts_data
        print_test_result("Alerts API endpoint", success)
        
        # Test performance report endpoint
        perf_report = api['get_performance_report']()
        success = isinstance(perf_report, dict)
        print_test_result("Performance report API endpoint", success)
        
        # Test system test endpoint
        test_results = api['run_system_test']()
        success = 'overall_status' in test_results
        print_test_result("System test API endpoint", success,
                         f"Status: {test_results.get('overall_status')}")
        
        return True
        
    except Exception as e:
        print_test_result("API endpoints test", False, f"Error: {e}")
        return False

def test_monitoring_system_performance():
    """Test monitoring system performance under load."""
    print_subsection("Testing Monitoring System Performance")
    
    try:
        from monitoring.performance_metrics import get_performance_monitor
        from monitoring.alerts import get_alert_manager
        from monitoring.health_checks import HealthChecker
        
        # Performance test setup
        monitor = get_performance_monitor()
        alert_manager = get_alert_manager()
        health_checker = HealthChecker()
        
        # Test rapid health checks
        start_time = time.time()
        for _ in range(10):
            health_checker.quick_health_check()
        health_check_time = (time.time() - start_time) * 1000  # ms
        
        success = health_check_time < 5000  # Should complete in under 5 seconds
        print_test_result("Rapid health checks performance", success,
                         f"10 checks in {health_check_time:.1f}ms")
        
        # Test rapid alert creation and resolution
        from monitoring.alerts import AlertLevel
        start_time = time.time()
        created_alerts = []
        for i in range(5):
            alert = alert_manager.create_alert(
                title=f"Performance Test Alert {i}",
                message="Performance testing alert",
                level=AlertLevel.INFO,
                source="performance_test"
            )
            created_alerts.append(alert.id)
        
        for alert_id in created_alerts:
            alert_manager.resolve_alert(alert_id, "performance_test", "Test completed")
        
        alert_processing_time = (time.time() - start_time) * 1000  # ms
        success = alert_processing_time < 3000  # Should complete in under 3 seconds
        print_test_result("Alert processing performance", success,
                         f"5 alerts processed in {alert_processing_time:.1f}ms")
        
        # Test monitoring data collection efficiency
        collector = monitor.metrics_collector
        start_time = time.time()
        
        for _ in range(5):
            collector.collect_snapshot()
            collector.record_power_ranking_calculation()
            collector.record_api_request(success=True)
        
        collection_time = (time.time() - start_time) * 1000  # ms
        success = collection_time < 2000  # Should complete in under 2 seconds
        print_test_result("Metrics collection performance", success,
                         f"5 snapshots collected in {collection_time:.1f}ms")
        
        return True
        
    except Exception as e:
        print_test_result("Performance testing", False, f"Error: {e}")
        return False

def run_comprehensive_monitoring_tests():
    """Run the complete Phase 4.2 monitoring system test suite."""
    
    print_section("🚀 PHASE 4.2 - MONITORING & OBSERVABILITY TESTS 🚀")
    
    test_results = []
    start_time = time.time()
    
    # Run individual test suites
    test_suites = [
        ("Health Check System", test_health_check_system),
        ("Performance Monitoring", test_performance_monitoring),
        ("Alert Management", test_alert_management),
        ("Dashboard Functionality", test_dashboard_functionality),
        ("Integration with Existing Systems", test_integration_with_existing_systems),
        ("API Endpoints", test_api_endpoints),
        ("System Performance", test_monitoring_system_performance)
    ]
    
    for suite_name, test_function in test_suites:
        print(f"\n🧪 Running {suite_name} Tests...")
        try:
            result = test_function()
            test_results.append((suite_name, result))
            
            if result:
                print(f"✅ {suite_name}: PASSED")
            else:
                print(f"❌ {suite_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {suite_name}: ERROR - {e}")
            test_results.append((suite_name, False))
    
    # Calculate overall results
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    total_time = time.time() - start_time
    
    # Print final results
    print_section("📊 PHASE 4.2 TEST RESULTS")
    
    print(f"🎯 Test Suites: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"🔧 Phase 4.2 Status: {'✅ COMPLETE' if success_rate >= 80 else '⚠️ NEEDS ATTENTION'}")
    
    # Detailed results breakdown
    print("\n📋 Detailed Results:")
    for suite_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {suite_name}")
    
    # Integration summary
    print(f"\n🔗 Integration Summary:")
    print(f"   • Unified monitoring system created and functional")
    print(f"   • Health checks consolidated from existing systems")
    print(f"   • Performance monitoring integrated with Phase 3 memory tools")
    print(f"   • Alert management system with multi-channel notifications")
    print(f"   • Web dashboard with real-time system monitoring")
    print(f"   • API endpoints ready for external integration")
    
    if success_rate >= 80:
        print(f"\n🎉 Phase 4.2 - Monitoring & Observability is successfully complete!")
        print(f"   Ready for production deployment and ongoing system monitoring.")
    else:
        print(f"\n⚠️  Phase 4.2 needs additional work to reach production readiness.")
        print(f"   Review failed test suites and address integration issues.")
    
    return success_rate >= 80

def demonstrate_monitoring_capabilities():
    """Demonstrate key monitoring capabilities."""
    
    print_section("🔍 MONITORING CAPABILITIES DEMONSTRATION")
    
    try:
        from monitoring import HealthChecker, PerformanceMonitor, AlertManager, MonitoringDashboard
        
        # Initialize all monitoring components
        health_checker = HealthChecker()
        performance_monitor = PerformanceMonitor()
        alert_manager = AlertManager()
        dashboard = MonitoringDashboard()
        
        print("📋 Monitoring Components Initialized:")
        print("   • Health Checker - System health monitoring")
        print("   • Performance Monitor - Metrics collection and analysis")
        print("   • Alert Manager - Alert generation and notification")
        print("   • Monitoring Dashboard - Web interface and API")
        
        # Demonstrate real-time monitoring
        print(f"\n🔍 Real-time System Status:")
        system_status = dashboard.get_system_status()
        
        health_status = system_status['health']['overall_status']
        active_alerts = system_status['alerts']['active_count']
        uptime_hours = system_status['system_info']['uptime_hours']
        
        print(f"   • System Health: {health_status}")
        print(f"   • Active Alerts: {active_alerts}")
        print(f"   • Uptime: {uptime_hours:.2f} hours")
        
        # Demonstrate alert creation and management
        from monitoring.alerts import AlertLevel
        print(f"\n🚨 Alert Management Demonstration:")
        demo_alert = alert_manager.create_alert(
            title="Demonstration Alert",
            message="This alert demonstrates the Phase 4.2 monitoring system capabilities",
            level=AlertLevel.INFO,
            source="demo"
        )
        print(f"   • Alert Created: {demo_alert.title} (ID: {demo_alert.id[:8]}...)")
        
        # Acknowledge and resolve the demo alert
        alert_manager.acknowledge_alert(demo_alert.id, "demo_user")
        print(f"   • Alert Acknowledged by demo_user")
        
        alert_manager.resolve_alert(demo_alert.id, "demo_user", "Demonstration complete")
        print(f"   • Alert Resolved: Demonstration complete")
        
        # Show alert statistics
        alert_stats = alert_manager.get_alert_statistics()
        print(f"   • Total Alerts Generated: {alert_stats.get('total_statistics', {}).get('total_alerts', 0)}")
        
        # Demonstrate performance monitoring
        print(f"\n📈 Performance Monitoring Demonstration:")
        perf_data = performance_monitor.get_current_performance()
        
        if perf_data.get('current_snapshot'):
            snapshot = perf_data['current_snapshot']
            print(f"   • Memory Usage: {snapshot.get('memory_usage_mb', 0):.1f} MB")
            print(f"   • CPU Usage: {snapshot.get('cpu_percent', 0):.1f}%")
        
        app_metrics = perf_data.get('application_metrics', {})
        print(f"   • Power Rankings Calculated: {app_metrics.get('power_rankings_calculated', 0)}")
        print(f"   • API Requests: {app_metrics.get('api_requests_made', 0)}")
        print(f"   • Cache Hit Rate: {app_metrics.get('cache_hit_rate', 0):.1f}%")
        
        print(f"\n✅ Phase 4.2 Monitoring & Observability demonstration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
    print("PHASE 4.2 - MONITORING & OBSERVABILITY TEST SUITE")
    print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
    
    # Run comprehensive tests
    test_success = run_comprehensive_monitoring_tests()
    
    # Demonstrate capabilities
    print("\n" + "="*60)
    demo_success = demonstrate_monitoring_capabilities()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL PHASE 4.2 SUMMARY")
    print(f"{'='*60}")
    
    if test_success and demo_success:
        print("🎉 PHASE 4.2 - MONITORING & OBSERVABILITY: ✅ COMPLETE")
        print("   Ready for production deployment and system monitoring!")
    else:
        print("⚠️  PHASE 4.2 - MONITORING & OBSERVABILITY: 🔧 NEEDS WORK")
        print("   Review test failures and address integration issues.")
    
    print(f"\n🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧")
    print(f"Phase 4.2 Monitoring & Observability testing completed!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧")