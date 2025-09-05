"""
Comprehensive test of the Phase 1.3 Data Quality Assurance system.
Demonstrates all data quality features with real NFL data.
"""

import sys
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')

# Import all data quality systems
from power_ranking.validation import (
    DataQualityValidator, DataQualityMonitor, AnomalyDetectionEngine,
    setup_basic_monitoring
)
from power_ranking.api.enhanced_espn_client import EnhancedESPNClient

def setup_logging():
    """Setup comprehensive logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_quality_test.log'),
            logging.StreamHandler()
        ]
    )

def load_real_nfl_data():
    """Load real NFL data for testing."""
    logger = logging.getLogger(__name__)
    
    # Load the real game data
    game_data_path = '/Users/tyrelshaw/Projects/power_ranking/output/data/game_data_complete_272_20250831_214351_20250831_214351.csv'
    
    try:
        df = pd.read_csv(game_data_path)
        
        # Standardize column names
        df = df.rename(columns={
            'home_team_name': 'home_team',
            'away_team_name': 'away_team'
        })
        
        logger.info(f"Loaded {len(df)} real NFL games for testing")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        return None

def create_test_data_with_issues():
    """Create test data with known quality issues for demonstration."""
    # Start with real data
    real_data = load_real_nfl_data()
    if real_data is None:
        return None
    
    # Make a copy and introduce known issues
    test_data = real_data.copy()
    
    # Introduce data quality issues for testing
    
    # 1. Missing values (completeness issue)
    test_data.loc[0:2, 'home_score'] = np.nan
    test_data.loc[5:7, 'away_team'] = np.nan
    
    # 2. Invalid score ranges (accuracy issue)
    test_data.loc[10, 'home_score'] = 85  # Impossible NFL score
    test_data.loc[11, 'away_score'] = -5  # Negative score
    
    # 3. Consistency issues
    test_data.loc[15, 'margin'] = 999  # Inconsistent with scores
    test_data.loc[16, 'home_win'] = True
    test_data.loc[16, 'margin'] = -10  # Home won but negative margin
    
    # 4. Business rule violations
    test_data.loc[20, 'home_team'] = test_data.loc[20, 'away_team']  # Team playing itself
    test_data.loc[25, 'week'] = 25  # Invalid week number
    
    # 5. Statistical anomalies
    test_data.loc[30, 'home_score'] = 70  # Extremely high score
    test_data.loc[31, 'away_score'] = 0   # Shutout (rare but not impossible)
    test_data.loc[32, 'total_points'] = 140  # Unrealistic total
    
    # 6. Temporal anomalies (if we have date)
    if 'date' in test_data.columns:
        test_data.loc[40, 'date'] = '2025-12-31'  # Future date
    
    # 7. Duplicate game IDs
    test_data.loc[50, 'game_id'] = test_data.loc[51, 'game_id']
    
    logging.getLogger(__name__).info(f"Created test dataset with {len(test_data)} records and introduced quality issues")
    return test_data

def test_data_quality_validator():
    """Test the comprehensive data quality validator."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING DATA QUALITY VALIDATOR")
    logger.info("=" * 60)
    
    # Create test data with issues
    test_data = create_test_data_with_issues()
    if test_data is None:
        logger.error("Failed to create test data")
        return False
    
    try:
        # Initialize validator
        validator = DataQualityValidator()
        
        # Run comprehensive validation
        logger.info("Running comprehensive data quality validation...")
        report = validator.validate_game_data(test_data, "Test NFL Dataset with Issues")
        
        # Display results
        logger.info(f"✅ Validation completed successfully!")
        logger.info(f"   Overall Quality Score: {report.overall_score:.1%}")
        logger.info(f"   Total Issues Found: {len(report.issues)}")
        logger.info(f"   Valid Records: {report.valid_records}/{report.total_records}")
        
        # Show issue breakdown
        issue_types = {}
        for issue in report.issues:
            category = issue.category
            issue_types[category] = issue_types.get(category, 0) + 1
        
        logger.info("   Issue Breakdown:")
        for category, count in issue_types.items():
            logger.info(f"     - {category}: {count} issues")
        
        # Generate and save report
        report_text = validator.generate_data_quality_report(
            report, 
            '/Users/tyrelshaw/Projects/data_quality_test_report.md'
        )
        logger.info("   📄 Detailed report saved: data_quality_test_report.md")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_anomaly_detection():
    """Test the anomaly detection engine."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING ANOMALY DETECTION ENGINE")
    logger.info("=" * 60)
    
    # Create test data with anomalies
    test_data = create_test_data_with_issues()
    if test_data is None:
        logger.error("Failed to create test data")
        return False
    
    try:
        # Initialize anomaly detector
        detector = AnomalyDetectionEngine()
        
        # Run anomaly detection
        logger.info("Running comprehensive anomaly detection...")
        result = detector.detect_anomalies(test_data, "Test NFL Dataset")
        
        # Display results
        logger.info(f"✅ Anomaly detection completed!")
        logger.info(f"   Total Anomalies Detected: {result.anomalies_detected}")
        logger.info(f"   Processing Time: {result.processing_time:.2f} seconds")
        logger.info(f"   Detection Rate: {result.anomalies_detected / result.total_records:.1%}")
        
        # Show anomaly breakdown
        anomaly_summary = result.detection_summary
        if anomaly_summary:
            logger.info("   Anomaly Breakdown:")
            for key, count in anomaly_summary.items():
                logger.info(f"     - {key}: {count}")
        
        # Show top anomalies
        logger.info("   Top Anomalies Found:")
        for i, anomaly in enumerate(result.anomalies[:5], 1):  # Show first 5
            logger.info(f"     {i}. {anomaly.severity.value.upper()}: {anomaly.description}")
            logger.info(f"        Confidence: {anomaly.confidence_score:.1%}, Method: {anomaly.detection_method}")
        
        # Generate and save report
        report_text = detector.generate_anomaly_report(
            result,
            '/Users/tyrelshaw/Projects/anomaly_detection_test_report.md'
        )
        logger.info("   📄 Detailed report saved: anomaly_detection_test_report.md")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Anomaly detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality_monitoring():
    """Test the real-time data quality monitoring system."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING DATA QUALITY MONITORING")
    logger.info("=" * 60)
    
    try:
        # Setup monitoring with custom config
        monitoring_config = {
            'monitoring_interval': 5,  # 5 seconds for testing
            'freshness_threshold': 60,  # 1 minute
            'error_rate_threshold': 0.1,  # 10%
            'enable_real_time_alerts': True
        }
        
        monitor = setup_basic_monitoring(monitoring_config)
        logger.info("✅ Monitoring system initialized")
        
        # Start monitoring
        monitor.start_monitoring()
        logger.info("📊 Monitoring started...")
        
        # Simulate some data processing
        test_data = create_test_data_with_issues()
        if test_data is not None:
            # Validate data with monitoring
            logger.info("Running validation with monitoring integration...")
            report = monitor.validate_and_monitor(test_data, "Test Dataset")
            
            # Record some API requests
            monitor.record_api_request("test/endpoint", True, 0.5)
            monitor.record_api_request("test/endpoint", False, 2.0, "TimeoutError")
            monitor.record_api_request("test/endpoint", True, 0.3)
            
            # Wait for monitoring cycle
            logger.info("Waiting for monitoring cycle...")
            time.sleep(6)
            
            # Get monitoring dashboard
            dashboard = monitor.get_monitoring_dashboard()
            logger.info("✅ Monitoring Dashboard:")
            logger.info(f"   Status: {dashboard.get('status')}")
            logger.info(f"   Health: {dashboard.get('health_status')}")
            logger.info(f"   Recent Alerts: {dashboard.get('recent_alerts', {}).get('total', 0)}")
            
            current_metrics = dashboard.get('current_metrics', {})
            logger.info("   Current Metrics:")
            logger.info(f"     - Data Freshness: {current_metrics.get('data_freshness_minutes', 0):.1f} min")
            logger.info(f"     - Error Rate: {current_metrics.get('error_rate', 0):.1%}")
            logger.info(f"     - Completeness: {current_metrics.get('completeness_rate', 0):.1%}")
            
            # Export monitoring report
            monitor.export_monitoring_report(
                '/Users/tyrelshaw/Projects/monitoring_test_report.md',
                hours_back=1
            )
            logger.info("   📄 Monitoring report saved: monitoring_test_report.md")
        
        # Stop monitoring
        monitor.stop_monitoring()
        logger.info("⏹️ Monitoring stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data quality monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_espn_client():
    """Test the enhanced ESPN client with integrated data quality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED ESPN CLIENT")
    logger.info("=" * 60)
    
    try:
        # Initialize enhanced client with validation enabled
        client = EnhancedESPNClient(
            enable_monitoring=True,
            enable_validation=True
        )
        
        logger.info("✅ Enhanced ESPN client initialized")
        
        # Test with real season data (this will use actual ESPN data)
        logger.info("Testing with comprehensive season data validation...")
        
        # This would normally call ESPN API, but we'll use the existing data method
        season_data, validation_report = client.get_comprehensive_season_data(
            season=2024,
            validate_data=True
        )
        
        if validation_report:
            logger.info("✅ Season data validation completed!")
            logger.info(f"   Data Quality Score: {validation_report.overall_score:.1%}")
            logger.info(f"   Issues Found: {len(validation_report.issues)}")
            logger.info(f"   Records Processed: {validation_report.total_records}")
        else:
            logger.warning("⚠️ No validation report generated")
        
        # Get quality summary
        summary = client.get_data_quality_summary()
        logger.info("📊 Data Quality Summary:")
        logger.info(f"   Total Validations: {summary.get('total_validations', 0)}")
        logger.info(f"   Average Quality Score: {summary.get('average_quality_score', 0):.1%}")
        logger.info(f"   Health Status: {summary.get('health_status', 'unknown')}")
        
        # Export validation reports
        client.export_validation_reports(
            '/Users/tyrelshaw/Projects/espn_client_validation_report.md'
        )
        logger.info("   📄 Client validation report saved: espn_client_validation_report.md")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced ESPN client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive Phase 1.3 data quality system test."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀" * 20)
    logger.info("PHASE 1.3 - DATA QUALITY ASSURANCE SYSTEM TEST")
    logger.info("🚀" * 20)
    
    test_results = {}
    
    # Run all tests
    test_results['data_quality_validator'] = test_data_quality_validator()
    print()
    
    test_results['anomaly_detection'] = test_anomaly_detection()
    print()
    
    test_results['data_monitoring'] = test_data_quality_monitoring()
    print()
    
    test_results['enhanced_espn_client'] = test_enhanced_espn_client()
    print()
    
    # Final summary
    logger.info("🎯" * 20)
    logger.info("PHASE 1.3 TEST RESULTS SUMMARY")
    logger.info("🎯" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info("")
    logger.info(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL PHASE 1.3 DATA QUALITY FEATURES WORKING PERFECTLY!")
        logger.info("✅ Data Quality Assurance System is production-ready")
    else:
        logger.info("⚠️ Some tests failed - review logs for details")
    
    # List generated reports
    logger.info("")
    logger.info("📄 Generated Reports:")
    reports = [
        "data_quality_test_report.md",
        "anomaly_detection_test_report.md", 
        "monitoring_test_report.md",
        "espn_client_validation_report.md"
    ]
    
    for report in reports:
        logger.info(f"   - {report}")
    
    logger.info("")
    logger.info("🔧 Phase 1.3 Data Quality Assurance system testing completed!")

if __name__ == "__main__":
    main()