"""
Simple validation test to verify the validation framework works.
Uses mock models to test the validation infrastructure.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

# Import validation systems
from power_ranking.validation import PowerRankingBacktester, StatisticalValidationReporter
from nfl_model.validation import SpreadBacktester, PerformanceAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class MockPowerRankingsModel:
    """Mock power rankings model for testing."""
    
    def predict_game_with_uncertainty(self, home_team_id, away_team_id, week):
        """Mock prediction method."""
        # Generate realistic predictions
        home_advantage = 2.5
        random_factor = np.random.normal(0, 3)
        predicted_margin = home_advantage + random_factor
        
        # Mock confidence intervals
        return {
            'predicted_margin': predicted_margin,
            'confidence_interval': (predicted_margin - 5, predicted_margin + 5),
            'win_probability': 0.5 + (predicted_margin / 14.0)
        }

class MockSpreadModel:
    """Mock spread model for testing."""
    
    def predict_spread(self, home_team, away_team, week):
        """Mock spread prediction."""
        # Generate realistic spread predictions
        home_advantage = 2.5
        random_factor = np.random.normal(0, 4)
        return -(home_advantage + random_factor)  # Negative means home favored

def generate_test_data():
    """Generate test data for validation."""
    np.random.seed(42)  # For reproducible results
    
    teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT']
    
    games = []
    for week in range(1, 6):  # Just 5 weeks for testing
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i]
                away_team = teams[i + 1]
                
                # Generate realistic scores
                home_score = max(0, min(50, np.random.normal(24, 8)))
                away_score = max(0, min(50, np.random.normal(21, 8)))
                
                games.append({
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': round(home_score),
                    'away_score': round(away_score),
                    'vegas_spread': np.random.normal(-2.5, 3)
                })
    
    return pd.DataFrame(games)

def test_power_rankings_validation():
    """Test power rankings validation framework."""
    logger = logging.getLogger(__name__)
    logger.info("Testing power rankings validation...")
    
    try:
        # Create mock model and backtester
        mock_model = MockPowerRankingsModel()
        backtester = PowerRankingBacktester(mock_model)
        
        # Generate test data
        test_data = generate_test_data()
        
        # Run backtesting
        results = backtester.backtest_rankings(test_data)
        
        logger.info(f"Power Rankings Results:")
        logger.info(f"  Accuracy: {results.accuracy:.1%}")
        logger.info(f"  Total Games: {results.total_games}")
        logger.info(f"  ROI: {results.roi:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Power rankings validation failed: {e}")
        return None

def test_spread_validation():
    """Test spread model validation framework."""
    logger = logging.getLogger(__name__)
    logger.info("Testing spread model validation...")
    
    try:
        # Create mock model and backtester
        mock_model = MockSpreadModel()
        backtester = SpreadBacktester(mock_model)
        
        # Generate test data
        test_data = generate_test_data()
        
        # Run backtesting
        results = backtester.backtest_predictions(test_data)
        
        logger.info(f"Spread Model Results:")
        logger.info(f"  Accuracy: {results.accuracy:.1%}")
        logger.info(f"  RMSE: {results.rmse:.2f}")
        logger.info(f"  Cover Rate: {results.cover_rate:.1%}")
        logger.info(f"  ROI: {results.roi:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Spread validation failed: {e}")
        return None

def test_performance_metrics():
    """Test performance metrics framework."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance metrics...")
    
    try:
        analyzer = PerformanceAnalyzer()
        
        # Generate test predictions and actuals
        np.random.seed(42)
        predictions = np.random.normal(0, 5, 50).tolist()
        actuals = [p + np.random.normal(0, 3) for p in predictions]  # Add some error
        
        # Run benchmark analysis
        results = analyzer.run_benchmark_analysis(predictions, actuals, "Test Model")
        
        logger.info("Performance Analysis Results:")
        test_metrics = results['metrics']['Test Model']
        logger.info(f"  Accuracy: {test_metrics.accuracy:.1%}")
        logger.info(f"  RMSE: {test_metrics.rmse:.2f}")
        logger.info(f"  ROI: {test_metrics.roi:.1f}%")
        
        # Generate dashboard
        dashboard = analyzer.generate_performance_dashboard(results)
        logger.info("Performance dashboard generated successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Performance metrics test failed: {e}")
        return None

def main():
    """Main test runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("VALIDATION FRAMEWORK TEST")
    logger.info("=" * 50)
    
    # Test power rankings validation
    power_results = test_power_rankings_validation()
    print()
    
    # Test spread model validation
    spread_results = test_spread_validation()
    print()
    
    # Test performance metrics
    perf_results = test_performance_metrics()
    print()
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY:")
    logger.info(f"Power Rankings: {'✅ PASSED' if power_results else '❌ FAILED'}")
    logger.info(f"Spread Model: {'✅ PASSED' if spread_results else '❌ FAILED'}")
    logger.info(f"Performance Metrics: {'✅ PASSED' if perf_results else '❌ FAILED'}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()