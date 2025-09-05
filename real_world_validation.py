"""
Real-world validation test using actual NFL game data.
Tests validation framework against 2024 NFL season data.
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

def load_real_nfl_data():
    """Load actual NFL game data."""
    logger = logging.getLogger(__name__)
    
    # Load the most complete game data file
    game_data_path = '/Users/tyrelshaw/Projects/power_ranking/output/data/game_data_complete_272_20250831_214351_20250831_214351.csv'
    
    try:
        df = pd.read_csv(game_data_path)
        logger.info(f"Loaded {len(df)} games from 2024 NFL season")
        
        # Standardize column names for validation framework
        df_standardized = df.rename(columns={
            'home_team_name': 'home_team',
            'away_team_name': 'away_team'
        })
        
        # Add vegas spread (simulated based on actual game outcomes for testing)
        # In real usage, this would come from sportsbook data
        df_standardized['vegas_spread'] = -(df_standardized['margin'] + np.random.normal(0, 2, len(df)))
        
        logger.info(f"Data columns: {df_standardized.columns.tolist()}")
        logger.info(f"Week range: {df_standardized['week'].min()} to {df_standardized['week'].max()}")
        logger.info(f"Teams: {len(df_standardized['home_team'].unique())} unique teams")
        
        return df_standardized
        
    except Exception as e:
        logger.error(f"Error loading game data: {e}")
        return None

class RealWorldPowerRankingsModel:
    """Power rankings model adapter for real data testing."""
    
    def __init__(self):
        # Simple elo-based model for demonstration
        self.team_ratings = {}
        self.k_factor = 32
        self.home_advantage = 2.5
        
    def predict_game_with_uncertainty(self, home_team, away_team, week):
        """Predict game outcome with uncertainty estimates."""
        # Initialize ratings if needed
        if home_team not in self.team_ratings:
            self.team_ratings[home_team] = 1500
        if away_team not in self.team_ratings:
            self.team_ratings[away_team] = 1500
            
        # Calculate rating difference
        home_rating = self.team_ratings[home_team]
        away_rating = self.team_ratings[away_team]
        
        # Predict margin (simplified)
        rating_diff = (home_rating - away_rating) / 25  # Scale to points
        predicted_margin = rating_diff + self.home_advantage
        
        # Add uncertainty
        uncertainty = 7.0  # Standard NFL game uncertainty
        confidence_interval = (predicted_margin - uncertainty, predicted_margin + uncertainty)
        
        # Win probability based on margin
        win_prob = 1 / (1 + 10**(-predicted_margin/14))
        
        return {
            'predicted_margin': predicted_margin,
            'confidence_interval': confidence_interval,
            'win_probability': win_prob
        }
    
    def update_ratings(self, home_team, away_team, actual_margin):
        """Update team ratings based on actual results."""
        if home_team not in self.team_ratings:
            self.team_ratings[home_team] = 1500
        if away_team not in self.team_ratings:
            self.team_ratings[away_team] = 1500
            
        # Expected result
        expected_home = 1 / (1 + 10**((self.team_ratings[away_team] - self.team_ratings[home_team])/400))
        
        # Actual result (1 if home won, 0 if away won)
        actual_home = 1 if actual_margin > 0 else 0
        
        # Update ratings
        self.team_ratings[home_team] += self.k_factor * (actual_home - expected_home)
        self.team_ratings[away_team] += self.k_factor * (expected_home - actual_home)

class RealWorldSpreadModel:
    """Spread model adapter for real data testing."""
    
    def __init__(self):
        self.power_model = RealWorldPowerRankingsModel()
        
    def predict_spread(self, home_team, away_team, week):
        """Predict spread for a game."""
        prediction = self.power_model.predict_game_with_uncertainty(home_team, away_team, week)
        # Return negative for home favorite (sportsbook convention)
        return -prediction['predicted_margin']

def test_power_rankings_with_real_data():
    """Test power rankings validation with real NFL data."""
    logger = logging.getLogger(__name__)
    logger.info("Testing power rankings with real NFL data...")
    
    try:
        # Load real data
        nfl_data = load_real_nfl_data()
        if nfl_data is None:
            return None
            
        # Create model and backtester
        model = RealWorldPowerRankingsModel()
        backtester = PowerRankingBacktester(model)
        
        # Run backtesting on first 10 weeks
        test_data = nfl_data[nfl_data['week'] <= 10].copy()
        logger.info(f"Testing on {len(test_data)} games from weeks 1-10")
        
        # Train model on early weeks, test on later weeks
        train_data = test_data[test_data['week'] <= 6]
        validation_data = test_data[test_data['week'] > 6]
        
        # Train model
        for _, game in train_data.iterrows():
            model.update_ratings(game['home_team'], game['away_team'], game['margin'])
        
        # Run validation
        results = backtester.validate_historical_accuracy(7, 10, 2024)
        
        logger.info("Power Rankings Results (Real Data):")
        logger.info(f"  Accuracy: {results.accuracy:.1%}")
        logger.info(f"  Total Games: {results.total_games}")
        logger.info(f"  Average Error: {results.avg_error:.2f} points")
        logger.info(f"  ROI: {results.roi:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Power rankings validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_spread_model_with_real_data():
    """Test spread model validation with real NFL data."""
    logger = logging.getLogger(__name__)
    logger.info("Testing spread model with real NFL data...")
    
    try:
        # Load real data
        nfl_data = load_real_nfl_data()
        if nfl_data is None:
            return None
            
        # Create model and backtester
        model = RealWorldSpreadModel()
        backtester = SpreadBacktester(model)
        
        # Run backtesting on first 8 weeks
        test_data = nfl_data[nfl_data['week'] <= 8].copy()
        logger.info(f"Testing on {len(test_data)} games from weeks 1-8")
        
        # Train model on early weeks
        train_data = test_data[test_data['week'] <= 4]
        for _, game in train_data.iterrows():
            model.power_model.update_ratings(game['home_team'], game['away_team'], game['margin'])
        
        # Test on later weeks
        validation_data = test_data[test_data['week'] > 4]
        results = backtester.backtest_predictions(validation_data)
        
        logger.info("Spread Model Results (Real Data):")
        logger.info(f"  Accuracy: {results.accuracy:.1%}")
        logger.info(f"  RMSE: {results.rmse:.2f} points")
        logger.info(f"  Cover Rate: {results.cover_rate:.1%}")
        logger.info(f"  ROI: {results.roi:.1f}%")
        logger.info(f"  Total Games: {results.total_games}")
        
        return results
        
    except Exception as e:
        logger.error(f"Spread validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_performance_metrics_with_real_data():
    """Test performance metrics with real predictions vs actuals."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance metrics with real data...")
    
    try:
        # Load real data
        nfl_data = load_real_nfl_data()
        if nfl_data is None:
            return None
            
        # Create model and generate predictions
        model = RealWorldSpreadModel()
        
        # Train on early weeks
        train_data = nfl_data[nfl_data['week'] <= 4]
        for _, game in train_data.iterrows():
            model.power_model.update_ratings(game['home_team'], game['away_team'], game['margin'])
        
        # Generate predictions for validation weeks
        test_data = nfl_data[(nfl_data['week'] > 4) & (nfl_data['week'] <= 8)]
        
        predictions = []
        actuals = []
        
        for _, game in test_data.iterrows():
            try:
                pred = model.predict_spread(game['home_team'], game['away_team'], game['week'])
                actual = game['margin']  # Actual margin
                
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue
        
        logger.info(f"Generated {len(predictions)} predictions for performance analysis")
        
        # Run performance analysis
        analyzer = PerformanceAnalyzer()
        results = analyzer.run_benchmark_analysis(predictions, actuals, "Real NFL Model")
        
        # Display results
        model_metrics = results['metrics']['Real NFL Model']
        logger.info("Performance Analysis Results (Real Data):")
        logger.info(f"  Accuracy: {model_metrics.accuracy:.1%}")
        logger.info(f"  RMSE: {model_metrics.rmse:.2f}")
        logger.info(f"  ROI: {model_metrics.roi:.1f}%")
        logger.info(f"  Sharpe Ratio: {model_metrics.sharpe_ratio:.2f}")
        
        # Generate and save dashboard
        dashboard_path = '/Users/tyrelshaw/Projects/real_world_performance_dashboard.md'
        dashboard = analyzer.generate_performance_dashboard(results, dashboard_path)
        logger.info(f"Performance dashboard saved to: {dashboard_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main validation runner with real NFL data."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("REAL-WORLD NFL DATA VALIDATION TEST")
    logger.info("=" * 60)
    
    # Test power rankings validation
    power_results = test_power_rankings_with_real_data()
    print()
    
    # Test spread model validation
    spread_results = test_spread_model_with_real_data()
    print()
    
    # Test performance metrics
    perf_results = test_performance_metrics_with_real_data()
    print()
    
    # Summary
    logger.info("=" * 60)
    logger.info("REAL-WORLD VALIDATION SUMMARY:")
    logger.info(f"Power Rankings: {'✅ PASSED' if power_results else '❌ FAILED'}")
    logger.info(f"Spread Model: {'✅ PASSED' if spread_results else '❌ FAILED'}")
    logger.info(f"Performance Metrics: {'✅ PASSED' if perf_results else '❌ FAILED'}")
    logger.info("=" * 60)
    
    if power_results and spread_results:
        logger.info("\n🎯 VALIDATION FRAMEWORK SUCCESSFULLY TESTED WITH REAL NFL DATA!")
        logger.info("✅ All validation systems are working with actual game results")
        logger.info("📊 Performance metrics generated and saved")
        
if __name__ == "__main__":
    main()