"""
Comprehensive validation runner for NFL prediction models.
Integrates power rankings and spread model validation systems.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

# Import validation systems
from power_ranking.validation import PowerRankingBacktester, ValidationReportGenerator
from nfl_model.validation import SpreadBacktester, PerformanceAnalyzer

# Import models
from power_ranking.models.power_rankings import PowerRankingsModel
from nfl_model.spread_model import SpreadCalculator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('validation_results.log'),
            logging.StreamHandler()
        ]
    )

def generate_sample_data():
    """Generate sample historical data for testing."""
    np.random.seed(42)  # For reproducible results
    
    teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 
             'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC']
    
    games = []
    for week in range(1, 18):
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i]
                away_team = teams[i + 1]
                
                # Generate realistic scores
                home_score = np.random.normal(24, 10)
                away_score = np.random.normal(21, 10)
                
                # Ensure scores are reasonable
                home_score = max(0, min(50, home_score))
                away_score = max(0, min(50, away_score))
                
                games.append({
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': round(home_score),
                    'away_score': round(away_score),
                    'vegas_spread': np.random.normal(-2.5, 4)  # Home team favored by 2.5 on average
                })
    
    return pd.DataFrame(games)

def run_power_rankings_validation():
    """Run comprehensive power rankings validation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting power rankings validation...")
    
    try:
        # Initialize models and validators
        power_model = PowerRankingsModel()
        backtester = PowerRankingBacktester(power_model)
        report_generator = ValidationReportGenerator()
        
        # Generate sample data
        historical_data = generate_sample_data()
        
        # Run backtesting
        backtest_results = backtester.backtest_rankings(historical_data)
        
        # Generate validation report
        report_path = '/Users/tyrelshaw/Projects/power_rankings_validation_report.md'
        report_generator.generate_comprehensive_report(
            backtest_results=backtest_results,
            save_path=report_path
        )
        
        logger.info(f"Power rankings validation completed. Report saved to: {report_path}")
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in power rankings validation: {e}")
        return None

def run_spread_model_validation():
    """Run comprehensive spread model validation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting spread model validation...")
    
    try:
        # Initialize models and validators
        spread_model = SpreadCalculator()
        backtester = SpreadBacktester(spread_model)
        performance_analyzer = PerformanceAnalyzer()
        
        # Generate sample data
        historical_data = generate_sample_data()
        
        # Run backtesting
        backtest_results = backtester.backtest_predictions(historical_data)
        
        # Extract predictions and actuals for performance analysis
        predictions = []
        actuals = []
        
        for _, game in historical_data.iterrows():
            try:
                # Use spread calculator method
                result = spread_model.calculate_spread(
                    game['home_team'], game['away_team'], game['week']
                )
                pred = result.projected_spread
                actual = game['home_score'] - game['away_score']
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Error processing game: {e}")
                continue
        
        # Run benchmark analysis
        benchmark_results = performance_analyzer.run_benchmark_analysis(
            predictions, actuals, "Spread Model"
        )
        
        # Generate performance dashboard
        dashboard_path = '/Users/tyrelshaw/Projects/spread_model_dashboard.md'
        performance_analyzer.generate_performance_dashboard(
            benchmark_results, dashboard_path
        )
        
        # Generate backtesting report
        backtest_report_path = '/Users/tyrelshaw/Projects/spread_model_backtest_report.md'
        backtester.generate_performance_report(backtest_results, backtest_report_path)
        
        logger.info(f"Spread model validation completed.")
        logger.info(f"Dashboard saved to: {dashboard_path}")
        logger.info(f"Backtest report saved to: {backtest_report_path}")
        
        return {
            'backtest_results': backtest_results,
            'benchmark_results': benchmark_results
        }
        
    except Exception as e:
        logger.error(f"Error in spread model validation: {e}")
        return None

def generate_combined_report(power_results, spread_results):
    """Generate a combined validation report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating combined validation report...")
    
    report = []
    report.append("# NFL Prediction Models - Comprehensive Validation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## Executive Summary")
    report.append("This report provides comprehensive validation results for both the NFL Power Rankings")
    report.append("and Spread Prediction models, including performance metrics, statistical analysis,")
    report.append("and benchmark comparisons.")
    report.append("")
    
    # Power Rankings Summary
    if power_results:
        report.append("### Power Rankings Model")
        report.append(f"- **Prediction Accuracy**: {power_results.accuracy:.1%}")
        report.append(f"- **Average Error**: {power_results.avg_error:.2f} points")
        report.append(f"- **Betting ROI**: {power_results.roi:.1f}%")
        report.append(f"- **Games Analyzed**: {power_results.total_games}")
        report.append("")
    
    # Spread Model Summary
    if spread_results and 'backtest_results' in spread_results:
        backtest = spread_results['backtest_results']
        report.append("### Spread Prediction Model")
        report.append(f"- **Prediction Accuracy**: {backtest.accuracy:.1%}")
        report.append(f"- **Cover Rate**: {backtest.cover_rate:.1%}")
        report.append(f"- **RMSE**: {backtest.rmse:.2f} points")
        report.append(f"- **Betting ROI**: {backtest.roi:.1f}%")
        report.append("")
    
    # Model Comparison
    report.append("## Model Comparison")
    if power_results and spread_results and 'backtest_results' in spread_results:
        spread_backtest = spread_results['backtest_results']
        
        report.append("| Metric | Power Rankings | Spread Model |")
        report.append("|--------|---------------|--------------|")
        report.append(f"| Accuracy | {power_results.accuracy:.1%} | {spread_backtest.accuracy:.1%} |")
        report.append(f"| ROI | {power_results.roi:.1f}% | {spread_backtest.roi:.1f}% |")
        report.append(f"| Games | {power_results.total_games} | {spread_backtest.total_games} |")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("Based on the validation results:")
    report.append("")
    report.append("1. **Model Selection**: Use the model with higher accuracy for predictions")
    report.append("2. **Risk Management**: Both models should be used with proper bankroll management")
    report.append("3. **Continuous Monitoring**: Regular validation should be performed with new data")
    report.append("4. **Ensemble Approach**: Consider combining both models for improved predictions")
    report.append("")
    
    report.append("## Detailed Reports")
    report.append("- Power Rankings Validation: `power_rankings_validation_report.md`")
    report.append("- Spread Model Dashboard: `spread_model_dashboard.md`")
    report.append("- Spread Model Backtest: `spread_model_backtest_report.md`")
    
    # Save combined report
    report_text = "\n".join(report)
    combined_path = '/Users/tyrelshaw/Projects/combined_validation_report.md'
    
    with open(combined_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Combined report saved to: {combined_path}")
    return report_text

def main():
    """Main validation runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("NFL PREDICTION MODELS - COMPREHENSIVE VALIDATION")
    logger.info("=" * 60)
    
    # Run power rankings validation
    power_results = run_power_rankings_validation()
    
    # Run spread model validation  
    spread_results = run_spread_model_validation()
    
    # Generate combined report
    if power_results or spread_results:
        generate_combined_report(power_results, spread_results)
        logger.info("Validation process completed successfully!")
    else:
        logger.error("Validation process failed - no results generated")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()