"""
Backtesting framework for NFL spread predictions.
Validates the spread prediction model against historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    accuracy: float
    avg_error: float
    rmse: float
    mae: float
    hit_rate: float
    cover_rate: float
    roi: float
    total_games: int
    correct_picks: int
    push_rate: float
    weekly_performance: Dict[int, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class BettingResult:
    """Results from simulated betting strategy."""
    total_bets: int
    winning_bets: int
    losing_bets: int
    push_bets: int
    total_return: float
    roi_percentage: float
    kelly_optimal: float
    max_drawdown: float

class SpreadBacktester:
    """Backtesting framework for spread predictions."""
    
    def __init__(self, spread_model, confidence_level: float = 0.95):
        self.spread_model = spread_model
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
    def backtest_predictions(self, historical_data: pd.DataFrame, 
                           start_week: int = 1, end_week: int = 18) -> BacktestResult:
        """
        Backtest spread predictions against historical results.
        
        Args:
            historical_data: DataFrame with historical game data
            start_week: Starting week for backtest
            end_week: Ending week for backtest
            
        Returns:
            BacktestResult with performance metrics
        """
        results = []
        weekly_performance = {}
        
        for week in range(start_week, end_week + 1):
            week_data = historical_data[historical_data['week'] == week]
            if week_data.empty:
                continue
                
            week_results = []
            for _, game in week_data.iterrows():
                try:
                    # Get model prediction
                    prediction = self.spread_model.predict_spread(
                        game['home_team'], game['away_team'], week
                    )
                    
                    # Calculate actual spread (positive means home team won by more)
                    actual_spread = game['home_score'] - game['away_score']
                    
                    # Calculate prediction error
                    error = prediction - actual_spread
                    
                    # Determine if prediction was correct (within 0.5 points)
                    correct = abs(error) <= 0.5
                    
                    # Check if bet would have covered
                    vegas_line = game.get('vegas_spread', prediction)
                    covered = (actual_spread + vegas_line) > 0
                    
                    week_results.append({
                        'predicted_spread': prediction,
                        'actual_spread': actual_spread,
                        'error': error,
                        'correct': correct,
                        'covered': covered,
                        'vegas_line': vegas_line
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing game in week {week}: {e}")
                    continue
            
            if week_results:
                week_accuracy = np.mean([r['correct'] for r in week_results])
                weekly_performance[week] = week_accuracy
                results.extend(week_results)
        
        return self._calculate_metrics(results, weekly_performance)
    
    def _calculate_metrics(self, results: List[Dict], 
                          weekly_performance: Dict[int, float]) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        if not results:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {})
        
        errors = [r['error'] for r in results]
        correct_predictions = sum(r['correct'] for r in results)
        covered_bets = sum(r['covered'] for r in results)
        total_games = len(results)
        
        # Basic metrics
        accuracy = correct_predictions / total_games
        avg_error = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        mae = np.mean([abs(e) for e in errors])
        hit_rate = accuracy
        cover_rate = covered_bets / total_games
        
        # ROI calculation (simplified betting simulation)
        roi = self._calculate_roi(results)
        
        # Push rate (games within 0.5 points)
        pushes = sum(1 for e in errors if abs(e) <= 0.5)
        push_rate = pushes / total_games
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(errors)
        
        return BacktestResult(
            accuracy=accuracy,
            avg_error=avg_error,
            rmse=rmse,
            mae=mae,
            hit_rate=hit_rate,
            cover_rate=cover_rate,
            roi=roi,
            total_games=total_games,
            correct_picks=correct_predictions,
            push_rate=push_rate,
            weekly_performance=weekly_performance,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_roi(self, results: List[Dict]) -> float:
        """Calculate return on investment for betting strategy."""
        total_bet = 0
        total_return = 0
        
        for result in results:
            bet_amount = 100  # Standard bet
            total_bet += bet_amount
            
            if result['covered']:
                total_return += bet_amount * 1.91  # -110 odds
            
        if total_bet == 0:
            return 0.0
            
        return (total_return - total_bet) / total_bet * 100
    
    def _calculate_confidence_intervals(self, errors: List[float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        if not errors:
            return {}
        
        alpha = 1 - self.confidence_level
        errors_array = np.array(errors)
        
        # Error confidence interval
        error_mean = np.mean(errors_array)
        error_se = np.std(errors_array) / np.sqrt(len(errors_array))
        error_margin = 1.96 * error_se  # Assuming normal distribution
        
        # RMSE confidence interval (using bootstrap)
        rmse_values = []
        for _ in range(1000):
            bootstrap_sample = np.random.choice(errors_array, size=len(errors_array), replace=True)
            rmse_values.append(np.sqrt(np.mean(bootstrap_sample**2)))
        
        rmse_ci = (
            np.percentile(rmse_values, (alpha/2) * 100),
            np.percentile(rmse_values, (1 - alpha/2) * 100)
        )
        
        return {
            'mean_error': (error_mean - error_margin, error_mean + error_margin),
            'rmse': rmse_ci
        }
    
    def rolling_validation(self, historical_data: pd.DataFrame, 
                          window_size: int = 4) -> Dict[int, BacktestResult]:
        """
        Perform rolling window validation.
        
        Args:
            historical_data: Historical game data
            window_size: Size of validation window in weeks
            
        Returns:
            Dictionary mapping week to BacktestResult
        """
        results = {}
        max_week = historical_data['week'].max()
        
        for start_week in range(1, max_week - window_size + 2):
            end_week = start_week + window_size - 1
            
            result = self.backtest_predictions(
                historical_data, start_week, end_week
            )
            results[start_week] = result
            
        return results
    
    def cross_validation(self, historical_data: pd.DataFrame, 
                        n_folds: int = 5) -> List[BacktestResult]:
        """
        Perform k-fold cross validation.
        
        Args:
            historical_data: Historical game data
            n_folds: Number of folds for cross validation
            
        Returns:
            List of BacktestResult for each fold
        """
        weeks = sorted(historical_data['week'].unique())
        fold_size = len(weeks) // n_folds
        results = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(weeks)
            
            test_weeks = weeks[start_idx:end_idx]
            test_data = historical_data[historical_data['week'].isin(test_weeks)]
            
            if not test_data.empty:
                result = self.backtest_predictions(
                    test_data, min(test_weeks), max(test_weeks)
                )
                results.append(result)
                
        return results
    
    def simulate_betting_strategy(self, historical_data: pd.DataFrame,
                                bankroll: float = 10000,
                                bet_sizing: str = "kelly") -> BettingResult:
        """
        Simulate a betting strategy using the model predictions.
        
        Args:
            historical_data: Historical game data
            bankroll: Starting bankroll
            bet_sizing: Betting strategy ("fixed", "kelly", "proportional")
            
        Returns:
            BettingResult with betting performance
        """
        current_bankroll = bankroll
        bet_history = []
        max_bankroll = bankroll
        max_drawdown = 0
        
        for _, game in historical_data.iterrows():
            try:
                # Get model prediction and confidence
                prediction = self.spread_model.predict_spread(
                    game['home_team'], game['away_team'], game['week']
                )
                
                # Determine bet size based on strategy
                if bet_sizing == "fixed":
                    bet_amount = min(100, current_bankroll * 0.02)
                elif bet_sizing == "kelly":
                    # Simplified Kelly criterion
                    edge = abs(prediction - game.get('vegas_spread', prediction)) / 10
                    bet_fraction = max(0, min(0.25, edge))
                    bet_amount = current_bankroll * bet_fraction
                else:  # proportional
                    bet_amount = current_bankroll * 0.02
                
                if bet_amount < 1:
                    continue
                
                # Simulate bet outcome
                actual_spread = game['home_score'] - game['away_score']
                vegas_line = game.get('vegas_spread', prediction)
                
                # Determine if bet won
                bet_result = (actual_spread + vegas_line) > 0
                
                if bet_result:
                    payout = bet_amount * 0.91  # -110 odds
                    current_bankroll += payout
                else:
                    current_bankroll -= bet_amount
                
                bet_history.append({
                    'bet_amount': bet_amount,
                    'won': bet_result,
                    'bankroll_after': current_bankroll
                })
                
                # Track maximum drawdown
                max_bankroll = max(max_bankroll, current_bankroll)
                drawdown = (max_bankroll - current_bankroll) / max_bankroll
                max_drawdown = max(max_drawdown, drawdown)
                
            except Exception as e:
                self.logger.warning(f"Error in betting simulation: {e}")
                continue
        
        # Calculate results
        winning_bets = sum(1 for bet in bet_history if bet['won'])
        losing_bets = len(bet_history) - winning_bets
        total_return = current_bankroll - bankroll
        roi_percentage = (total_return / bankroll) * 100 if bankroll > 0 else 0
        
        return BettingResult(
            total_bets=len(bet_history),
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            push_bets=0,  # Simplified for now
            total_return=total_return,
            roi_percentage=roi_percentage,
            kelly_optimal=0.0,  # Would need more sophisticated calculation
            max_drawdown=max_drawdown
        )
    
    def generate_performance_report(self, results: BacktestResult, 
                                  save_path: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# Spread Prediction Backtesting Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Performance
        report.append("## Overall Performance")
        report.append(f"- **Total Games Analyzed**: {results.total_games}")
        report.append(f"- **Prediction Accuracy**: {results.accuracy:.1%}")
        report.append(f"- **Cover Rate**: {results.cover_rate:.1%}")
        report.append(f"- **Average Error**: {results.avg_error:.2f} points")
        report.append(f"- **RMSE**: {results.rmse:.2f} points")
        report.append(f"- **MAE**: {results.mae:.2f} points")
        report.append(f"- **Push Rate**: {results.push_rate:.1%}")
        report.append(f"- **Simulated ROI**: {results.roi:.1f}%")
        report.append("")
        
        # Weekly Performance
        if results.weekly_performance:
            report.append("## Weekly Performance")
            for week, accuracy in sorted(results.weekly_performance.items()):
                report.append(f"- Week {week}: {accuracy:.1%}")
            report.append("")
        
        # Confidence Intervals
        if results.confidence_intervals:
            report.append("## Confidence Intervals (95%)")
            for metric, (lower, upper) in results.confidence_intervals.items():
                report.append(f"- {metric.replace('_', ' ').title()}: [{lower:.2f}, {upper:.2f}]")
            report.append("")
        
        # Performance Analysis
        report.append("## Performance Analysis")
        if results.accuracy > 0.53:
            report.append("✅ **Strong Performance**: Model shows significant predictive power")
        elif results.accuracy > 0.50:
            report.append("⚠️ **Moderate Performance**: Model slightly beats random chance")
        else:
            report.append("❌ **Poor Performance**: Model underperforms random selection")
        
        if results.rmse < 10:
            report.append("✅ **Good Precision**: Low prediction error variance")
        elif results.rmse < 14:
            report.append("⚠️ **Moderate Precision**: Average prediction error variance")
        else:
            report.append("❌ **Poor Precision**: High prediction error variance")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text