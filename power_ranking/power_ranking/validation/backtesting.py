"""
Backtesting framework for validating power ranking model performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from power_ranking.models.power_rankings import PowerRankModel
from power_ranking.api.espn_client import ESPNClient

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    period: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    mean_absolute_error: float
    root_mean_squared_error: float
    correlation_coefficient: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    ranking_stability: float
    detailed_results: List[Dict[str, Any]]


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for model performance."""
    ranking_accuracy: float  # How often better teams beat worse teams
    spread_accuracy: float  # How often spreads predict winners correctly
    calibration_score: float  # How well confidence matches actual performance
    ranking_correlation: float  # Correlation with final season standings
    predictive_power: float  # R² against actual game outcomes
    stability_score: float  # Week-to-week ranking consistency


class PowerRankingBacktester:
    """Comprehensive backtesting framework for power ranking models."""
    
    def __init__(self, model: PowerRankModel = None, client: ESPNClient = None):
        """
        Initialize backtester.
        
        Args:
            model: PowerRankModel instance to test
            client: ESPNClient for data retrieval
        """
        self.model = model or PowerRankModel()
        self.client = client or ESPNClient()
        self.results_history = []
    
    def validate_historical_accuracy(self, 
                                   start_week: int = 1, 
                                   end_week: int = 18,
                                   season: int = 2024) -> BacktestResult:
        """
        Validate model accuracy against historical game outcomes.
        
        Args:
            start_week: Starting week for validation
            end_week: Ending week for validation
            season: Season to validate against
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting historical accuracy validation for weeks {start_week}-{end_week}, season {season}")
        
        predictions = []
        actuals = []
        detailed_results = []
        week_rankings = {}
        
        for week in range(start_week, end_week + 1):
            try:
                # Get historical data up to this week
                historical_data = self._get_historical_data_through_week(week - 1, season)
                if not historical_data:
                    logger.warning(f"No historical data available for week {week}")
                    continue
                
                # Generate rankings based on data up to previous week
                teams_info = self._get_teams_info(season)
                rankings, _ = self.model.compute(historical_data, teams_info)
                
                if not rankings:
                    logger.warning(f"No rankings generated for week {week}")
                    continue
                
                week_rankings[week] = {team_id: score for team_id, _, score in rankings}
                
                # Get actual games for this week to validate against
                week_games = self._get_week_games(week, season)
                
                # Generate predictions for each game
                for game in week_games:
                    home_team = game.get('home_team_id')
                    away_team = game.get('away_team_id')
                    actual_margin = game.get('margin', 0)  # Home team perspective
                    
                    if home_team in week_rankings[week] and away_team in week_rankings[week]:
                        predicted_margin = self._predict_game_margin(
                            home_team, away_team, week_rankings[week]
                        )
                        
                        predictions.append(predicted_margin)
                        actuals.append(actual_margin)
                        
                        # Store detailed results
                        detailed_results.append({
                            'week': week,
                            'home_team': home_team,
                            'away_team': away_team,
                            'predicted_margin': predicted_margin,
                            'actual_margin': actual_margin,
                            'prediction_correct': (predicted_margin > 0) == (actual_margin > 0),
                            'absolute_error': abs(predicted_margin - actual_margin)
                        })
            
            except Exception as e:
                logger.error(f"Error validating week {week}: {e}")
                continue
        
        if not predictions:
            logger.warning("No predictions generated for validation")
            return self._empty_backtest_result(f"Weeks {start_week}-{end_week}, Season {season}")
        
        # Calculate comprehensive metrics
        return self._calculate_backtest_metrics(
            predictions, actuals, detailed_results, week_rankings,
            f"Weeks {start_week}-{end_week}, Season {season}"
        )
    
    def validate_ranking_stability(self, season: int = 2024) -> Dict[str, float]:
        """
        Validate week-to-week ranking stability.
        
        Args:
            season: Season to analyze
            
        Returns:
            Dictionary of stability metrics
        """
        logger.info(f"Validating ranking stability for season {season}")
        
        weekly_rankings = {}
        stability_scores = []
        
        for week in range(2, 19):  # Weeks 2-18 (need previous week for comparison)
            try:
                historical_data = self._get_historical_data_through_week(week - 1, season)
                if not historical_data:
                    continue
                
                teams_info = self._get_teams_info(season)
                rankings, _ = self.model.compute(historical_data, teams_info)
                
                if rankings:
                    weekly_rankings[week] = {team_id: rank for rank, (team_id, _, _) in enumerate(rankings, 1)}
                
                # Calculate stability with previous week
                if week > 2 and (week - 1) in weekly_rankings and week in weekly_rankings:
                    stability = self._calculate_ranking_stability(
                        weekly_rankings[week - 1], weekly_rankings[week]
                    )
                    stability_scores.append(stability)
            
            except Exception as e:
                logger.error(f"Error calculating stability for week {week}: {e}")
                continue
        
        return {
            'mean_stability': np.mean(stability_scores) if stability_scores else 0.0,
            'std_stability': np.std(stability_scores) if stability_scores else 0.0,
            'min_stability': np.min(stability_scores) if stability_scores else 0.0,
            'max_stability': np.max(stability_scores) if stability_scores else 1.0,
            'weeks_analyzed': len(stability_scores)
        }
    
    def compare_to_final_standings(self, season: int = 2024) -> Dict[str, float]:
        """
        Compare rankings to final season standings.
        
        Args:
            season: Season to analyze
            
        Returns:
            Dictionary of comparison metrics
        """
        logger.info(f"Comparing rankings to final standings for season {season}")
        
        try:
            # Get final season data and generate end-of-season rankings
            final_data = self._get_historical_data_through_week(18, season)
            if not final_data:
                return {'error': 'No final season data available'}
            
            teams_info = self._get_teams_info(season)
            final_rankings, computation_data = self.model.compute(final_data, teams_info)
            
            if not final_rankings:
                return {'error': 'Could not generate final rankings'}
            
            # Get actual final standings (wins/losses)
            actual_standings = self._calculate_actual_standings(final_data, teams_info)
            
            # Calculate correlation between power rankings and actual win percentages
            power_scores = [score for _, _, score in final_rankings]
            win_percentages = []
            
            for team_id, _, _ in final_rankings:
                if team_id in actual_standings:
                    win_percentages.append(actual_standings[team_id]['win_pct'])
                else:
                    win_percentages.append(0.5)  # Neutral if missing
            
            correlation = np.corrcoef(power_scores, win_percentages)[0, 1] if len(power_scores) > 1 else 0.0
            
            return {
                'power_ranking_correlation': correlation,
                'teams_analyzed': len(final_rankings),
                'mean_power_score': np.mean(power_scores),
                'mean_win_percentage': np.mean(win_percentages),
                'power_score_range': np.max(power_scores) - np.min(power_scores),
                'win_pct_range': np.max(win_percentages) - np.min(win_percentages)
            }
        
        except Exception as e:
            logger.error(f"Error comparing to final standings: {e}")
            return {'error': str(e)}
    
    def validate_predictive_power(self, test_weeks: List[int] = None, season: int = 2024) -> ValidationMetrics:
        """
        Comprehensive validation of model's predictive power.
        
        Args:
            test_weeks: Weeks to test (default: weeks 6-18)
            season: Season to analyze
            
        Returns:
            ValidationMetrics with comprehensive evaluation
        """
        if test_weeks is None:
            test_weeks = list(range(6, 19))  # Weeks 6-18 (after some data accumulation)
        
        logger.info(f"Validating predictive power for weeks {test_weeks}, season {season}")
        
        correct_predictions = 0
        total_predictions = 0
        ranking_correct = 0
        ranking_total = 0
        prediction_errors = []
        weekly_correlations = []
        
        for week in test_weeks:
            try:
                # Use data through previous week to predict current week
                historical_data = self._get_historical_data_through_week(week - 1, season)
                if not historical_data:
                    continue
                
                teams_info = self._get_teams_info(season)
                rankings, _ = self.model.compute(historical_data, teams_info)
                
                if not rankings:
                    continue
                
                power_dict = {team_id: score for team_id, _, score in rankings}
                week_games = self._get_week_games(week, season)
                
                week_predictions = []
                week_actuals = []
                
                for game in week_games:
                    home_team = game.get('home_team_id')
                    away_team = game.get('away_team_id')
                    actual_margin = game.get('margin', 0)
                    
                    if home_team in power_dict and away_team in power_dict:
                        predicted_margin = self._predict_game_margin(home_team, away_team, power_dict)
                        
                        # Winner prediction accuracy
                        predicted_winner = predicted_margin > 0
                        actual_winner = actual_margin > 0
                        
                        if predicted_winner == actual_winner:
                            correct_predictions += 1
                        total_predictions += 1
                        
                        # Ranking accuracy (better team should win)
                        home_power = power_dict[home_team]
                        away_power = power_dict[away_team]
                        better_team_home = home_power > away_power
                        
                        if better_team_home == actual_winner:
                            ranking_correct += 1
                        ranking_total += 1
                        
                        prediction_errors.append(abs(predicted_margin - actual_margin))
                        week_predictions.append(predicted_margin)
                        week_actuals.append(actual_margin)
                
                # Calculate weekly correlation
                if len(week_predictions) > 1:
                    week_corr = np.corrcoef(week_predictions, week_actuals)[0, 1]
                    if not np.isnan(week_corr):
                        weekly_correlations.append(week_corr)
            
            except Exception as e:
                logger.error(f"Error validating week {week}: {e}")
                continue
        
        # Calculate final metrics
        ranking_accuracy = ranking_correct / ranking_total if ranking_total > 0 else 0.0
        spread_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        mean_error = np.mean(prediction_errors) if prediction_errors else float('inf')
        correlation = np.mean(weekly_correlations) if weekly_correlations else 0.0
        
        return ValidationMetrics(
            ranking_accuracy=ranking_accuracy,
            spread_accuracy=spread_accuracy,
            calibration_score=self._calculate_calibration_score(prediction_errors),
            ranking_correlation=correlation,
            predictive_power=correlation ** 2 if correlation > 0 else 0.0,  # R²
            stability_score=0.0  # Will be calculated separately
        )
    
    def _predict_game_margin(self, home_team: str, away_team: str, power_dict: Dict[str, float]) -> float:
        """
        Predict game margin using power ratings.
        
        Args:
            home_team: Home team ID
            away_team: Away team ID
            power_dict: Dictionary of team power ratings
            
        Returns:
            Predicted margin (home team perspective)
        """
        home_power = power_dict.get(home_team, 0.0)
        away_power = power_dict.get(away_team, 0.0)
        
        # Basic prediction: power difference + home field advantage
        return (home_power - away_power) + 2.5  # 2.5 point home field advantage
    
    def _calculate_backtest_metrics(self, 
                                   predictions: List[float], 
                                   actuals: List[float],
                                   detailed_results: List[Dict],
                                   week_rankings: Dict,
                                   period: str) -> BacktestResult:
        """Calculate comprehensive backtesting metrics."""
        if not predictions or not actuals:
            return self._empty_backtest_result(period)
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # Basic metrics
        correct_predictions = sum(1 for p, a in zip(predictions, actuals) if (p > 0) == (a > 0))
        accuracy = correct_predictions / len(predictions)
        mae = np.mean(np.abs(predictions_array - actuals_array))
        rmse = np.sqrt(np.mean((predictions_array - actuals_array) ** 2))
        correlation = np.corrcoef(predictions_array, actuals_array)[0, 1] if len(predictions) > 1 else 0.0
        
        # Confidence intervals (assuming normal distribution)
        error_std = np.std(predictions_array - actuals_array)
        confidence_intervals = {
            '68%': (-error_std, error_std),
            '95%': (-1.96 * error_std, 1.96 * error_std),
            '99%': (-2.58 * error_std, 2.58 * error_std)
        }
        
        # Ranking stability
        stability = self._calculate_overall_ranking_stability(week_rankings)
        
        return BacktestResult(
            period=period,
            total_predictions=len(predictions),
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            correlation_coefficient=correlation if not np.isnan(correlation) else 0.0,
            confidence_intervals=confidence_intervals,
            ranking_stability=stability,
            detailed_results=detailed_results
        )
    
    def _calculate_ranking_stability(self, prev_rankings: Dict[str, int], curr_rankings: Dict[str, int]) -> float:
        """Calculate ranking stability between two weeks."""
        if not prev_rankings or not curr_rankings:
            return 0.0
        
        common_teams = set(prev_rankings.keys()) & set(curr_rankings.keys())
        if len(common_teams) < 2:
            return 0.0
        
        rank_changes = []
        for team in common_teams:
            change = abs(prev_rankings[team] - curr_rankings[team])
            rank_changes.append(change)
        
        # Normalize by maximum possible change
        max_change = len(common_teams) - 1
        if max_change == 0:
            return 1.0
        
        mean_change = np.mean(rank_changes)
        stability = 1.0 - (mean_change / max_change)
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_overall_ranking_stability(self, week_rankings: Dict) -> float:
        """Calculate overall ranking stability across all weeks."""
        stabilities = []
        weeks = sorted(week_rankings.keys())
        
        for i in range(1, len(weeks)):
            prev_week = weeks[i-1]
            curr_week = weeks[i]
            
            # Convert power scores to rankings
            prev_rankings = self._power_scores_to_rankings(week_rankings[prev_week])
            curr_rankings = self._power_scores_to_rankings(week_rankings[curr_week])
            
            stability = self._calculate_ranking_stability(prev_rankings, curr_rankings)
            stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 0.0
    
    def _power_scores_to_rankings(self, power_scores: Dict[str, float]) -> Dict[str, int]:
        """Convert power scores to rankings (1 = best)."""
        sorted_teams = sorted(power_scores.items(), key=lambda x: x[1], reverse=True)
        return {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}
    
    def _calculate_calibration_score(self, errors: List[float]) -> float:
        """Calculate calibration score based on prediction errors."""
        if not errors:
            return 0.0
        
        # Simple calibration: how well do prediction intervals match actual error distribution
        error_array = np.array(errors)
        mean_error = np.mean(error_array)
        std_error = np.std(error_array)
        
        # Good calibration means errors are close to normal distribution
        try:
            from scipy import stats
            # Shapiro-Wilk test for normality (closer to 1 = better calibration)
            statistic, p_value = stats.shapiro(error_array[:min(5000, len(error_array))])
            return statistic
        except ImportError:
            # Fallback: simple measure based on error distribution
            return max(0.0, 1.0 - (abs(mean_error) + std_error) / 20.0)
    
    def _get_historical_data_through_week(self, through_week: int, season: int) -> Optional[Dict]:
        """Get historical data through specified week."""
        # This would integrate with the ESPN client to get historical data
        # For now, return None to indicate no data available
        logger.debug(f"Requesting historical data through week {through_week}, season {season}")
        return None
    
    def _get_teams_info(self, season: int) -> List[Dict]:
        """Get teams information for specified season."""
        try:
            return self.client.get_teams()
        except Exception as e:
            logger.error(f"Error getting teams info: {e}")
            return []
    
    def _get_week_games(self, week: int, season: int) -> List[Dict]:
        """Get games for specified week and season."""
        # This would get actual game results for the specified week
        logger.debug(f"Requesting games for week {week}, season {season}")
        return []
    
    def _calculate_actual_standings(self, season_data: Dict, teams_info: List[Dict]) -> Dict[str, Dict]:
        """Calculate actual win/loss standings from season data."""
        standings = {}
        
        # Extract game results and calculate standings
        # This would process the season data to generate actual standings
        
        return standings
    
    def _empty_backtest_result(self, period: str) -> BacktestResult:
        """Return empty backtest result for cases with no data."""
        return BacktestResult(
            period=period,
            total_predictions=0,
            correct_predictions=0,
            accuracy=0.0,
            mean_absolute_error=float('inf'),
            root_mean_squared_error=float('inf'),
            correlation_coefficient=0.0,
            confidence_intervals={},
            ranking_stability=0.0,
            detailed_results=[]
        )


def run_comprehensive_validation(model: PowerRankModel = None, 
                               client: ESPNClient = None,
                               season: int = 2024) -> Dict[str, Any]:
    """
    Run comprehensive validation of power ranking model.
    
    Args:
        model: PowerRankModel to validate
        client: ESPNClient for data
        season: Season to validate
        
    Returns:
        Dictionary with all validation results
    """
    backtester = PowerRankingBacktester(model, client)
    
    results = {
        'validation_timestamp': datetime.now().isoformat(),
        'season': season,
        'model_config': {
            'weights': model.weights if model else None,
            'rolling_window': model.rolling_window if model else None
        }
    }
    
    try:
        # Historical accuracy validation
        logger.info("Running historical accuracy validation...")
        historical_result = backtester.validate_historical_accuracy(season=season)
        results['historical_accuracy'] = historical_result
        
        # Ranking stability validation
        logger.info("Running ranking stability validation...")
        stability_results = backtester.validate_ranking_stability(season=season)
        results['ranking_stability'] = stability_results
        
        # Final standings comparison
        logger.info("Comparing to final standings...")
        standings_comparison = backtester.compare_to_final_standings(season=season)
        results['standings_comparison'] = standings_comparison
        
        # Predictive power validation
        logger.info("Validating predictive power...")
        predictive_metrics = backtester.validate_predictive_power(season=season)
        results['predictive_power'] = predictive_metrics
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        results['error'] = str(e)
    
    return results