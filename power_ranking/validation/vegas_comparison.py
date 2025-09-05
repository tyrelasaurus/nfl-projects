"""
Vegas lines comparison system for validating power rankings and spread predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class VegasComparisonResult:
    """Results from comparing model predictions to Vegas lines."""
    period: str
    total_games: int
    model_accuracy: float
    vegas_accuracy: float
    model_vs_vegas_correlation: float
    model_beat_vegas_count: int
    average_line_difference: float
    edge_opportunities: List[Dict[str, Any]]
    calibration_metrics: Dict[str, float]


@dataclass
class BettingOpportunity:
    """Represents a potential betting opportunity where model disagrees with Vegas."""
    game_id: str
    week: int
    home_team: str
    away_team: str
    vegas_line: float
    model_prediction: float
    edge_magnitude: float
    confidence: float
    actual_result: Optional[float] = None
    profitable: Optional[bool] = None


class VegasLinesComparator:
    """System for comparing model predictions to Vegas betting lines."""
    
    def __init__(self):
        """Initialize Vegas lines comparator."""
        self.historical_comparisons = []
        self.edge_threshold = 3.0  # Points difference to consider significant
        self.confidence_threshold = 0.7  # Minimum confidence for betting opportunities
    
    def load_vegas_lines(self, file_path: str = None, data: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Load Vegas lines data from file or dictionary.
        
        Args:
            file_path: Path to CSV file with Vegas lines
            data: Dictionary with Vegas lines data
            
        Returns:
            DataFrame with Vegas lines data
        """
        if file_path:
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded Vegas lines from {file_path}: {len(df)} games")
                return df
            except Exception as e:
                logger.error(f"Error loading Vegas lines from {file_path}: {e}")
                return pd.DataFrame()
        
        elif data:
            try:
                df = pd.DataFrame(data)
                logger.info(f"Loaded Vegas lines from data: {len(df)} games")
                return df
            except Exception as e:
                logger.error(f"Error creating DataFrame from data: {e}")
                return pd.DataFrame()
        
        else:
            logger.warning("No Vegas lines data provided")
            return pd.DataFrame()
    
    def compare_predictions(self, 
                          model_predictions: List[Dict[str, Any]], 
                          vegas_lines: pd.DataFrame,
                          actual_results: List[Dict[str, Any]] = None) -> VegasComparisonResult:
        """
        Compare model predictions to Vegas lines and actual results.
        
        Args:
            model_predictions: List of model prediction dictionaries
            vegas_lines: DataFrame with Vegas lines
            actual_results: List of actual game results
            
        Returns:
            VegasComparisonResult with comprehensive comparison
        """
        logger.info(f"Comparing {len(model_predictions)} predictions to Vegas lines")
        
        # Merge data sources
        comparison_data = self._merge_prediction_data(model_predictions, vegas_lines, actual_results)
        
        if comparison_data.empty:
            logger.warning("No matching data found for comparison")
            return self._empty_comparison_result()
        
        # Calculate accuracy metrics
        model_accuracy = self._calculate_prediction_accuracy(
            comparison_data['model_prediction'].tolist(), 
            comparison_data['actual_margin'].tolist() if 'actual_margin' in comparison_data.columns else None
        )
        
        vegas_accuracy = self._calculate_prediction_accuracy(
            comparison_data['vegas_line'].tolist(),
            comparison_data['actual_margin'].tolist() if 'actual_margin' in comparison_data.columns else None
        )
        
        # Calculate correlation between model and Vegas
        correlation = np.corrcoef(
            comparison_data['model_prediction'], 
            comparison_data['vegas_line']
        )[0, 1] if len(comparison_data) > 1 else 0.0
        
        # Find edge opportunities
        edge_opportunities = self._find_edge_opportunities(comparison_data)
        
        # Calculate how often model beat Vegas
        model_beat_vegas = 0
        if 'actual_margin' in comparison_data.columns:
            for _, row in comparison_data.iterrows():
                model_error = abs(row['model_prediction'] - row['actual_margin'])
                vegas_error = abs(row['vegas_line'] - row['actual_margin'])
                if model_error < vegas_error:
                    model_beat_vegas += 1
        
        # Calculate average line difference
        line_differences = abs(comparison_data['model_prediction'] - comparison_data['vegas_line'])
        avg_line_difference = np.mean(line_differences)
        
        # Calculate calibration metrics
        calibration = self._calculate_calibration_metrics(comparison_data)
        
        return VegasComparisonResult(
            period=f"{datetime.now().strftime('%Y-%m-%d')}",
            total_games=len(comparison_data),
            model_accuracy=model_accuracy,
            vegas_accuracy=vegas_accuracy,
            model_vs_vegas_correlation=correlation if not np.isnan(correlation) else 0.0,
            model_beat_vegas_count=model_beat_vegas,
            average_line_difference=avg_line_difference,
            edge_opportunities=edge_opportunities,
            calibration_metrics=calibration
        )
    
    def identify_betting_opportunities(self, 
                                     model_predictions: List[Dict[str, Any]], 
                                     vegas_lines: pd.DataFrame) -> List[BettingOpportunity]:
        """
        Identify potential betting opportunities where model significantly disagrees with Vegas.
        
        Args:
            model_predictions: Model predictions
            vegas_lines: Vegas lines data
            
        Returns:
            List of BettingOpportunity objects
        """
        logger.info("Identifying betting opportunities")
        
        comparison_data = self._merge_prediction_data(model_predictions, vegas_lines)
        opportunities = []
        
        for _, row in comparison_data.iterrows():
            model_pred = row['model_prediction']
            vegas_line = row['vegas_line']
            edge_magnitude = abs(model_pred - vegas_line)
            
            # Only consider significant differences
            if edge_magnitude >= self.edge_threshold:
                # Calculate confidence based on historical model performance
                confidence = self._calculate_prediction_confidence(row)
                
                if confidence >= self.confidence_threshold:
                    opportunity = BettingOpportunity(
                        game_id=row.get('game_id', f"week_{row.get('week', 0)}_{row.get('home_team', 'unknown')}"),
                        week=row.get('week', 0),
                        home_team=row.get('home_team', 'Unknown'),
                        away_team=row.get('away_team', 'Unknown'),
                        vegas_line=vegas_line,
                        model_prediction=model_pred,
                        edge_magnitude=edge_magnitude,
                        confidence=confidence
                    )
                    opportunities.append(opportunity)
        
        # Sort by edge magnitude and confidence
        opportunities.sort(key=lambda x: x.edge_magnitude * x.confidence, reverse=True)
        
        logger.info(f"Found {len(opportunities)} betting opportunities")
        return opportunities
    
    def calculate_roi_analysis(self, 
                              betting_opportunities: List[BettingOpportunity],
                              actual_results: List[Dict[str, Any]],
                              bet_size: float = 100.0,
                              betting_strategy: str = 'kelly') -> Dict[str, Any]:
        """
        Calculate ROI analysis for betting opportunities.
        
        Args:
            betting_opportunities: List of betting opportunities
            actual_results: Actual game results
            bet_size: Base bet size for analysis
            betting_strategy: Strategy ('flat', 'kelly', 'proportional')
            
        Returns:
            Dictionary with ROI analysis results
        """
        logger.info(f"Calculating ROI analysis for {len(betting_opportunities)} opportunities")
        
        # Match opportunities with actual results
        opportunities_with_results = self._match_opportunities_with_results(
            betting_opportunities, actual_results
        )
        
        if not opportunities_with_results:
            return {'error': 'No matching results found for ROI analysis'}
        
        total_bet = 0.0
        total_winnings = 0.0
        wins = 0
        losses = 0
        
        bet_details = []
        
        for opp in opportunities_with_results:
            if opp.actual_result is None:
                continue
            
            # Calculate bet size based on strategy
            if betting_strategy == 'flat':
                bet_amount = bet_size
            elif betting_strategy == 'kelly':
                bet_amount = self._kelly_bet_size(opp, bet_size)
            elif betting_strategy == 'proportional':
                bet_amount = bet_size * (opp.confidence * opp.edge_magnitude / 10.0)
            else:
                bet_amount = bet_size
            
            total_bet += bet_amount
            
            # Determine if bet won (assuming standard -110 odds)
            model_correct = self._is_prediction_correct(opp.model_prediction, opp.actual_result)
            vegas_correct = self._is_prediction_correct(opp.vegas_line, opp.actual_result)
            
            # Bet on model when it disagrees with Vegas
            if model_correct and not vegas_correct:
                # Model was right, Vegas wrong - win
                winnings = bet_amount * 0.909  # Standard -110 payout
                total_winnings += bet_amount + winnings
                wins += 1
                opp.profitable = True
            elif not model_correct and vegas_correct:
                # Model was wrong, Vegas right - loss
                losses += 1
                opp.profitable = False
            else:
                # Both right or both wrong - push or unclear
                total_winnings += bet_amount  # Return bet
                opp.profitable = None
            
            bet_details.append({
                'week': opp.week,
                'home_team': opp.home_team,
                'away_team': opp.away_team,
                'bet_amount': bet_amount,
                'edge_magnitude': opp.edge_magnitude,
                'confidence': opp.confidence,
                'model_prediction': opp.model_prediction,
                'vegas_line': opp.vegas_line,
                'actual_result': opp.actual_result,
                'profitable': opp.profitable
            })
        
        roi = ((total_winnings - total_bet) / total_bet * 100) if total_bet > 0 else 0.0
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        
        return {
            'total_opportunities': len(opportunities_with_results),
            'total_bet': total_bet,
            'total_winnings': total_winnings,
            'net_profit': total_winnings - total_bet,
            'roi_percentage': roi,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'betting_strategy': betting_strategy,
            'average_bet_size': total_bet / len(opportunities_with_results) if opportunities_with_results else 0,
            'bet_details': bet_details
        }
    
    def generate_market_efficiency_report(self, comparison_results: List[VegasComparisonResult]) -> Dict[str, Any]:
        """
        Generate report on market efficiency based on model vs Vegas performance.
        
        Args:
            comparison_results: List of comparison results over time
            
        Returns:
            Market efficiency analysis report
        """
        logger.info("Generating market efficiency report")
        
        if not comparison_results:
            return {'error': 'No comparison results provided'}
        
        # Aggregate metrics across all periods
        total_games = sum(r.total_games for r in comparison_results)
        avg_model_accuracy = np.mean([r.model_accuracy for r in comparison_results])
        avg_vegas_accuracy = np.mean([r.vegas_accuracy for r in comparison_results])
        avg_correlation = np.mean([r.model_vs_vegas_correlation for r in comparison_results])
        
        # Calculate market inefficiencies
        total_edge_opportunities = sum(len(r.edge_opportunities) for r in comparison_results)
        edge_frequency = total_edge_opportunities / total_games if total_games > 0 else 0
        
        # Model performance vs market
        model_beats_vegas_rate = np.mean([
            r.model_beat_vegas_count / r.total_games if r.total_games > 0 else 0 
            for r in comparison_results
        ])
        
        return {
            'analysis_period': f"{len(comparison_results)} periods",
            'total_games_analyzed': total_games,
            'market_efficiency_metrics': {
                'model_accuracy': avg_model_accuracy,
                'vegas_accuracy': avg_vegas_accuracy,
                'accuracy_difference': avg_model_accuracy - avg_vegas_accuracy,
                'model_vegas_correlation': avg_correlation,
                'model_beats_vegas_rate': model_beats_vegas_rate
            },
            'inefficiency_indicators': {
                'edge_opportunity_frequency': edge_frequency,
                'total_edge_opportunities': total_edge_opportunities,
                'average_line_difference': np.mean([r.average_line_difference for r in comparison_results]),
                'market_consensus_strength': avg_correlation
            },
            'recommendations': self._generate_market_recommendations(
                avg_model_accuracy, avg_vegas_accuracy, edge_frequency, avg_correlation
            )
        }
    
    def _merge_prediction_data(self, 
                             model_predictions: List[Dict[str, Any]], 
                             vegas_lines: pd.DataFrame,
                             actual_results: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Merge model predictions with Vegas lines and actual results."""
        try:
            # Convert model predictions to DataFrame
            model_df = pd.DataFrame(model_predictions)
            
            # Create merge keys (game identification)
            if 'game_id' in model_df.columns and 'game_id' in vegas_lines.columns:
                merge_key = 'game_id'
            else:
                # Try to match on week + teams
                merge_key = ['week', 'home_team', 'away_team']
            
            # Merge model predictions with Vegas lines
            merged = pd.merge(model_df, vegas_lines, on=merge_key, how='inner', suffixes=('', '_vegas'))
            
            # Add actual results if provided
            if actual_results:
                results_df = pd.DataFrame(actual_results)
                if not results_df.empty:
                    merged = pd.merge(merged, results_df, on=merge_key, how='left', suffixes=('', '_actual'))
            
            return merged
        
        except Exception as e:
            logger.error(f"Error merging prediction data: {e}")
            return pd.DataFrame()
    
    def _calculate_prediction_accuracy(self, predictions: List[float], actuals: List[float] = None) -> float:
        """Calculate prediction accuracy (winner prediction rate)."""
        if not predictions or not actuals:
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actuals) if (p > 0) == (a > 0))
        return correct / len(predictions)
    
    def _find_edge_opportunities(self, comparison_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find games where model significantly disagrees with Vegas."""
        opportunities = []
        
        for _, row in comparison_data.iterrows():
            edge_magnitude = abs(row['model_prediction'] - row['vegas_line'])
            
            if edge_magnitude >= self.edge_threshold:
                opportunity = {
                    'week': row.get('week', 0),
                    'home_team': row.get('home_team', 'Unknown'),
                    'away_team': row.get('away_team', 'Unknown'),
                    'model_prediction': row['model_prediction'],
                    'vegas_line': row['vegas_line'],
                    'edge_magnitude': edge_magnitude,
                    'model_favors': 'home' if row['model_prediction'] > row['vegas_line'] else 'away'
                }
                
                # Add actual result if available
                if 'actual_margin' in row and pd.notna(row['actual_margin']):
                    opportunity['actual_margin'] = row['actual_margin']
                    opportunity['model_correct'] = self._is_prediction_correct(
                        row['model_prediction'], row['actual_margin']
                    )
                    opportunity['vegas_correct'] = self._is_prediction_correct(
                        row['vegas_line'], row['actual_margin']
                    )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_prediction_confidence(self, row: pd.Series) -> float:
        """Calculate confidence in a prediction based on various factors."""
        # Simple confidence calculation based on magnitude of prediction
        # In practice, this would use historical model performance metrics
        prediction_magnitude = abs(row['model_prediction'])
        
        # Higher confidence for larger predicted margins
        magnitude_confidence = min(1.0, prediction_magnitude / 14.0)  # Cap at 14 points
        
        # Additional factors could include:
        # - Historical accuracy for similar games
        # - Data quality/completeness
        # - Model uncertainty estimates
        
        return max(0.1, min(0.95, magnitude_confidence))
    
    def _is_prediction_correct(self, prediction: float, actual: float) -> bool:
        """Check if prediction correctly predicted the winner."""
        return (prediction > 0) == (actual > 0)
    
    def _match_opportunities_with_results(self, 
                                        opportunities: List[BettingOpportunity],
                                        results: List[Dict[str, Any]]) -> List[BettingOpportunity]:
        """Match betting opportunities with actual game results."""
        results_dict = {}
        for result in results:
            key = (result.get('week'), result.get('home_team'), result.get('away_team'))
            results_dict[key] = result.get('actual_margin', 0)
        
        matched_opportunities = []
        for opp in opportunities:
            key = (opp.week, opp.home_team, opp.away_team)
            if key in results_dict:
                opp.actual_result = results_dict[key]
                matched_opportunities.append(opp)
        
        return matched_opportunities
    
    def _kelly_bet_size(self, opportunity: BettingOpportunity, base_bet: float) -> float:
        """Calculate Kelly bet size for opportunity."""
        # Simplified Kelly formula
        # In practice, would need precise odds and win probabilities
        edge = opportunity.edge_magnitude / 14.0  # Normalize to 0-1
        confidence = opportunity.confidence
        
        # Kelly fraction = (bp - q) / b, where b = odds, p = win prob, q = lose prob
        # Simplified version using edge and confidence
        kelly_fraction = (confidence * edge - (1 - confidence)) / 1.0
        kelly_fraction = max(0.01, min(0.25, kelly_fraction))  # Cap at 25% of bankroll
        
        return base_bet * kelly_fraction
    
    def _calculate_calibration_metrics(self, comparison_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate calibration metrics for predictions."""
        if 'actual_margin' not in comparison_data.columns:
            return {}
        
        model_errors = abs(comparison_data['model_prediction'] - comparison_data['actual_margin'])
        vegas_errors = abs(comparison_data['vegas_line'] - comparison_data['actual_margin'])
        
        return {
            'model_mean_absolute_error': np.mean(model_errors),
            'vegas_mean_absolute_error': np.mean(vegas_errors),
            'model_rmse': np.sqrt(np.mean(model_errors ** 2)),
            'vegas_rmse': np.sqrt(np.mean(vegas_errors ** 2)),
            'error_correlation': np.corrcoef(model_errors, vegas_errors)[0, 1] if len(model_errors) > 1 else 0.0
        }
    
    def _generate_market_recommendations(self, 
                                       model_accuracy: float, 
                                       vegas_accuracy: float,
                                       edge_frequency: float, 
                                       correlation: float) -> List[str]:
        """Generate recommendations based on market efficiency analysis."""
        recommendations = []
        
        if model_accuracy > vegas_accuracy + 0.05:  # 5% better
            recommendations.append("Model shows superior predictive power - consider betting strategy")
        
        if edge_frequency > 0.1:  # More than 10% of games have edges
            recommendations.append("Frequent edge opportunities suggest market inefficiencies")
        
        if correlation < 0.7:
            recommendations.append("Low correlation with Vegas suggests unique model insights")
        elif correlation > 0.95:
            recommendations.append("High correlation with Vegas suggests limited edge potential")
        
        if not recommendations:
            recommendations.append("Model performance appears aligned with market efficiency")
        
        return recommendations
    
    def _empty_comparison_result(self) -> VegasComparisonResult:
        """Return empty comparison result."""
        return VegasComparisonResult(
            period="No Data",
            total_games=0,
            model_accuracy=0.0,
            vegas_accuracy=0.0,
            model_vs_vegas_correlation=0.0,
            model_beat_vegas_count=0,
            average_line_difference=0.0,
            edge_opportunities=[],
            calibration_metrics={}
        )


def create_sample_vegas_data(weeks: List[int] = None, teams: List[str] = None) -> pd.DataFrame:
    """
    Create sample Vegas lines data for testing.
    
    Args:
        weeks: List of weeks to create data for
        teams: List of teams to include
        
    Returns:
        DataFrame with sample Vegas lines
    """
    if weeks is None:
        weeks = list(range(1, 19))  # Weeks 1-18
    
    if teams is None:
        teams = ['Kansas City Chiefs', 'Buffalo Bills', 'Cincinnati Bengals', 'Baltimore Ravens',
                'Miami Dolphins', 'New England Patriots', 'Pittsburgh Steelers', 'Cleveland Browns']
    
    np.random.seed(42)  # For reproducible results
    sample_data = []
    
    game_id = 1
    for week in weeks:
        # Generate random matchups for the week
        week_teams = np.random.choice(teams, size=min(len(teams), 8), replace=False)
        
        for i in range(0, len(week_teams), 2):
            if i + 1 < len(week_teams):
                home_team = week_teams[i]
                away_team = week_teams[i + 1]
                
                # Generate realistic Vegas line (-14 to +14)
                vegas_line = np.random.normal(0, 5.0)
                vegas_line = np.clip(vegas_line, -14, 14)
                
                sample_data.append({
                    'game_id': f'game_{game_id}',
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'vegas_line': round(vegas_line, 1)
                })
                game_id += 1
    
    return pd.DataFrame(sample_data)