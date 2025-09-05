import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
import os
import sys

# Import from power_ranking config_manager
from power_ranking.config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class PowerRankingWithConfidence:
    """Power ranking with confidence interval information."""
    team_id: str
    team_name: str
    power_score: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    sample_size: int
    stability_score: float


@dataclass
class PredictionWithUncertainty:
    """Game prediction with uncertainty quantification."""
    home_team: str
    away_team: str
    predicted_margin: float
    confidence_interval: Tuple[float, float]
    win_probability_home: float
    prediction_variance: float
    model_confidence: float


class PowerRankModel:
    """
    Advanced NFL power rankings calculation engine.
    
    This model implements a sophisticated multi-factor ranking system that combines:
    - Season-long performance metrics with margin of victory adjustments
    - Rolling averages to emphasize recent performance trends
    - Strength of Schedule (SOS) normalization for fair team comparison
    - Contextual game weighting (e.g., reduced Week 18 impact)
    
    The ranking algorithm uses iterative refinement to handle circular dependencies
    in strength of schedule calculations and applies statistical confidence intervals
    to ranking outputs.
    
    Attributes:
        weights (Dict[str, float]): Weighting factors for different ranking components
        config: Configuration manager instance with model parameters
        rolling_window (int): Number of recent games for rolling averages
        week18_weight (float): Weight reduction factor for Week 18 games
        
    Example:
        >>> model = PowerRankModel()
        >>> rankings, data = model.compute(scoreboard_data, teams_info)
        >>> print(f"Top team: {rankings[0][1]} with score {rankings[0][2]:.2f}")
    """
    
    def __init__(self, weights: Dict[str, float] = None, config=None):
        """
        Initialize the power ranking model with configuration and weights.
        
        Args:
            weights: Optional custom weights for ranking components. If None,
                    uses configuration defaults or standard NFL weighting.
            config: Configuration manager instance. If None, loads default config.
                   
        The default weighting scheme emphasizes:
        - Season average margin: Primary factor (typically 0.4-0.6 weight)
        - Rolling average margin: Recent trend factor (typically 0.2-0.3 weight)
        - Strength of schedule: Opponent quality adjustment (typically 0.1-0.2 weight)
        - Recency factor: Temporal weighting for recent games
        """
        # Load configuration
        self.config = config or get_config()
        
        # Use configured weights or provided weights or defaults
        if weights:
            self.weights = weights
        else:
            model_weights = self.config.model.weights
            self.weights = {
                'season_avg_margin': model_weights.season_avg_margin,
                'rolling_avg_margin': model_weights.rolling_avg_margin, 
                'sos': model_weights.sos,
                'recency_factor': model_weights.recency_factor
            }
        
        # Use configured values
        self.rolling_window = self.config.model.rolling_window
        self.week18_weight = self.config.model.week18_weight
    
    def compute(self, scoreboard_data: Dict, teams_info: List[Dict], last_n_games: Optional[int] = None) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        """
        Compute comprehensive power rankings from ESPN scoreboard data.
        
        This is the main entry point for power ranking calculations. The method processes
        raw ESPN API data through a multi-stage pipeline:
        
        1. Data extraction and team mapping
        2. Contextual game weighting (Week 18 adjustments, etc.)
        3. Statistical computation (season stats, rolling averages, SOS)
        4. Final power score calculation using weighted factors
        5. Ranking generation with confidence metrics
        
        Args:
            scoreboard_data: Raw ESPN scoreboard API response containing game events
            teams_info: List of team information dictionaries with IDs and names
            
        Returns:
            Tuple containing:
            - List of (team_id, team_name, power_score) tuples, sorted by score descending
            - Dict of comprehensive computation data for analysis and export
            
        The computation data includes:
            - game_results: Processed game data with margins and contexts
            - season_stats: Full season statistical metrics
            - rolling_stats: Recent performance rolling averages  
            - sos_scores: Strength of schedule calculations
            - power_scores: Final power scores by team
            
        Example:
            >>> rankings, data = model.compute(espn_scoreboard, team_list)
            >>> top_team = rankings[0]
            >>> print(f"{top_team[1]}: {top_team[2]:.2f}")
            >>> print(f"Based on {len(data['game_results'])} games")
        """
        teams_map = self._build_teams_map(teams_info)
        game_results = self._extract_game_results(scoreboard_data, teams_map)
        
        if not game_results:
            logger.warning("No game results found")
            return [], {}
        
        # Apply Week 18 weighting to games
        weighted_games = self._apply_game_weights(game_results)

        # If requested, restrict calculations to each team's last N games using event timestamps
        last_n_filter = None
        if last_n_games and last_n_games > 0:
            last_n_filter = self._build_last_n_filter(weighted_games, last_n_games)
        
        # Calculate comprehensive season statistics with weighted games
        season_stats = self._calculate_season_stats(weighted_games, last_n_filter=last_n_filter)
        rolling_stats = self._calculate_rolling_stats(weighted_games, self.rolling_window, last_n_filter=last_n_filter)
        sos_scores = self._calculate_comprehensive_sos(weighted_games, season_stats)
        
        # Calculate final power scores using layered approach
        power_scores = self._calculate_comprehensive_power_scores(
            season_stats, rolling_stats, sos_scores
        )
        
        rankings = [(team_id, teams_map.get(team_id, team_id), score) 
                   for team_id, score in power_scores.items()]
        rankings.sort(key=lambda x: x[2], reverse=True)
        
        # Return rankings and comprehensive data for export
        computation_data = {
            'game_results': game_results,
            'season_stats': season_stats,
            'rolling_stats': rolling_stats, 
            'sos_scores': sos_scores,
            'teams_map': teams_map,
            'power_scores': power_scores,
            'last_n_games': last_n_games or 0
        }
        
        return rankings, computation_data
    
    def _apply_game_weights(self, games: List[Dict]) -> List[Dict]:
        """Apply contextual weights to games (e.g., reduce Week 18 impact)"""
        weighted_games = []
        week18_games = 0
        
        for game in games:
            weighted_game = game.copy()
            
            # Significantly reduce impact of Week 18 games where teams often rest starters
            if game['week'] == 18:
                # Scale down the margin but preserve the game structure
                original_margin = game['margin']
                weighted_game['margin'] = int(original_margin * self.week18_weight)
                weighted_game['week18_adjusted'] = True
                week18_games += 1
            else:
                weighted_game['week18_adjusted'] = False
            
            weighted_games.append(weighted_game)
        
        if week18_games > 0:
            logger.info(f"Applied reduced weighting to {week18_games} Week 18 games")
        
        return weighted_games
    
    def _build_teams_map(self, teams_info: List[Dict]) -> Dict[str, str]:
        teams_map = {}
        for team_data in teams_info:
            team = team_data.get('team', {})
            team_id = team.get('id')
            team_name = team.get('displayName', team.get('name', 'Unknown'))
            if team_id:
                teams_map[str(team_id)] = team_name
        return teams_map
    
    def _extract_game_results(self, scoreboard_data: Dict, teams_map: Dict[str, str]) -> List[Dict]:
        games = []
        events = scoreboard_data.get('events', [])
        
        for event in events:
            if event.get('status', {}).get('type', {}).get('name') != 'STATUS_FINAL':
                continue
                
            # Extract week number from event or use default
            week_number = event.get('week_number', scoreboard_data.get('week', {}).get('number', 1))
            # Extract event id and date timestamp if available
            event_id = str(event.get('id', ''))
            raw_date = event.get('date') or event.get('startDate')
            ts = 0
            if raw_date:
                try:
                    ts = int(datetime.fromisoformat(str(raw_date).replace('Z', '+00:00')).timestamp())
                except Exception:
                    ts = 0
                
            competitions = event.get('competitions', [])
            for comp in competitions:
                competitors = comp.get('competitors', [])
                if len(competitors) != 2:
                    continue
                
                home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                
                if not home_team or not away_team:
                    continue
                
                home_id = home_team.get('team', {}).get('id')
                away_id = away_team.get('team', {}).get('id')
                home_score = int(home_team.get('score', 0))
                away_score = int(away_team.get('score', 0))
                
                if home_id and away_id:
                    games.append({
                        'event_id': event_id,
                        'home_team_id': str(home_id),
                        'away_team_id': str(away_id),
                        'home_score': home_score,
                        'away_score': away_score,
                        'margin': home_score - away_score,
                        'week': week_number,
                        'timestamp': ts
                    })
        
        return games
    
    def _calculate_season_stats(self, games: List[Dict], last_n_filter: Optional[Dict[str, set]] = None) -> Dict[str, Dict]:
        """Calculate full season statistics for each team"""
        team_stats = defaultdict(lambda: {
            'games_played': 0,
            'total_margin': 0,
            'wins': 0,
            'losses': 0,
            'points_for': 0,
            'points_against': 0,
            'opponents': [],
            'game_details': []
        })
        
        for game in games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            home_score = game['home_score']
            away_score = game['away_score']
            margin = game['margin']  # This may be weighted
            week = game['week']
            
            # Home team stats (respect last-n filter if provided)
            include_home = True
            include_away = True
            if last_n_filter is not None:
                event_id = game.get('event_id')
                include_home = event_id in last_n_filter.get(home_id, set())
                include_away = event_id in last_n_filter.get(away_id, set())

            # Home team stats
            if include_home:
                team_stats[home_id]['games_played'] += 1
                team_stats[home_id]['total_margin'] += margin
                team_stats[home_id]['points_for'] += home_score
                team_stats[home_id]['points_against'] += away_score
                team_stats[home_id]['opponents'].append(away_id)
                team_stats[home_id]['game_details'].append({
                'week': week, 'opponent': away_id, 'margin': margin,
                'points_for': home_score, 'points_against': away_score, 'event_id': game.get('event_id'), 'timestamp': game.get('timestamp', 0)
                })
                # Use original margin for win/loss (not weighted margin)
                original_margin = home_score - away_score
                if original_margin > 0:
                    team_stats[home_id]['wins'] += 1
                else:
                    team_stats[home_id]['losses'] += 1
            
            # Away team stats
            if include_away:
                team_stats[away_id]['games_played'] += 1
                team_stats[away_id]['total_margin'] -= margin
                team_stats[away_id]['points_for'] += away_score
                team_stats[away_id]['points_against'] += home_score
                team_stats[away_id]['opponents'].append(home_id)
                team_stats[away_id]['game_details'].append({
                    'week': week, 'opponent': home_id, 'margin': -margin,
                    'points_for': away_score, 'points_against': home_score, 'event_id': game.get('event_id'), 'timestamp': game.get('timestamp', 0)
                })
                if original_margin < 0:
                    team_stats[away_id]['wins'] += 1
                else:
                    team_stats[away_id]['losses'] += 1
        
        # Calculate derived statistics
        for team_id in team_stats:
            stats = team_stats[team_id]
            games_played = stats['games_played']
            if games_played > 0:
                stats['avg_margin'] = stats['total_margin'] / games_played
                stats['avg_points_for'] = stats['points_for'] / games_played  
                stats['avg_points_against'] = stats['points_against'] / games_played
                stats['win_pct'] = stats['wins'] / games_played
            else:
                stats['avg_margin'] = 0
                stats['avg_points_for'] = 0
                stats['avg_points_against'] = 0
                stats['win_pct'] = 0
        
        return dict(team_stats)
    
    def _calculate_rolling_stats(self, games: List[Dict], window: int, last_n_filter: Optional[Dict[str, set]] = None) -> Dict[str, Dict]:
        """Calculate rolling statistics.

        If last_n_filter is provided, compute rolling stats over each team's
        most recent min(window, last_n) games. Otherwise, fall back to week-based window.
        """
        if last_n_filter:
            # Build per-team filter for last min(window, available) events by timestamp
            per_team_events: Dict[str, List[Tuple[int, str]]] = {}
            for g in games:
                eid = g.get('event_id')
                ts = g.get('timestamp', 0)
                per_team_events.setdefault(g['home_team_id'], []).append((ts, eid))
                per_team_events.setdefault(g['away_team_id'], []).append((ts, eid))
            rolling_filter: Dict[str, set] = {}
            for team, seq in per_team_events.items():
                seq_sorted = sorted(seq, key=lambda x: x[0])
                keep = {eid for _, eid in seq_sorted[-min(window, len(seq_sorted)):]}  # last window games
                rolling_filter[team] = keep
            return self._calculate_season_stats(games, last_n_filter=rolling_filter)

        # Legacy behavior: week-based window
        sorted_games = sorted(games, key=lambda x: x['week'])
        max_week = max(game['week'] for game in games) if games else 18
        target_weeks = list(range(max(1, max_week - window + 1), max_week + 1))
        recent_games = [g for g in sorted_games if g['week'] in target_weeks]
        logger.info(f"Rolling stats using weeks {target_weeks} ({len(recent_games)} games)")
        return self._calculate_season_stats(recent_games)

    def _build_last_n_filter(self, games: List[Dict], last_n_games: int) -> Dict[str, set]:
        """Build a per-team set of allowed event_ids for the last N games by timestamp."""
        per_team: Dict[str, List[Tuple[int, str]]] = {}
        for g in games:
            eid = g.get('event_id')
            ts = g.get('timestamp', 0)
            per_team.setdefault(g['home_team_id'], []).append((ts, eid))
            per_team.setdefault(g['away_team_id'], []).append((ts, eid))
        allowed: Dict[str, set] = {}
        for team, seq in per_team.items():
            seq_sorted = sorted(seq, key=lambda x: x[0])
            keep = {eid for _, eid in seq_sorted[-min(last_n_games, len(seq_sorted)):]}
            allowed[team] = keep
        return allowed
    
    def _calculate_comprehensive_sos(self, games: List[Dict], season_stats: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate strength of schedule based on opponent season performance"""
        sos_scores = {}
        
        for team_id, stats in season_stats.items():
            opponents = stats['opponents']
            if not opponents:
                sos_scores[team_id] = 0
                continue
            
            # Calculate average opponent strength using multiple metrics
            opponent_margins = []
            opponent_win_pcts = []
            
            for opponent_id in opponents:
                if opponent_id in season_stats:
                    opponent_margins.append(season_stats[opponent_id]['avg_margin'])
                    opponent_win_pcts.append(season_stats[opponent_id]['win_pct'])
            
            if opponent_margins and opponent_win_pcts:
                # Weighted SOS combining margin and win percentage
                avg_opp_margin = np.mean(opponent_margins)
                avg_opp_win_pct = np.mean(opponent_win_pcts)
                sos_scores[team_id] = (avg_opp_margin * 0.6 + (avg_opp_win_pct - 0.5) * 20 * 0.4)
            else:
                sos_scores[team_id] = 0
        
        return sos_scores
    
    def _calculate_comprehensive_power_scores(self, season_stats: Dict[str, Dict], 
                                            rolling_stats: Dict[str, Dict], 
                                            sos_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate final power scores using balanced approach"""
        power_scores = {}
        
        for team_id in season_stats:
            if season_stats[team_id]['games_played'] == 0:
                power_scores[team_id] = 0
                continue
            
            # Season-long performance (most important)
            season_margin = season_stats[team_id]['avg_margin']
            season_win_pct = season_stats[team_id]['win_pct']
            
            # Rolling performance (moderate weight)
            rolling_margin = rolling_stats.get(team_id, {}).get('avg_margin', season_margin)
            rolling_games = rolling_stats.get(team_id, {}).get('games_played', 0)
            
            # Strength of schedule adjustment
            sos = sos_scores.get(team_id, 0)
            
            # Calculate weighted power score with more conservative approach
            base_score = (season_margin * self.weights['season_avg_margin'] + 
                         rolling_margin * self.weights['rolling_avg_margin'] + 
                         sos * self.weights['sos'])
            
            # Very limited recency factor to prevent single-game distortion
            if rolling_games >= 3:
                recency_factor = (rolling_margin - season_margin) * self.weights['recency_factor']
                base_score += recency_factor
            
            # More conservative win percentage component 
            win_pct_bonus = (season_win_pct - 0.5) * 3  # Reduced from 5 to 3
            
            power_scores[team_id] = base_score + win_pct_bonus
        
        return power_scores
    
    def compute_with_confidence(self, scoreboard_data: Dict, teams_info: List[Dict], 
                               confidence_level: float = 0.95) -> Tuple[List[PowerRankingWithConfidence], Dict[str, Any]]:
        """
        Compute power rankings with confidence intervals.
        
        Args:
            scoreboard_data: Game data
            teams_info: Team information
            confidence_level: Confidence level for intervals (default 0.95)
            
        Returns:
            Tuple of (rankings with confidence, computation data)
        """
        # Get standard rankings first
        rankings, computation_data = self.compute(scoreboard_data, teams_info)
        
        if not rankings:
            return [], computation_data
        
        # Calculate confidence intervals
        rankings_with_confidence = []
        
        for team_id, team_name, power_score in rankings:
            confidence_interval = self._calculate_confidence_interval(
                team_id, computation_data, confidence_level
            )
            
            # Calculate stability score
            stability = self._calculate_team_stability(team_id, computation_data)
            
            ranking_with_conf = PowerRankingWithConfidence(
                team_id=team_id,
                team_name=team_name,
                power_score=power_score,
                confidence_lower=confidence_interval[0],
                confidence_upper=confidence_interval[1],
                confidence_level=confidence_level,
                sample_size=computation_data.get('season_stats', {}).get(team_id, {}).get('games_played', 0),
                stability_score=stability
            )
            
            rankings_with_confidence.append(ranking_with_conf)
        
        return rankings_with_confidence, computation_data
    
    def predict_game_with_uncertainty(self, home_team_id: str, away_team_id: str,
                                    power_scores: Dict[str, float],
                                    computation_data: Dict[str, Any] = None,
                                    teams_map: Dict[str, str] = None) -> PredictionWithUncertainty:
        """
        Predict game outcome with uncertainty quantification.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            power_scores: Dictionary of power scores
            computation_data: Additional computation data
            teams_map: Team ID to name mapping
            
        Returns:
            PredictionWithUncertainty object
        """
        if computation_data is None:
            computation_data = {}
        if teams_map is None:
            teams_map = {}
        
        # Get team power scores
        home_power = power_scores.get(home_team_id, 0.0)
        away_power = power_scores.get(away_team_id, 0.0)
        
        # Basic prediction: power difference + home field advantage
        predicted_margin = (home_power - away_power) + 2.5
        
        # Calculate prediction variance based on multiple factors
        prediction_variance = self._calculate_prediction_variance(
            home_team_id, away_team_id, computation_data
        )
        
        # Calculate confidence interval (assuming normal distribution)
        std_error = np.sqrt(prediction_variance)
        confidence_interval = (
            predicted_margin - 1.96 * std_error,  # 95% CI lower bound
            predicted_margin + 1.96 * std_error   # 95% CI upper bound
        )
        
        # Calculate win probability using logistic function
        win_prob_home = self._calculate_win_probability(predicted_margin, std_error)
        
        # Calculate overall model confidence
        model_confidence = self._calculate_model_confidence(
            home_team_id, away_team_id, computation_data, std_error
        )
        
        return PredictionWithUncertainty(
            home_team=teams_map.get(home_team_id, home_team_id),
            away_team=teams_map.get(away_team_id, away_team_id),
            predicted_margin=predicted_margin,
            confidence_interval=confidence_interval,
            win_probability_home=win_prob_home,
            prediction_variance=prediction_variance,
            model_confidence=model_confidence
        )
    
    def _calculate_confidence_interval(self, team_id: str, computation_data: Dict[str, Any], 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for a team's power score."""
        season_stats = computation_data.get('season_stats', {}).get(team_id, {})
        power_score = computation_data.get('power_scores', {}).get(team_id, 0.0)
        
        # Get sample size and variance estimates
        games_played = season_stats.get('games_played', 0)
        if games_played == 0:
            return (power_score, power_score)  # No uncertainty with no data
        
        # Estimate standard error based on margin variance and sample size
        game_details = season_stats.get('game_details', [])
        if game_details:
            margins = [game['margin'] for game in game_details]
            margin_std = np.std(margins) if len(margins) > 1 else 5.0  # Default NFL margin std
        else:
            margin_std = 5.0  # Default NFL game margin standard deviation
        
        # Standard error decreases with square root of sample size
        standard_error = margin_std / np.sqrt(max(1, games_played))
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
        
        margin_of_error = z_score * standard_error
        
        return (power_score - margin_of_error, power_score + margin_of_error)
    
    def _calculate_team_stability(self, team_id: str, computation_data: Dict[str, Any]) -> float:
        """Calculate stability score for a team's rankings."""
        season_stats = computation_data.get('season_stats', {}).get(team_id, {})
        rolling_stats = computation_data.get('rolling_stats', {}).get(team_id, {})
        
        if not season_stats or not rolling_stats:
            return 0.5  # Neutral stability
        
        # Compare season average to rolling average
        season_margin = season_stats.get('avg_margin', 0.0)
        rolling_margin = rolling_stats.get('avg_margin', season_margin)
        
        # Stability is higher when season and rolling averages are similar
        margin_difference = abs(season_margin - rolling_margin)
        
        # Normalize to 0-1 scale (higher = more stable)
        stability = max(0.0, 1.0 - (margin_difference / 10.0))  # 10 point difference = 0 stability
        
        return min(1.0, stability)
    
    def _calculate_prediction_variance(self, home_team_id: str, away_team_id: str,
                                     computation_data: Dict[str, Any]) -> float:
        """Calculate variance of game prediction."""
        season_stats = computation_data.get('season_stats', {})
        
        # Base variance from NFL game unpredictability
        base_variance = 49.0  # ~7 point standard deviation for NFL games
        
        # Adjust based on team data quality
        home_stats = season_stats.get(home_team_id, {})
        away_stats = season_stats.get(away_team_id, {})
        
        home_games = home_stats.get('games_played', 0)
        away_games = away_stats.get('games_played', 0)
        
        # Higher variance for teams with fewer games (less reliable data)
        sample_size_factor = 1.0 + max(0, (10 - min(home_games, away_games)) / 10.0)
        
        # Adjust for team consistency (more consistent teams = lower variance)
        home_details = home_stats.get('game_details', [])
        away_details = away_stats.get('game_details', [])
        
        consistency_factor = 1.0
        if home_details and away_details:
            home_margins = [game['margin'] for game in home_details]
            away_margins = [game['margin'] for game in away_details]
            
            if len(home_margins) > 1 and len(away_margins) > 1:
                home_consistency = np.std(home_margins)
                away_consistency = np.std(away_margins)
                avg_consistency = (home_consistency + away_consistency) / 2
                
                # Higher standard deviation = less consistent = higher prediction variance
                consistency_factor = 1.0 + (avg_consistency / 14.0)  # Normalize by typical margin range
        
        final_variance = base_variance * sample_size_factor * consistency_factor
        
        return min(100.0, max(25.0, final_variance))  # Cap between reasonable bounds
    
    def _calculate_win_probability(self, predicted_margin: float, std_error: float) -> float:
        """Calculate win probability using logistic function."""
        if std_error == 0:
            return 1.0 if predicted_margin > 0 else 0.0
        
        # Convert to z-score
        z_score = predicted_margin / max(0.1, std_error)
        
        # Use cumulative normal distribution to get probability
        try:
            from scipy.stats import norm
            win_prob = norm.cdf(z_score)
        except ImportError:
            # Fallback approximation using logistic function
            win_prob = 1 / (1 + np.exp(-predicted_margin / 3.0))
        
        return max(0.01, min(0.99, win_prob))
    
    def _calculate_model_confidence(self, home_team_id: str, away_team_id: str,
                                  computation_data: Dict[str, Any], std_error: float) -> float:
        """Calculate overall confidence in the model prediction."""
        season_stats = computation_data.get('season_stats', {})
        
        # Factors that affect confidence:
        # 1. Sample size (more games = higher confidence)
        home_games = season_stats.get(home_team_id, {}).get('games_played', 0)
        away_games = season_stats.get(away_team_id, {}).get('games_played', 0)
        
        sample_confidence = min(1.0, (min(home_games, away_games) / 10.0))  # Max confidence at 10 games
        
        # 2. Prediction precision (lower std error = higher confidence)
        precision_confidence = max(0.1, 1.0 - (std_error / 10.0))  # Normalize by typical error
        
        # 3. Power score magnitude (larger differences = higher confidence)
        home_power = computation_data.get('power_scores', {}).get(home_team_id, 0.0)
        away_power = computation_data.get('power_scores', {}).get(away_team_id, 0.0)
        power_difference = abs(home_power - away_power)
        
        magnitude_confidence = min(1.0, power_difference / 14.0)  # Max confidence at 14 point difference
        
        # Combine factors (weighted average)
        overall_confidence = (
            0.4 * sample_confidence +
            0.4 * precision_confidence +
            0.2 * magnitude_confidence
        )
        
        return max(0.1, min(0.95, overall_confidence))
