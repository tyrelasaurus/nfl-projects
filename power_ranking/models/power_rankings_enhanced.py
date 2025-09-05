"""
Enhanced Power Rankings Model with comprehensive error handling.
Extends the original power rankings model with structured exception management and logging.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

# Import custom exceptions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import (
    PowerRankingCalculationError, StatisticalCalculationError, StrengthOfScheduleError,
    DataProcessingError, DataIncompleteError, ConfigurationError,
    handle_exception_with_recovery, log_and_handle_error
)

# Import base model
from power_rankings import PowerRankModel, PowerRankingWithConfidence, PredictionWithUncertainty

logger = logging.getLogger(__name__)

class EnhancedPowerRankModel(PowerRankModel):
    """Enhanced power ranking model with comprehensive error handling."""
    
    def __init__(self, weights: Dict[str, float] = None, 
                 strict_validation: bool = True,
                 fallback_enabled: bool = True):
        """
        Initialize enhanced power ranking model.
        
        Args:
            weights: Custom weights for ranking calculations
            strict_validation: Enable strict data validation
            fallback_enabled: Enable fallback calculations for missing data
        """
        try:
            super().__init__(weights)
            self.strict_validation = strict_validation
            self.fallback_enabled = fallback_enabled
            self.calculation_cache = {}
            
            logger.info(f"Enhanced power ranking model initialized")
            logger.debug(f"Weights: {self.weights}")
            logger.debug(f"Strict validation: {strict_validation}, Fallbacks: {fallback_enabled}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize power ranking model: {str(e)}",
                context={'weights': weights, 'error_type': type(e).__name__}
            ) from e
    
    @handle_exception_with_recovery
    def compute(self, scoreboard_data: Dict, teams_info: List[Dict]) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        """
        Compute power rankings with enhanced error handling.
        
        Args:
            scoreboard_data: ESPN scoreboard data
            teams_info: Team information data
            
        Returns:
            Tuple of (rankings, computation_data)
            
        Raises:
            DataProcessingError: For data validation errors
            PowerRankingCalculationError: For calculation errors
        """
        logger.info("Starting enhanced power ranking computation")
        
        try:
            # Validate input data
            self._validate_input_data(scoreboard_data, teams_info)
            
            # Call parent computation with error handling
            rankings, computation_data = super().compute(scoreboard_data, teams_info)
            
            # Validate output
            self._validate_computation_results(rankings, computation_data)
            
            logger.info(f"Power ranking computation completed: {len(rankings)} teams ranked")
            return rankings, computation_data
            
        except Exception as e:
            context = {
                'teams_count': len(teams_info) if teams_info else 0,
                'events_count': len(scoreboard_data.get('events', [])) if scoreboard_data else 0
            }
            log_and_handle_error(e, logger, context=context)
            
            if self.fallback_enabled:
                logger.warning("Attempting fallback computation")
                return self._fallback_computation(teams_info)
            
            raise
    
    def _validate_input_data(self, scoreboard_data: Dict, teams_info: List[Dict]) -> None:
        """
        Validate input data quality and completeness.
        
        Args:
            scoreboard_data: Scoreboard data to validate
            teams_info: Teams information to validate
            
        Raises:
            DataIncompleteError: For missing or incomplete data
            DataProcessingError: For data quality issues
        """
        logger.debug("Validating input data")
        
        # Validate scoreboard data
        if not scoreboard_data:
            raise DataIncompleteError(
                "Scoreboard data is empty or None",
                missing_fields=['scoreboard_data'],
                context={'data_type': 'scoreboard'}
            )
        
        events = scoreboard_data.get('events', [])
        if not events:
            raise DataIncompleteError(
                "No events found in scoreboard data",
                missing_fields=['events'],
                completeness_percentage=0.0,
                context={'scoreboard_keys': list(scoreboard_data.keys())}
            )
        
        # Validate teams info
        if not teams_info:
            raise DataIncompleteError(
                "Teams information is empty or None",
                missing_fields=['teams_info'],
                context={'data_type': 'teams_info'}
            )
        
        # Check for minimum data requirements
        if len(events) < 10:  # Arbitrary minimum for meaningful rankings
            logger.warning(f"Low event count: {len(events)} games (minimum recommended: 10)")
            
        if len(teams_info) < 20:  # NFL should have 32 teams
            logger.warning(f"Low team count: {len(teams_info)} teams (expected: 32)")
        
        logger.debug(f"Input validation passed: {len(events)} events, {len(teams_info)} teams")
    
    def _validate_computation_results(self, rankings: List[Tuple[str, str, float]], 
                                    computation_data: Dict[str, Any]) -> None:
        """
        Validate computation results.
        
        Args:
            rankings: Computed rankings to validate
            computation_data: Computation metadata to validate
            
        Raises:
            PowerRankingCalculationError: For invalid results
        """
        if not rankings:
            raise PowerRankingCalculationError(
                "No rankings were computed",
                calculation_step="final_validation",
                context={'rankings_count': 0}
            )
        
        # Check for valid power scores
        invalid_teams = []
        for team_id, team_name, power_score in rankings:
            if not isinstance(power_score, (int, float)):
                invalid_teams.append(team_name)
            elif np.isnan(power_score) or np.isinf(power_score):
                invalid_teams.append(team_name)
        
        if invalid_teams:
            raise PowerRankingCalculationError(
                f"Invalid power scores computed for teams: {invalid_teams}",
                calculation_step="score_validation",
                teams_affected=invalid_teams,
                context={'total_teams': len(rankings)}
            )
        
        # Check for reasonable score distribution
        scores = [score for _, _, score in rankings]
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std < 0.1:  # Scores too similar
                logger.warning(f"Power scores have very low variance (std: {score_std:.3f})")
            elif score_std > 50:  # Scores too spread out
                logger.warning(f"Power scores have very high variance (std: {score_std:.3f})")
        
        logger.debug("Computation results validation passed")
    
    @handle_exception_with_recovery
    def _calculate_comprehensive_power_scores(self, season_stats: Dict[str, Dict], 
                                            sos_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate comprehensive power scores with enhanced error handling.
        
        Args:
            season_stats: Season statistics for each team
            sos_scores: Strength of schedule scores
            
        Returns:
            Dictionary of team power scores
            
        Raises:
            StatisticalCalculationError: For calculation errors
        """
        logger.debug("Calculating comprehensive power scores with enhanced error handling")
        
        try:
            power_scores = {}
            teams_with_errors = []
            
            for team_id, stats in season_stats.items():
                try:
                    power_score = self._calculate_single_team_power_score(team_id, stats, sos_scores)
                    power_scores[team_id] = power_score
                    
                except Exception as e:
                    logger.warning(f"Error calculating power score for {team_id}: {e}")
                    teams_with_errors.append(team_id)
                    
                    if self.fallback_enabled:
                        # Use fallback calculation
                        fallback_score = self._calculate_fallback_power_score(team_id, stats)
                        power_scores[team_id] = fallback_score
                        logger.info(f"Used fallback power score for {team_id}: {fallback_score:.2f}")
                    elif not self.strict_validation:
                        # Use default score
                        power_scores[team_id] = 50.0  # Neutral score
                        logger.info(f"Used default power score for {team_id}: 50.0")
                    else:
                        # Re-raise in strict mode
                        raise
            
            if not power_scores:
                raise StatisticalCalculationError(
                    "Failed to calculate power scores for any team",
                    statistic_type="comprehensive_power_scores",
                    sample_size=len(season_stats),
                    context={'teams_with_errors': teams_with_errors}
                )
            
            if teams_with_errors:
                logger.warning(f"Teams with calculation errors: {teams_with_errors}")
            
            logger.debug(f"Calculated power scores for {len(power_scores)} teams")
            return power_scores
            
        except Exception as e:
            context = {
                'season_stats_count': len(season_stats),
                'sos_scores_count': len(sos_scores),
                'operation': 'comprehensive_power_scores'
            }
            log_and_handle_error(e, logger, context=context)
            raise
    
    def _calculate_single_team_power_score(self, team_id: str, stats: Dict, 
                                         sos_scores: Dict[str, float]) -> float:
        """
        Calculate power score for a single team with error handling.
        
        Args:
            team_id: Team identifier
            stats: Team statistics
            sos_scores: Strength of schedule scores
            
        Returns:
            Team power score
            
        Raises:
            StatisticalCalculationError: For calculation errors
        """
        try:
            # Get individual components
            season_margin = stats.get('avg_margin', 0.0)
            rolling_margin = stats.get('rolling_avg_margin', season_margin)  # Fallback to season avg
            sos = sos_scores.get(team_id, 0.0)
            
            # Validate components
            if not all(isinstance(x, (int, float)) for x in [season_margin, rolling_margin, sos]):
                raise StatisticalCalculationError(
                    f"Invalid statistics for team {team_id}",
                    statistic_type="team_power_components",
                    context={
                        'team_id': team_id,
                        'season_margin': season_margin,
                        'rolling_margin': rolling_margin,
                        'sos': sos
                    }
                )
            
            if any(np.isnan(x) or np.isinf(x) for x in [season_margin, rolling_margin, sos]):
                raise StatisticalCalculationError(
                    f"NaN or infinite values in statistics for team {team_id}",
                    statistic_type="team_power_components",
                    context={
                        'team_id': team_id,
                        'season_margin': season_margin,
                        'rolling_margin': rolling_margin, 
                        'sos': sos
                    }
                )
            
            # Calculate weighted score
            power_score = (
                self.weights['season_avg_margin'] * season_margin +
                self.weights['rolling_avg_margin'] * rolling_margin +
                self.weights['sos'] * sos
            )
            
            # Add league baseline
            power_score += 50  # NFL league average baseline
            
            # Apply recency factor if available
            if 'recent_performance' in stats:
                recent_factor = stats['recent_performance']
                if isinstance(recent_factor, (int, float)) and not (np.isnan(recent_factor) or np.isinf(recent_factor)):
                    power_score += self.weights['recency_factor'] * recent_factor
            
            return float(power_score)
            
        except Exception as e:
            raise StatisticalCalculationError(
                f"Error calculating power score for team {team_id}: {str(e)}",
                statistic_type="single_team_power_score",
                context={'team_id': team_id, 'stats_keys': list(stats.keys())}
            ) from e
    
    def _calculate_fallback_power_score(self, team_id: str, stats: Dict) -> float:
        """
        Calculate fallback power score when normal calculation fails.
        
        Args:
            team_id: Team identifier
            stats: Available team statistics
            
        Returns:
            Fallback power score
        """
        logger.debug(f"Calculating fallback power score for {team_id}")
        
        # Use simple average margin if available
        if 'avg_margin' in stats and isinstance(stats['avg_margin'], (int, float)):
            margin = stats['avg_margin']
            if not (np.isnan(margin) or np.isinf(margin)):
                return 50.0 + margin  # League baseline + margin
        
        # Use win percentage if available
        if 'win_pct' in stats and isinstance(stats['win_pct'], (int, float)):
            win_pct = stats['win_pct']
            if not (np.isnan(win_pct) or np.isinf(win_pct)) and 0 <= win_pct <= 1:
                # Convert win percentage to power score (rough approximation)
                return 30.0 + (win_pct * 40.0)  # 30-70 range based on win %
        
        # Ultimate fallback - league average
        logger.warning(f"Using ultimate fallback score for {team_id}")
        return 50.0
    
    @handle_exception_with_recovery
    def _calculate_comprehensive_sos(self, games: List[Dict], season_stats: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate strength of schedule with enhanced error handling.
        
        Args:
            games: List of game data
            season_stats: Season statistics for all teams
            
        Returns:
            Dictionary of SOS scores
            
        Raises:
            StrengthOfScheduleError: For SOS calculation errors
        """
        logger.debug("Calculating strength of schedule with enhanced error handling")
        
        try:
            sos_scores = {}
            teams_with_errors = []
            
            # Build opponent lists
            team_opponents = defaultdict(list)
            for game in games:
                home_team = game.get('home_team_id')
                away_team = game.get('away_team_id')
                
                if home_team and away_team and home_team != away_team:
                    team_opponents[home_team].append(away_team)
                    team_opponents[away_team].append(home_team)
            
            # Calculate SOS for each team
            for team_id in season_stats:
                try:
                    opponents = team_opponents.get(team_id, [])
                    if not opponents:
                        logger.warning(f"No opponents found for {team_id}")
                        if self.fallback_enabled:
                            sos_scores[team_id] = 0.0  # Neutral SOS
                        continue
                    
                    sos_score = self._calculate_single_team_sos(team_id, opponents, season_stats)
                    sos_scores[team_id] = sos_score
                    
                except Exception as e:
                    logger.warning(f"Error calculating SOS for {team_id}: {e}")
                    teams_with_errors.append(team_id)
                    
                    if self.fallback_enabled:
                        sos_scores[team_id] = 0.0  # Neutral SOS as fallback
                        logger.info(f"Used fallback SOS for {team_id}: 0.0")
                    elif not self.strict_validation:
                        sos_scores[team_id] = 0.0
                    else:
                        raise StrengthOfScheduleError(
                            f"Failed to calculate SOS for {team_id}: {str(e)}",
                            team_id=team_id,
                            opponents=opponents,
                            context={'error': str(e)}
                        ) from e
            
            if teams_with_errors:
                logger.warning(f"Teams with SOS calculation errors: {teams_with_errors}")
            
            logger.debug(f"Calculated SOS for {len(sos_scores)} teams")
            return sos_scores
            
        except Exception as e:
            context = {
                'games_count': len(games),
                'season_stats_count': len(season_stats),
                'operation': 'comprehensive_sos'
            }
            log_and_handle_error(e, logger, context=context)
            raise
    
    def _calculate_single_team_sos(self, team_id: str, opponents: List[str], 
                                 season_stats: Dict[str, Dict]) -> float:
        """
        Calculate SOS for a single team.
        
        Args:
            team_id: Team identifier
            opponents: List of opponent team IDs
            season_stats: Season statistics for all teams
            
        Returns:
            Team SOS score
        """
        if not opponents:
            return 0.0
        
        opponent_margins = []
        opponent_win_pcts = []
        
        for opp_id in opponents:
            if opp_id in season_stats:
                opp_stats = season_stats[opp_id]
                
                # Get opponent margin
                if 'avg_margin' in opp_stats:
                    margin = opp_stats['avg_margin']
                    if isinstance(margin, (int, float)) and not (np.isnan(margin) or np.isinf(margin)):
                        opponent_margins.append(margin)
                
                # Get opponent win percentage
                if 'win_pct' in opp_stats:
                    win_pct = opp_stats['win_pct']
                    if isinstance(win_pct, (int, float)) and not (np.isnan(win_pct) or np.isinf(win_pct)):
                        opponent_win_pcts.append(win_pct)
        
        # Calculate SOS components
        margin_component = np.mean(opponent_margins) if opponent_margins else 0.0
        win_pct_component = np.mean(opponent_win_pcts) if opponent_win_pcts else 0.5
        
        # Convert win percentage to margin equivalent (rough approximation)
        win_pct_margin = (win_pct_component - 0.5) * 20  # Scale to roughly match margin range
        
        # Weighted combination (60% margin, 40% win percentage)
        sos_score = 0.6 * margin_component + 0.4 * win_pct_margin
        
        return float(sos_score)
    
    def _fallback_computation(self, teams_info: List[Dict]) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        """
        Fallback computation when normal calculation fails.
        
        Args:
            teams_info: Basic team information
            
        Returns:
            Tuple of (fallback_rankings, computation_data)
        """
        logger.warning("Using fallback computation - rankings will be basic")
        
        rankings = []
        for i, team_info in enumerate(teams_info):
            team_id = team_info.get('team', {}).get('id', f'team_{i}')
            team_name = team_info.get('team', {}).get('displayName', f'Team {i}')
            
            # Assign neutral power score with slight variation
            power_score = 50.0 + np.random.normal(0, 2)  # Slight randomization
            rankings.append((team_id, team_name, power_score))
        
        # Sort by power score (descending)
        rankings.sort(key=lambda x: x[2], reverse=True)
        
        computation_data = {
            'fallback_used': True,
            'computation_method': 'fallback_neutral_scores',
            'teams_processed': len(rankings)
        }
        
        logger.info(f"Fallback computation completed: {len(rankings)} teams with neutral scores")
        return rankings, computation_data
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and health information.
        
        Returns:
            Dictionary with diagnostic information
        """
        return {
            'model_type': 'EnhancedPowerRankModel',
            'weights': self.weights.copy(),
            'strict_validation': self.strict_validation,
            'fallback_enabled': self.fallback_enabled,
            'rolling_window': self.rolling_window,
            'week18_weight': self.week18_weight,
            'cache_size': len(self.calculation_cache)
        }