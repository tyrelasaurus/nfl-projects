"""
Unit tests for PowerRankModel statistical functions.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import defaultdict

from power_ranking.models.power_rankings import PowerRankModel


class TestPowerRankModel:
    """Test cases for PowerRankModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a PowerRankModel instance with default weights."""
        return PowerRankModel()
    
    @pytest.fixture
    def custom_model(self):
        """Create a PowerRankModel instance with custom weights."""
        weights = {
            'season_avg_margin': 0.6,
            'rolling_avg_margin': 0.2,
            'sos': 0.15,
            'recency_factor': 0.05
        }
        return PowerRankModel(weights=weights)
    
    @pytest.fixture
    def sample_teams_info(self):
        """Sample teams data for testing."""
        return [
            {'team': {'id': '1', 'displayName': 'Buffalo Bills'}},
            {'team': {'id': '11', 'displayName': 'Kansas City Chiefs'}},
            {'team': {'id': '15', 'displayName': 'Miami Dolphins'}},
            {'team': {'id': '12', 'displayName': 'Las Vegas Raiders'}},
        ]
    
    @pytest.fixture
    def sample_scoreboard_data(self):
        """Sample scoreboard data with completed games."""
        return {
            'events': [
                {
                    'status': {'type': {'name': 'STATUS_FINAL'}},
                    'week_number': 1,
                    'competitions': [{
                        'competitors': [
                            {'homeAway': 'home', 'team': {'id': '1'}, 'score': '31'},
                            {'homeAway': 'away', 'team': {'id': '15'}, 'score': '10'}
                        ]
                    }]
                },
                {
                    'status': {'type': {'name': 'STATUS_FINAL'}},
                    'week_number': 1,
                    'competitions': [{
                        'competitors': [
                            {'homeAway': 'home', 'team': {'id': '11'}, 'score': '27'},
                            {'homeAway': 'away', 'team': {'id': '12'}, 'score': '17'}
                        ]
                    }]
                },
                {
                    'status': {'type': {'name': 'STATUS_FINAL'}},
                    'week_number': 2,
                    'competitions': [{
                        'competitors': [
                            {'homeAway': 'home', 'team': {'id': '15'}, 'score': '24'},
                            {'homeAway': 'away', 'team': {'id': '11'}, 'score': '21'}
                        ]
                    }]
                }
            ]
        }
    
    def test_model_initialization_default_weights(self, model):
        """Test model initializes with correct default weights."""
        expected_weights = {
            'season_avg_margin': 0.5,
            'rolling_avg_margin': 0.25,
            'sos': 0.2,
            'recency_factor': 0.05
        }
        assert model.weights == expected_weights
        assert model.rolling_window == 5
        assert model.week18_weight == 0.3
    
    def test_model_initialization_custom_weights(self, custom_model):
        """Test model initializes with custom weights."""
        expected_weights = {
            'season_avg_margin': 0.6,
            'rolling_avg_margin': 0.2,
            'sos': 0.15,
            'recency_factor': 0.05
        }
        assert custom_model.weights == expected_weights
    
    def test_build_teams_map(self, model, sample_teams_info):
        """Test building teams map from teams info."""
        teams_map = model._build_teams_map(sample_teams_info)
        
        expected = {
            '1': 'Buffalo Bills',
            '11': 'Kansas City Chiefs', 
            '15': 'Miami Dolphins',
            '12': 'Las Vegas Raiders'
        }
        assert teams_map == expected
    
    def test_build_teams_map_missing_data(self, model):
        """Test teams map handles missing data gracefully."""
        incomplete_teams = [
            {'team': {'id': '1', 'displayName': 'Buffalo Bills'}},
            {'team': {'displayName': 'No ID Team'}},  # Missing ID
            {'team': {'id': '11'}},  # Missing name
            {}  # Missing team data
        ]
        
        teams_map = model._build_teams_map(incomplete_teams)
        
        assert teams_map['1'] == 'Buffalo Bills'
        assert teams_map['11'] == 'Unknown'
        assert len(teams_map) == 2  # Only valid entries
    
    def test_extract_game_results(self, model, sample_scoreboard_data, sample_teams_info):
        """Test extracting game results from scoreboard data."""
        teams_map = model._build_teams_map(sample_teams_info)
        games = model._extract_game_results(sample_scoreboard_data, teams_map)
        
        assert len(games) == 3
        
        # Check first game: Bills vs Dolphins
        game1 = games[0]
        assert game1['home_team_id'] == '1'
        assert game1['away_team_id'] == '15'
        assert game1['home_score'] == 31
        assert game1['away_score'] == 10
        assert game1['margin'] == 21
        assert game1['week'] == 1
        
        # Check second game: Chiefs vs Raiders  
        game2 = games[1]
        assert game2['home_team_id'] == '11'
        assert game2['away_team_id'] == '12'
        assert game2['margin'] == 10
        
        # Check third game: Dolphins vs Chiefs
        game3 = games[2]
        assert game3['home_team_id'] == '15'
        assert game3['away_team_id'] == '11'
        assert game3['margin'] == 3
        assert game3['week'] == 2
    
    def test_extract_game_results_filters_incomplete_games(self, model, sample_teams_info):
        """Test that incomplete games are filtered out."""
        scoreboard_with_incomplete = {
            'events': [
                # Valid completed game
                {
                    'status': {'type': {'name': 'STATUS_FINAL'}},
                    'week_number': 1,
                    'competitions': [{
                        'competitors': [
                            {'homeAway': 'home', 'team': {'id': '1'}, 'score': '31'},
                            {'homeAway': 'away', 'team': {'id': '15'}, 'score': '10'}
                        ]
                    }]
                },
                # In-progress game (should be filtered out)
                {
                    'status': {'type': {'name': 'STATUS_IN_PROGRESS'}},
                    'week_number': 1,
                    'competitions': [{
                        'competitors': [
                            {'homeAway': 'home', 'team': {'id': '11'}, 'score': '14'},
                            {'homeAway': 'away', 'team': {'id': '12'}, 'score': '7'}
                        ]
                    }]
                },
                # Missing competitors (should be filtered out)
                {
                    'status': {'type': {'name': 'STATUS_FINAL'}},
                    'week_number': 1,
                    'competitions': [{'competitors': []}]
                }
            ]
        }
        
        teams_map = model._build_teams_map(sample_teams_info)
        games = model._extract_game_results(scoreboard_with_incomplete, teams_map)
        
        assert len(games) == 1  # Only the valid completed game
        assert games[0]['home_team_id'] == '1'
    
    def test_apply_game_weights_week18(self, model):
        """Test that Week 18 games get reduced weighting."""
        games = [
            {'home_team_id': '1', 'away_team_id': '2', 'margin': 20, 'week': 17, 'home_score': 30, 'away_score': 10},
            {'home_team_id': '3', 'away_team_id': '4', 'margin': 14, 'week': 18, 'home_score': 21, 'away_score': 7}
        ]
        
        weighted_games = model._apply_game_weights(games)
        
        # Week 17 game should be unchanged
        assert weighted_games[0]['margin'] == 20
        assert weighted_games[0]['week18_adjusted'] == False
        
        # Week 18 game should have reduced margin (14 * 0.3 = 4.2, rounded to 4)
        assert weighted_games[1]['margin'] == 4  # 14 * 0.3 = 4.2 -> int(4.2) = 4
        assert weighted_games[1]['week18_adjusted'] == True
    
    def test_calculate_season_stats(self, model):
        """Test calculation of season statistics."""
        games = [
            {'home_team_id': '1', 'away_team_id': '2', 'home_score': 30, 'away_score': 10, 'margin': 20, 'week': 1},
            {'home_team_id': '2', 'away_team_id': '3', 'home_score': 21, 'away_score': 17, 'margin': 4, 'week': 2},
            {'home_team_id': '1', 'away_team_id': '3', 'home_score': 14, 'away_score': 28, 'margin': -14, 'week': 3}
        ]
        
        stats = model._calculate_season_stats(games)
        
        # Team 1 stats: Won vs Team 2 (+20), Lost vs Team 3 (-14)
        team1_stats = stats['1']
        assert team1_stats['games_played'] == 2
        assert team1_stats['wins'] == 1
        assert team1_stats['losses'] == 1
        assert team1_stats['total_margin'] == 6  # 20 + (-14)
        assert team1_stats['avg_margin'] == 3.0  # 6 / 2
        assert team1_stats['points_for'] == 44  # 30 + 14
        assert team1_stats['points_against'] == 38  # 10 + 28
        assert team1_stats['win_pct'] == 0.5
        
        # Team 2 stats: Lost vs Team 1 (-20), Won vs Team 3 (+4)
        team2_stats = stats['2']
        assert team2_stats['games_played'] == 2
        assert team2_stats['wins'] == 1
        assert team2_stats['losses'] == 1
        assert team2_stats['total_margin'] == -16  # -20 + 4
        assert team2_stats['avg_margin'] == -8.0
        
        # Team 3 stats: Lost vs Team 2 (-4), Won vs Team 1 (+14)
        team3_stats = stats['3']
        assert team3_stats['games_played'] == 2
        assert team3_stats['total_margin'] == 10  # -4 + 14
        assert team3_stats['avg_margin'] == 5.0
    
    def test_calculate_rolling_stats(self, model):
        """Test calculation of rolling statistics."""
        # Create games across 7 weeks
        games = []
        for week in range(1, 8):
            games.append({
                'home_team_id': '1', 'away_team_id': '2', 
                'home_score': 20 + week, 'away_score': 15, 
                'margin': 5 + week, 'week': week
            })
        
        rolling_stats = model._calculate_rolling_stats(games, window=5)
        
        # Should only include weeks 3-7 (last 5 weeks)
        team1_rolling = rolling_stats['1']
        assert team1_rolling['games_played'] == 5
        
        # Margins for weeks 3-7: 8, 9, 10, 11, 12
        # Total = 50, Average = 10
        assert team1_rolling['avg_margin'] == 10.0
    
    def test_calculate_comprehensive_sos(self, model):
        """Test strength of schedule calculation."""
        # Create sample season stats
        season_stats = {
            '1': {'opponents': ['2', '3'], 'avg_margin': 5, 'win_pct': 0.5},
            '2': {'opponents': ['1', '3'], 'avg_margin': -2, 'win_pct': 0.3},
            '3': {'opponents': ['1', '2'], 'avg_margin': 8, 'win_pct': 0.8}
        }
        
        games = []  # Empty games list for this test
        sos_scores = model._calculate_comprehensive_sos(games, season_stats)
        
        # Team 1 opponents: Team 2 (avg_margin: -2, win_pct: 0.3) and Team 3 (avg_margin: 8, win_pct: 0.8)
        # Average opponent margin = (-2 + 8) / 2 = 3
        # Average opponent win_pct = (0.3 + 0.8) / 2 = 0.55
        # SOS = 3 * 0.6 + (0.55 - 0.5) * 20 * 0.4 = 1.8 + 0.4 = 2.2
        
        assert abs(sos_scores['1'] - 2.2) < 0.01
    
    def test_calculate_comprehensive_power_scores(self, model):
        """Test comprehensive power score calculation."""
        season_stats = {
            '1': {'avg_margin': 5, 'win_pct': 0.6, 'games_played': 10},
            '2': {'avg_margin': -3, 'win_pct': 0.4, 'games_played': 10}
        }
        
        rolling_stats = {
            '1': {'avg_margin': 7, 'games_played': 5},
            '2': {'avg_margin': -1, 'games_played': 5}
        }
        
        sos_scores = {'1': 2, '2': -1}
        
        power_scores = model._calculate_comprehensive_power_scores(
            season_stats, rolling_stats, sos_scores
        )
        
        # Team 1 calculation:
        # base_score = 5 * 0.5 + 7 * 0.25 + 2 * 0.2 = 2.5 + 1.75 + 0.4 = 4.65
        # recency_factor = (7 - 5) * 0.05 = 0.1
        # win_pct_bonus = (0.6 - 0.5) * 3 = 0.3
        # final = 4.65 + 0.1 + 0.3 = 5.05
        
        assert abs(power_scores['1'] - 5.05) < 0.01
        
        # Team 2 should have negative power score due to negative margins and low win %
        assert power_scores['2'] < 0
    
    def test_compute_full_integration(self, model, sample_scoreboard_data, sample_teams_info):
        """Test the complete compute method integration."""
        rankings, computation_data = model.compute(sample_scoreboard_data, sample_teams_info)
        
        # Should return rankings sorted by power score (highest first)
        assert len(rankings) > 0
        assert isinstance(rankings, list)
        
        # Each ranking should be a tuple of (team_id, team_name, score)
        for ranking in rankings:
            assert len(ranking) == 3
            assert isinstance(ranking[0], str)  # team_id
            assert isinstance(ranking[1], str)  # team_name
            assert isinstance(ranking[2], float)  # score
        
        # Rankings should be sorted by score (descending)
        scores = [r[2] for r in rankings]
        assert scores == sorted(scores, reverse=True)
        
        # Check computation data structure
        expected_keys = ['game_results', 'season_stats', 'rolling_stats', 'sos_scores', 'teams_map', 'power_scores']
        for key in expected_keys:
            assert key in computation_data
    
    def test_compute_with_empty_data(self, model):
        """Test compute method with empty data."""
        empty_scoreboard = {'events': []}
        empty_teams = []
        
        rankings, computation_data = model.compute(empty_scoreboard, empty_teams)
        
        assert rankings == []
        assert computation_data == {}
    
    def test_zero_games_played_handling(self, model):
        """Test handling of teams with zero games played."""
        season_stats = {
            '1': {'avg_margin': 5, 'win_pct': 0.6, 'games_played': 10},
            '2': {'avg_margin': 0, 'win_pct': 0, 'games_played': 0}  # No games
        }
        
        rolling_stats = {'1': {'avg_margin': 7, 'games_played': 5}}
        sos_scores = {'1': 2, '2': 0}
        
        power_scores = model._calculate_comprehensive_power_scores(
            season_stats, rolling_stats, sos_scores
        )
        
        assert power_scores['2'] == 0  # Team with no games should get 0 score
        assert power_scores['1'] > 0  # Team with games should get positive score


class TestPowerRankModelEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def model(self):
        return PowerRankModel()
    
    def test_negative_margins_handling(self, model):
        """Test proper handling of negative margins."""
        games = [
            {'home_team_id': '1', 'away_team_id': '2', 'home_score': 10, 'away_score': 30, 'margin': -20, 'week': 1},
            {'home_team_id': '1', 'away_team_id': '3', 'home_score': 7, 'away_score': 35, 'margin': -28, 'week': 2}
        ]
        
        stats = model._calculate_season_stats(games)
        
        # Team 1 should have negative average margin
        assert stats['1']['avg_margin'] == -24.0  # (-20 + -28) / 2
        assert stats['1']['wins'] == 0
        assert stats['1']['losses'] == 2
    
    def test_week18_weight_boundary_values(self, model):
        """Test Week 18 weighting with boundary margin values."""
        games = [
            {'home_team_id': '1', 'away_team_id': '2', 'margin': 0, 'week': 18, 'home_score': 14, 'away_score': 14},  # Tie
            {'home_team_id': '3', 'away_team_id': '4', 'margin': 1, 'week': 18, 'home_score': 15, 'away_score': 14},  # 1-point game
            {'home_team_id': '5', 'away_team_id': '6', 'margin': -1, 'week': 18, 'home_score': 13, 'away_score': 14}   # Negative margin
        ]
        
        weighted_games = model._apply_game_weights(games)
        
        # All should be reduced by week18_weight (0.3)
        assert weighted_games[0]['margin'] == 0  # 0 * 0.3 = 0
        assert weighted_games[1]['margin'] == 0  # int(1 * 0.3) = int(0.3) = 0
        assert weighted_games[2]['margin'] == 0  # int(-1 * 0.3) = int(-0.3) = 0
    
    def test_missing_opponent_in_sos_calculation(self, model):
        """Test SOS calculation when opponent data is missing."""
        season_stats = {
            '1': {'opponents': ['2', '999'], 'avg_margin': 5, 'win_pct': 0.5},  # Team 999 doesn't exist
            '2': {'opponents': ['1'], 'avg_margin': -2, 'win_pct': 0.3}
        }
        
        games = []
        sos_scores = model._calculate_comprehensive_sos(games, season_stats)
        
        # Should handle missing opponent gracefully
        assert '1' in sos_scores
        assert isinstance(sos_scores['1'], float)
    
    def test_rolling_stats_with_insufficient_data(self, model):
        """Test rolling stats when there are fewer games than the rolling window."""
        games = [
            {'home_team_id': '1', 'away_team_id': '2', 'home_score': 20, 'away_score': 15, 'margin': 5, 'week': 1},
            {'home_team_id': '1', 'away_team_id': '3', 'home_score': 24, 'away_score': 21, 'margin': 3, 'week': 2}
        ]
        
        rolling_stats = model._calculate_rolling_stats(games, window=5)
        
        # Should include both games even though window is 5
        assert rolling_stats['1']['games_played'] == 2
        assert rolling_stats['1']['avg_margin'] == 4.0  # (5 + 3) / 2