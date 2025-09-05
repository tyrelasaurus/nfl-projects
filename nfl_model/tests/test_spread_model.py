"""
Unit tests for NFL Spread Model functionality.
"""
import pytest
from dataclasses import asdict

from nfl_model.spread_model import SpreadCalculator, MatchupResult


class TestMatchupResult:
    """Test cases for MatchupResult dataclass."""
    
    def test_matchup_result_creation(self):
        """Test MatchupResult can be created with all fields."""
        result = MatchupResult(
            week=5,
            home_team="Kansas City Chiefs",
            away_team="Las Vegas Raiders", 
            home_power=8.5,
            away_power=-2.0,
            neutral_diff=10.5,
            home_field_adj=2.0,
            projected_spread=12.5,
            game_date="2024-10-15"
        )
        
        assert result.week == 5
        assert result.home_team == "Kansas City Chiefs"
        assert result.away_team == "Las Vegas Raiders"
        assert result.home_power == 8.5
        assert result.away_power == -2.0
        assert result.neutral_diff == 10.5
        assert result.home_field_adj == 2.0
        assert result.projected_spread == 12.5
        assert result.game_date == "2024-10-15"
    
    def test_matchup_result_as_dict(self):
        """Test MatchupResult can be converted to dictionary."""
        result = MatchupResult(
            week=1, home_team="Team A", away_team="Team B",
            home_power=5.0, away_power=3.0, neutral_diff=2.0,
            home_field_adj=2.0, projected_spread=4.0, game_date=""
        )
        
        result_dict = asdict(result)
        expected_keys = {
            'week', 'home_team', 'away_team', 'home_power', 'away_power',
            'neutral_diff', 'home_field_adj', 'projected_spread', 'game_date'
        }
        
        assert set(result_dict.keys()) == expected_keys


class TestSpreadCalculator:
    """Test cases for SpreadCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a SpreadCalculator with default home field advantage."""
        return SpreadCalculator()
    
    @pytest.fixture
    def custom_calculator(self):
        """Create a SpreadCalculator with custom home field advantage."""
        return SpreadCalculator(home_field_advantage=3.0)
    
    def test_calculator_initialization_default(self, calculator):
        """Test calculator initializes with correct default home field advantage."""
        assert calculator.home_field_advantage == 2.0
    
    def test_calculator_initialization_custom(self, custom_calculator):
        """Test calculator initializes with custom home field advantage."""
        assert custom_calculator.home_field_advantage == 3.0
    
    def test_calculate_neutral_spread_positive(self, calculator):
        """Test neutral spread calculation when team A is better."""
        # Chiefs (+8) vs Raiders (-4) = +12 neutral spread
        spread = calculator.calculate_neutral_spread(8.0, -4.0)
        assert spread == 12.0
    
    def test_calculate_neutral_spread_negative(self, calculator):
        """Test neutral spread calculation when team B is better.""" 
        # Raiders (-4) vs Chiefs (+8) = -12 neutral spread
        spread = calculator.calculate_neutral_spread(-4.0, 8.0)
        assert spread == -12.0
    
    def test_calculate_neutral_spread_equal(self, calculator):
        """Test neutral spread calculation when teams are equal."""
        spread = calculator.calculate_neutral_spread(5.0, 5.0)
        assert spread == 0.0
    
    def test_calculate_neutral_spread_negative_ratings(self, calculator):
        """Test neutral spread with both negative ratings."""
        # Team A (-2) vs Team B (-5) = +3 (A is less bad)
        spread = calculator.calculate_neutral_spread(-2.0, -5.0)
        assert spread == 3.0
    
    def test_calculate_matchup_spread_home_favored(self, calculator):
        """Test matchup spread calculation where home team is favored."""
        result = calculator.calculate_matchup_spread(
            home_team="Kansas City Chiefs",
            away_team="Las Vegas Raiders",
            home_power=8.0,
            away_power=-4.0,
            week=5,
            game_date="2024-10-15"
        )
        
        assert result.week == 5
        assert result.home_team == "Kansas City Chiefs"
        assert result.away_team == "Las Vegas Raiders"
        assert result.home_power == 8.0
        assert result.away_power == -4.0
        assert result.neutral_diff == 12.0  # 8 - (-4)
        assert result.home_field_adj == 2.0
        assert result.projected_spread == 14.0  # 12 + 2
        assert result.game_date == "2024-10-15"
    
    def test_calculate_matchup_spread_away_favored(self, calculator):
        """Test matchup spread calculation where away team is favored."""
        result = calculator.calculate_matchup_spread(
            home_team="Las Vegas Raiders",
            away_team="Kansas City Chiefs", 
            home_power=-4.0,
            away_power=8.0,
            week=3,
            game_date="2024-09-20"
        )
        
        assert result.neutral_diff == -12.0  # -4 - 8
        assert result.projected_spread == -10.0  # -12 + 2
    
    def test_calculate_matchup_spread_pick_em(self, calculator):
        """Test matchup spread when teams are very close after home field."""
        result = calculator.calculate_matchup_spread(
            home_team="Team A",
            away_team="Team B",
            home_power=1.0,
            away_power=3.0,  # Away team 2 points better
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == -2.0  # 1 - 3
        assert result.projected_spread == 0.0  # -2 + 2 (home field cancels out)
    
    def test_calculate_matchup_spread_custom_home_field(self, custom_calculator):
        """Test matchup spread with custom home field advantage."""
        result = custom_calculator.calculate_matchup_spread(
            home_team="Home Team",
            away_team="Away Team",
            home_power=5.0,
            away_power=5.0,  # Equal teams
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == 0.0
        assert result.home_field_adj == 3.0
        assert result.projected_spread == 3.0  # 0 + 3
    
    def test_calculate_week_spreads_success(self, calculator):
        """Test calculating spreads for multiple games in a week."""
        matchups = [
            ("Kansas City Chiefs", "Las Vegas Raiders", "2024-10-15"),
            ("Buffalo Bills", "Miami Dolphins", "2024-10-15"), 
            ("Dallas Cowboys", "New York Giants", "2024-10-15")
        ]
        
        power_ratings = {
            "Kansas City Chiefs": 8.0,
            "Las Vegas Raiders": -4.0,
            "Buffalo Bills": 6.5,
            "Miami Dolphins": 2.0,
            "Dallas Cowboys": 3.5,
            "New York Giants": -1.0
        }
        
        results = calculator.calculate_week_spreads(matchups, power_ratings, week=5)
        
        assert len(results) == 3
        
        # Check first game: Chiefs vs Raiders
        chiefs_game = results[0]
        assert chiefs_game.home_team == "Kansas City Chiefs"
        assert chiefs_game.away_team == "Las Vegas Raiders"
        assert chiefs_game.projected_spread == 14.0  # (8 - (-4)) + 2
        
        # Check second game: Bills vs Dolphins  
        bills_game = results[1]
        assert bills_game.home_team == "Buffalo Bills"
        assert bills_game.projected_spread == 6.5  # (6.5 - 2) + 2
        
        # Check third game: Cowboys vs Giants
        cowboys_game = results[2]
        assert cowboys_game.projected_spread == 6.5  # (3.5 - (-1)) + 2
    
    def test_calculate_week_spreads_missing_team(self, calculator):
        """Test error handling when team not found in power ratings."""
        matchups = [
            ("Known Team", "Unknown Team", "2024-10-15")
        ]
        
        power_ratings = {
            "Known Team": 5.0
            # "Unknown Team" is missing
        }
        
        with pytest.raises(KeyError, match="Team not found in power ratings"):
            calculator.calculate_week_spreads(matchups, power_ratings, week=1)
    
    def test_calculate_week_spreads_empty_matchups(self, calculator):
        """Test calculating spreads with empty matchups list."""
        results = calculator.calculate_week_spreads([], {}, week=1)
        assert results == []
    
    def test_format_spread_as_betting_line_home_favored(self, calculator):
        """Test formatting spread as betting line when home team favored."""
        line = calculator.format_spread_as_betting_line(7.5, "Kansas City Chiefs")
        assert line == "Kansas City Chiefs -7.5"
    
    def test_format_spread_as_betting_line_away_favored(self, calculator):
        """Test formatting spread as betting line when away team favored."""
        line = calculator.format_spread_as_betting_line(-3.5, "Las Vegas Raiders")
        assert line == "Las Vegas Raiders +3.5"
    
    def test_format_spread_as_betting_line_pick_em(self, calculator):
        """Test formatting spread as betting line for pick 'em game."""
        line = calculator.format_spread_as_betting_line(0.0, "Dallas Cowboys")
        assert line == "Dallas Cowboys PK"
    
    def test_format_spread_as_betting_line_whole_number(self, calculator):
        """Test formatting spread with whole number."""
        line = calculator.format_spread_as_betting_line(7.0, "Buffalo Bills")
        assert line == "Buffalo Bills -7.0"
    
    def test_format_spread_as_betting_line_small_favorite(self, calculator):
        """Test formatting spread with small favorite."""
        line = calculator.format_spread_as_betting_line(1.0, "Team A")
        assert line == "Team A -1.0"


class TestSpreadCalculatorEdgeCases:
    """Test edge cases and boundary conditions for SpreadCalculator."""
    
    @pytest.fixture
    def calculator(self):
        return SpreadCalculator()
    
    def test_extreme_power_rating_differences(self, calculator):
        """Test with extreme power rating differences."""
        # Very good team vs very bad team
        result = calculator.calculate_matchup_spread(
            home_team="Elite Team",
            away_team="Poor Team", 
            home_power=15.0,
            away_power=-12.0,
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == 27.0
        assert result.projected_spread == 29.0  # 27 + 2
    
    def test_zero_power_ratings(self, calculator):
        """Test with zero power ratings (average teams)."""
        result = calculator.calculate_matchup_spread(
            home_team="Average Home",
            away_team="Average Away",
            home_power=0.0,
            away_power=0.0,
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == 0.0
        assert result.projected_spread == 2.0  # 0 + 2 (home field only)
    
    def test_negative_home_field_advantage(self):
        """Test with negative home field advantage (unusual but possible)."""
        calculator = SpreadCalculator(home_field_advantage=-1.0)
        
        result = calculator.calculate_matchup_spread(
            home_team="Home Team",
            away_team="Away Team",
            home_power=5.0,
            away_power=3.0,
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == 2.0
        assert result.home_field_adj == -1.0
        assert result.projected_spread == 1.0  # 2 + (-1)
    
    def test_large_home_field_advantage(self):
        """Test with unusually large home field advantage."""
        calculator = SpreadCalculator(home_field_advantage=5.0)
        
        result = calculator.calculate_matchup_spread(
            home_team="Strong Home",
            away_team="Visitor",
            home_power=-2.0,
            away_power=1.0,  # Away team better on neutral field
            week=1,
            game_date=""
        )
        
        assert result.neutral_diff == -3.0  # Away team 3 points better
        assert result.projected_spread == 2.0  # -3 + 5 (large home field)
    
    def test_precision_with_fractional_ratings(self, calculator):
        """Test precision handling with fractional power ratings."""
        result = calculator.calculate_matchup_spread(
            home_team="Team A",
            away_team="Team B",
            home_power=4.33,
            away_power=-2.67,
            week=1, 
            game_date=""
        )
        
        assert result.neutral_diff == 7.0  # 4.33 - (-2.67) = 7.0
        assert result.projected_spread == 9.0  # 7.0 + 2.0
    
    def test_matchup_result_with_empty_strings(self, calculator):
        """Test matchup calculation with empty string inputs."""
        result = calculator.calculate_matchup_spread(
            home_team="",
            away_team="",
            home_power=0.0,
            away_power=0.0,
            week=1,
            game_date=""
        )
        
        assert result.home_team == ""
        assert result.away_team == ""
        assert result.game_date == ""
        assert result.projected_spread == 2.0
    
    def test_week_spreads_with_identical_power_ratings(self, calculator):
        """Test calculating spreads when all teams have identical ratings."""
        matchups = [
            ("Team A", "Team B", "2024-10-15"),
            ("Team C", "Team D", "2024-10-15")
        ]
        
        power_ratings = {
            "Team A": 3.0,
            "Team B": 3.0,
            "Team C": 3.0, 
            "Team D": 3.0
        }
        
        results = calculator.calculate_week_spreads(matchups, power_ratings, week=5)
        
        # All games should be home field advantage only
        for result in results:
            assert result.neutral_diff == 0.0
            assert result.projected_spread == 2.0
    
    def test_betting_line_formatting_edge_cases(self, calculator):
        """Test betting line formatting with edge case values."""
        # Very large spread
        line = calculator.format_spread_as_betting_line(25.5, "Dominant Team")
        assert line == "Dominant Team -25.5"
        
        # Very small spread
        line = calculator.format_spread_as_betting_line(0.5, "Slight Favorite")
        assert line == "Slight Favorite -0.5"
        
        # Very small underdog
        line = calculator.format_spread_as_betting_line(-0.5, "Slight Underdog")
        assert line == "Slight Underdog +0.5"


class TestSpreadCalculatorIntegration:
    """Integration tests combining multiple SpreadCalculator methods."""
    
    @pytest.fixture
    def calculator(self):
        return SpreadCalculator(home_field_advantage=2.5)
    
    def test_full_week_calculation_and_formatting(self, calculator):
        """Test complete workflow: calculate spreads and format as betting lines."""
        matchups = [
            ("Kansas City Chiefs", "Denver Broncos", "2024-12-01"),
            ("Buffalo Bills", "New England Patriots", "2024-12-01"),
            ("Miami Dolphins", "New York Jets", "2024-12-01")
        ]
        
        power_ratings = {
            "Kansas City Chiefs": 8.5,
            "Denver Broncos": -1.5,
            "Buffalo Bills": 6.0,
            "New England Patriots": -3.0,
            "Miami Dolphins": 2.5,
            "New York Jets": 1.0
        }
        
        results = calculator.calculate_week_spreads(matchups, power_ratings, week=13)
        
        # Generate betting lines for all games
        betting_lines = []
        for result in results:
            line = calculator.format_spread_as_betting_line(
                result.projected_spread, result.home_team
            )
            betting_lines.append({
                'matchup': f"{result.away_team} @ {result.home_team}",
                'line': line,
                'spread': result.projected_spread
            })
        
        # Verify results
        assert len(betting_lines) == 3
        
        # Chiefs should be big favorites at home
        chiefs_line = next(bl for bl in betting_lines if "Chiefs" in bl['matchup'])
        assert "Kansas City Chiefs -" in chiefs_line['line']
        assert chiefs_line['spread'] > 10  # Should be double-digit favorite
        
        # Bills should be solid favorites
        bills_line = next(bl for bl in betting_lines if "Bills" in bl['matchup'])  
        assert "Buffalo Bills -" in bills_line['line']
        assert 5 < bills_line['spread'] < 15
        
        # Dolphins should be slight favorites
        dolphins_line = next(bl for bl in betting_lines if "Dolphins" in bl['matchup'])
        assert "Miami Dolphins -" in dolphins_line['line'] or "Miami Dolphins PK" in dolphins_line['line']
    
    def test_power_rating_validation_through_spreads(self, calculator):
        """Test that power ratings produce expected spread relationships."""
        # Set up teams with known relationships
        power_ratings = {
            "Elite Team": 10.0,      # Best team
            "Good Team": 5.0,        # Above average
            "Average Team": 0.0,     # League average  
            "Poor Team": -5.0,       # Below average
            "Terrible Team": -10.0   # Worst team
        }
        
        # Test various matchups
        test_matchups = [
            ("Elite Team", "Terrible Team", "max_spread"),
            ("Average Team", "Average Team", "home_field_only"),
            ("Good Team", "Poor Team", "moderate_spread"),
            ("Poor Team", "Elite Team", "large_underdog")
        ]
        
        for home, away, expected_type in test_matchups:
            matchups = [(home, away, "2024-01-01")]
            results = calculator.calculate_week_spreads(matchups, power_ratings, week=1)
            spread = results[0].projected_spread
            
            if expected_type == "max_spread":
                # Elite vs Terrible: (10 - (-10)) + 2.5 = 22.5
                assert spread == 22.5
            elif expected_type == "home_field_only":
                # Average vs Average: (0 - 0) + 2.5 = 2.5 
                assert spread == 2.5
            elif expected_type == "moderate_spread":
                # Good vs Poor: (5 - (-5)) + 2.5 = 12.5
                assert spread == 12.5
            elif expected_type == "large_underdog":
                # Poor vs Elite: (-5 - 10) + 2.5 = -12.5
                assert spread == -12.5