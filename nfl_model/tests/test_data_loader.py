"""
Unit tests for NFL Model Data Loader functionality.
"""
import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from io import StringIO

from nfl_model.data_loader import (
    load_power_rankings,
    load_schedule,
    PowerRankingsLoader,
    ScheduleLoader
)


class TestLoadPowerRankings:
    """Test cases for load_power_rankings function."""
    
    def test_load_power_rankings_success(self):
        """Test successful loading of power rankings CSV."""
        csv_data = """week,rank,team_id,team_name,power_score
1,1,11,Kansas City Chiefs,8.5
1,2,1,Buffalo Bills,6.2
1,3,15,Miami Dolphins,2.1
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            power_ratings = load_power_rankings('test_file.csv')
        
        expected = {
            'Kansas City Chiefs': 8.5,
            'Buffalo Bills': 6.2,
            'Miami Dolphins': 2.1
        }
        
        assert power_ratings == expected
    
    def test_load_power_rankings_file_not_found(self):
        """Test handling of missing power rankings file."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                load_power_rankings('missing_file.csv')
    
    def test_load_power_rankings_empty_file(self):
        """Test handling of empty power rankings file."""
        empty_csv = "week,rank,team_id,team_name,power_score\n"
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(empty_csv))):
            power_ratings = load_power_rankings('empty_file.csv')
        
        assert power_ratings == {}
    
    def test_load_power_rankings_missing_columns(self):
        """Test handling of CSV with missing required columns."""
        csv_missing_cols = """rank,week,team_id
1,1,11
2,1,1
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_missing_cols))):
            with pytest.raises(KeyError, match="Missing required columns"):
                load_power_rankings('bad_file.csv')
    
    def test_load_power_rankings_duplicate_teams(self):
        """Test handling of duplicate team entries (should use last occurrence)."""
        csv_duplicates = """week,rank,team_id,team_name,power_score
1,1,11,Kansas City Chiefs,8.5
1,2,11,Kansas City Chiefs,9.0
1,3,1,Buffalo Bills,6.2
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_duplicates))):
            power_ratings = load_power_rankings('duplicate_file.csv')
        
        # Should use the last occurrence (9.0)
        assert power_ratings['Kansas City Chiefs'] == 9.0
        assert power_ratings['Buffalo Bills'] == 6.2
    
    def test_load_power_rankings_invalid_power_scores(self):
        """Test handling of non-numeric power scores."""
        csv_invalid = """week,rank,team_id,team_name,power_score
1,1,11,Kansas City Chiefs,invalid
1,2,1,Buffalo Bills,6.2
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_invalid))):
            with pytest.raises((ValueError, TypeError)):
                load_power_rankings('invalid_file.csv')


class TestLoadSchedule:
    """Test cases for load_schedule function."""
    
    def test_load_schedule_success(self):
        """Test successful loading of schedule CSV."""
        csv_data = """week,home_team,away_team,game_date,game_time
5,Kansas City Chiefs,Las Vegas Raiders,2024-10-15,8:20 PM
5,Buffalo Bills,Miami Dolphins,2024-10-15,1:00 PM
5,Dallas Cowboys,New York Giants,2024-10-15,4:25 PM
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            matchups = load_schedule('test_schedule.csv', week=5)
        
        expected = [
            ('Kansas City Chiefs', 'Las Vegas Raiders', '2024-10-15'),
            ('Buffalo Bills', 'Miami Dolphins', '2024-10-15'),
            ('Dallas Cowboys', 'New York Giants', '2024-10-15')
        ]
        
        assert matchups == expected
    
    def test_load_schedule_filter_by_week(self):
        """Test filtering schedule by specific week."""
        csv_data = """week,home_team,away_team,game_date,game_time
4,Team A,Team B,2024-10-08,1:00 PM
5,Team C,Team D,2024-10-15,1:00 PM
6,Team E,Team F,2024-10-22,1:00 PM
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            matchups = load_schedule('test_schedule.csv', week=5)
        
        # Should only return week 5 games
        assert len(matchups) == 1
        assert matchups[0] == ('Team C', 'Team D', '2024-10-15')
    
    def test_load_schedule_no_games_for_week(self):
        """Test handling when no games exist for specified week."""
        csv_data = """week,home_team,away_team,game_date,game_time
4,Team A,Team B,2024-10-08,1:00 PM
6,Team E,Team F,2024-10-22,1:00 PM
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            matchups = load_schedule('test_schedule.csv', week=5)
        
        assert matchups == []
    
    def test_load_schedule_missing_columns(self):
        """Test handling of CSV with missing required columns."""
        csv_missing_cols = """home_team,away_team,game_date
Team A,Team B,2024-10-15
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_missing_cols))):
            with pytest.raises(KeyError):
                load_schedule('bad_schedule.csv', week=5)
    
    def test_load_schedule_missing_game_date(self):
        """Test handling of schedule with missing game_date column."""
        csv_no_date = """week,home_team,away_team,game_time
5,Team A,Team B,1:00 PM
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_no_date))):
            matchups = load_schedule('no_date_schedule.csv', week=5)
        
        # Should use empty string for missing date
        assert matchups == [('Team A', 'Team B', '')]
    
    def test_load_schedule_file_not_found(self):
        """Test handling of missing schedule file."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                load_schedule('missing_schedule.csv', week=5)


class TestPowerRankingsLoader:
    """Test cases for PowerRankingsLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a PowerRankingsLoader instance."""
        return PowerRankingsLoader()
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert isinstance(loader, PowerRankingsLoader)
    
    def test_load_from_file_success(self, loader):
        """Test successful loading through loader class."""
        csv_data = """week,rank,team_id,team_name,power_score
1,1,11,Kansas City Chiefs,8.5
1,2,1,Buffalo Bills,6.2
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            power_ratings = loader.load_from_file('test_file.csv')
        
        assert power_ratings['Kansas City Chiefs'] == 8.5
        assert power_ratings['Buffalo Bills'] == 6.2
    
    def test_validate_required_columns_success(self, loader):
        """Test validation of required columns."""
        valid_df = pd.DataFrame({
            'week': [1, 1],
            'rank': [1, 2],
            'team_id': ['11', '1'],
            'team_name': ['Chiefs', 'Bills'],
            'power_score': [8.5, 6.2]
        })
        
        # Should not raise exception
        loader._validate_required_columns(valid_df)
    
    def test_validate_required_columns_missing(self, loader):
        """Test validation fails with missing columns."""
        invalid_df = pd.DataFrame({
            'rank': [1, 2],
            'team_name': ['Chiefs', 'Bills'],
            'power_score': [8.5, 6.2]
            # Missing 'week' and 'team_id'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            loader._validate_required_columns(invalid_df)
    
    def test_convert_to_dict_success(self, loader):
        """Test conversion of DataFrame to dictionary."""
        df = pd.DataFrame({
            'team_name': ['Kansas City Chiefs', 'Buffalo Bills'],
            'power_score': [8.5, 6.2]
        })
        
        result = loader._convert_to_dict(df)
        
        expected = {
            'Kansas City Chiefs': 8.5,
            'Buffalo Bills': 6.2
        }
        assert result == expected
    
    def test_convert_to_dict_with_duplicates(self, loader):
        """Test conversion handles duplicate team names."""
        df = pd.DataFrame({
            'team_name': ['Chiefs', 'Chiefs', 'Bills'],
            'power_score': [8.0, 8.5, 6.2]
        })
        
        result = loader._convert_to_dict(df)
        
        # Should keep the last occurrence
        assert result['Chiefs'] == 8.5
        assert result['Bills'] == 6.2


class TestScheduleLoader:
    """Test cases for ScheduleLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a ScheduleLoader instance."""
        return ScheduleLoader()
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert isinstance(loader, ScheduleLoader)
    
    def test_load_from_file_success(self, loader):
        """Test successful loading through loader class."""
        csv_data = """week,home_team,away_team,game_date,game_time
5,Kansas City Chiefs,Las Vegas Raiders,2024-10-15,8:20 PM
5,Buffalo Bills,Miami Dolphins,2024-10-15,1:00 PM
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_data))):
            matchups = loader.load_from_file('test_schedule.csv', week=5)
        
        assert len(matchups) == 2
        assert matchups[0] == ('Kansas City Chiefs', 'Las Vegas Raiders', '2024-10-15')
    
    def test_filter_by_week_success(self, loader):
        """Test filtering DataFrame by week."""
        df = pd.DataFrame({
            'week': [4, 5, 5, 6],
            'home_team': ['A', 'B', 'C', 'D'],
            'away_team': ['E', 'F', 'G', 'H'],
            'game_date': ['2024-10-08', '2024-10-15', '2024-10-15', '2024-10-22']
        })
        
        result = loader._filter_by_week(df, week=5)
        
        assert len(result) == 2
        assert list(result['home_team']) == ['B', 'C']
    
    def test_convert_to_tuples_with_date(self, loader):
        """Test conversion to tuples with game_date column."""
        df = pd.DataFrame({
            'home_team': ['Chiefs', 'Bills'],
            'away_team': ['Raiders', 'Dolphins'],
            'game_date': ['2024-10-15', '2024-10-15']
        })
        
        result = loader._convert_to_tuples(df)
        
        expected = [
            ('Chiefs', 'Raiders', '2024-10-15'),
            ('Bills', 'Dolphins', '2024-10-15')
        ]
        assert result == expected
    
    def test_convert_to_tuples_without_date(self, loader):
        """Test conversion to tuples without game_date column."""
        df = pd.DataFrame({
            'home_team': ['Chiefs', 'Bills'],
            'away_team': ['Raiders', 'Dolphins']
        })
        
        result = loader._convert_to_tuples(df)
        
        expected = [
            ('Chiefs', 'Raiders', ''),
            ('Bills', 'Dolphins', '')
        ]
        assert result == expected


class TestDataLoaderEdgeCases:
    """Test edge cases and error conditions for data loaders."""
    
    def test_power_rankings_with_nan_values(self):
        """Test handling of NaN values in power rankings."""
        csv_with_nan = """week,rank,team_id,team_name,power_score
1,1,11,Kansas City Chiefs,8.5
1,2,1,Buffalo Bills,
1,3,15,Miami Dolphins,2.1
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_with_nan))):
            # Should handle NaN by raising appropriate error
            with pytest.raises(ValueError, match="Invalid power_score values"):
                load_power_rankings('nan_file.csv')
    
    def test_schedule_with_special_characters_in_team_names(self):
        """Test handling of special characters in team names."""
        csv_special_chars = """week,home_team,away_team,game_date
5,Team A & B,Team C/D,2024-10-15
5,Team E's,Team F-G,2024-10-15
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_special_chars))):
            matchups = load_schedule('special_chars.csv', week=5)
        
        assert matchups[0] == ('Team A & B', 'Team C/D', '2024-10-15')
        assert matchups[1] == ("Team E's", 'Team F-G', '2024-10-15')
    
    def test_power_rankings_with_extreme_values(self):
        """Test handling of extreme power rating values."""
        csv_extreme = """week,rank,team_id,team_name,power_score
1,1,11,Dominant Team,99.9
1,2,1,Terrible Team,-99.9
1,3,15,Zero Team,0.0
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_extreme))):
            power_ratings = load_power_rankings('extreme_values.csv')
        
        assert power_ratings['Dominant Team'] == 99.9
        assert power_ratings['Terrible Team'] == -99.9
        assert power_ratings['Zero Team'] == 0.0
    
    def test_schedule_with_multiple_weeks_same_teams(self):
        """Test schedule with same teams playing multiple weeks."""
        csv_multiple_weeks = """week,home_team,away_team,game_date
5,Team A,Team B,2024-10-15
17,Team A,Team B,2024-12-28
"""
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(csv_multiple_weeks))):
            week5_matchups = load_schedule('multiple_weeks.csv', week=5)
            week17_matchups = load_schedule('multiple_weeks.csv', week=17)
        
        assert len(week5_matchups) == 1
        assert len(week17_matchups) == 1
        assert week5_matchups[0][2] == '2024-10-15'
        assert week17_matchups[0][2] == '2024-12-28'


class TestDataLoaderIntegration:
    """Integration tests for data loader functionality."""
    
    def test_load_and_use_power_rankings_and_schedule(self):
        """Test complete workflow of loading both power rankings and schedule."""
        power_csv = """week,rank,team_id,team_name,power_score
5,1,11,Kansas City Chiefs,8.5
5,2,1,Buffalo Bills,6.2
5,3,12,Las Vegas Raiders,-4.0
5,4,15,Miami Dolphins,2.1
"""
        
        schedule_csv = """week,home_team,away_team,game_date,game_time
5,Kansas City Chiefs,Las Vegas Raiders,2024-10-15,8:20 PM
5,Buffalo Bills,Miami Dolphins,2024-10-15,1:00 PM
"""
        
        # Pre-create the DataFrames to avoid recursive mocking
        power_df = pd.read_csv(StringIO(power_csv))
        schedule_df = pd.read_csv(StringIO(schedule_csv))
        
        with patch('pandas.read_csv') as mock_read:
            def side_effect(filename, *args, **kwargs):
                if 'power' in str(filename).lower():
                    return power_df.copy()
                elif 'schedule' in str(filename).lower():
                    return schedule_df.copy()
                else:
                    # Fallback - return power first, then schedule
                    return power_df.copy() if mock_read.call_count <= 1 else schedule_df.copy()
            
            mock_read.side_effect = side_effect
            
            # Load both datasets
            power_ratings = load_power_rankings('power_rankings.csv')
            matchups = load_schedule('schedule.csv', week=5)
        
        # Verify data loaded correctly
        assert len(power_ratings) == 4
        assert len(matchups) == 2
        
        # Verify all teams in schedule have power ratings
        for home_team, away_team, _ in matchups:
            assert home_team in power_ratings
            assert away_team in power_ratings
        
        # Calculate hypothetical spreads to verify integration
        for home_team, away_team, game_date in matchups:
            home_power = power_ratings[home_team]
            away_power = power_ratings[away_team]
            neutral_spread = home_power - away_power
            projected_spread = neutral_spread + 2.0  # Home field advantage
            
            # Verify calculations make sense
            assert isinstance(projected_spread, (int, float))
            if home_team == "Kansas City Chiefs" and away_team == "Las Vegas Raiders":
                assert projected_spread == 14.5  # 8.5 - (-4.0) + 2.0
    
    def test_data_consistency_validation(self):
        """Test validation of data consistency between power rankings and schedule."""
        power_csv = """week,rank,team_id,team_name,power_score
5,1,11,Kansas City Chiefs,8.5
5,2,1,Buffalo Bills,6.2
"""
        
        schedule_csv = """week,home_team,away_team,game_date
5,Kansas City Chiefs,Unknown Team,2024-10-15
5,Buffalo Bills,Miami Dolphins,2024-10-15
"""
        
        # Pre-create the DataFrames to avoid recursive mocking
        power_df = pd.read_csv(StringIO(power_csv))
        schedule_df = pd.read_csv(StringIO(schedule_csv))
        
        with patch('pandas.read_csv') as mock_read:
            def side_effect(filename, *args, **kwargs):
                if 'power' in str(filename).lower():
                    return power_df.copy()
                elif 'schedule' in str(filename).lower():
                    return schedule_df.copy()
                else:
                    # Fallback - return power first, then schedule
                    return power_df.copy() if mock_read.call_count <= 1 else schedule_df.copy()
            
            mock_read.side_effect = side_effect
            
            power_ratings = load_power_rankings('power_rankings.csv')
            matchups = load_schedule('schedule.csv', week=5)
        
        # Check for teams in schedule that don't have power ratings
        missing_teams = []
        for home_team, away_team, _ in matchups:
            if home_team not in power_ratings:
                missing_teams.append(home_team)
            if away_team not in power_ratings:
                missing_teams.append(away_team)
        
        # Should find missing teams
        assert 'Unknown Team' in missing_teams
        assert 'Miami Dolphins' in missing_teams
        assert len(missing_teams) == 2