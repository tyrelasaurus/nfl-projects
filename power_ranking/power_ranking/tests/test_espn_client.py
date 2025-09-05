"""
Unit tests for ESPNClient API functionality.
"""
import pytest
import requests
import json
from unittest.mock import Mock, patch, mock_open
from requests.exceptions import RequestException, Timeout, HTTPError

from power_ranking.api.espn_client import ESPNClient


class TestESPNClient:
    """Test cases for ESPNClient class."""
    
    @pytest.fixture
    def client(self):
        """Create an ESPNClient instance."""
        return ESPNClient()
    
    @pytest.fixture
    def custom_client(self):
        """Create an ESPNClient with custom base URL."""
        return ESPNClient(base_url="https://custom.api.test.com/v2")
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"test": "data"}
        return mock_resp
    
    def test_client_initialization_default(self, client):
        """Test client initializes with correct default values."""
        assert client.base_url == "https://site.api.espn.com/apis/site/v2"
        assert isinstance(client.session, requests.Session)
        assert "Mozilla" in client.session.headers['User-Agent']
    
    def test_client_initialization_custom(self, custom_client):
        """Test client initializes with custom base URL."""
        assert custom_client.base_url == "https://custom.api.test.com/v2"
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    def test_make_request_success(self, mock_get, client, mock_response):
        """Test successful API request."""
        mock_get.return_value = mock_response
        
        result = client._make_request("test/endpoint")
        
        mock_get.assert_called_once_with(
            "https://site.api.espn.com/apis/site/v2/test/endpoint", 
            params=None, 
            timeout=30
        )
        assert result == {"test": "data"}
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    def test_make_request_with_params(self, mock_get, client, mock_response):
        """Test API request with parameters."""
        mock_get.return_value = mock_response
        params = {"week": 1, "season": 2024}
        
        client._make_request("scoreboard", params=params)
        
        mock_get.assert_called_once_with(
            "https://site.api.espn.com/apis/site/v2/scoreboard",
            params=params,
            timeout=30
        )
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    @patch('power_ranking.api.espn_client.time.sleep')
    def test_make_request_retry_on_failure(self, mock_sleep, mock_get, client):
        """Test request retry logic on failure."""
        # First two calls fail, third succeeds
        mock_get.side_effect = [
            RequestException("Network error"),
            Timeout("Timeout error"), 
            Mock(status_code=200, json=lambda: {"success": True})
        ]
        
        result = client._make_request("test", retries=3)
        
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries
        assert result == {"success": True}
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    def test_make_request_all_retries_fail(self, mock_get, client):
        """Test request failure after all retries exhausted."""
        mock_get.side_effect = RequestException("Persistent error")
        
        with pytest.raises(RequestException):
            client._make_request("test", retries=2)
        
        assert mock_get.call_count == 2
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    def test_make_request_http_error(self, mock_get, client):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            client._make_request("nonexistent")
    
    @patch.object(ESPNClient, '_make_request')
    def test_get_teams_success(self, mock_request, client):
        """Test successful teams retrieval."""
        mock_request.return_value = {
            'sports': [{
                'leagues': [{
                    'teams': [
                        {'team': {'id': '1', 'displayName': 'Buffalo Bills'}},
                        {'team': {'id': '2', 'displayName': 'Miami Dolphins'}}
                    ]
                }]
            }]
        }
        
        teams = client.get_teams()
        
        mock_request.assert_called_once_with("sports/football/nfl/teams")
        assert len(teams) == 2
        assert teams[0]['team']['displayName'] == 'Buffalo Bills'
    
    @patch.object(ESPNClient, '_make_request')
    def test_get_teams_empty_response(self, mock_request, client):
        """Test teams retrieval with empty response."""
        mock_request.return_value = {'sports': [{}]}
        
        teams = client.get_teams()
        
        assert teams == []
    
    @patch.object(ESPNClient, '_make_request')
    def test_get_teams_api_error(self, mock_request, client):
        """Test teams retrieval API error handling."""
        mock_request.side_effect = RequestException("API Error")
        
        with pytest.raises(RequestException):
            client.get_teams()
    
    @patch.object(ESPNClient, '_make_request')
    def test_get_scoreboard_with_params(self, mock_request, client):
        """Test scoreboard retrieval with parameters."""
        mock_request.return_value = {
            'events': [{'id': 'game1'}],
            'week': {'number': 5}
        }
        
        result = client.get_scoreboard(week=5, season=2024)
        
        expected_params = {'week': 5, 'seasontype': '2', 'year': '2024'}
        mock_request.assert_called_once_with("sports/football/nfl/scoreboard", params=expected_params)
        assert result['week']['number'] == 5
    
    @patch.object(ESPNClient, '_make_request')
    def test_get_scoreboard_no_params(self, mock_request, client):
        """Test scoreboard retrieval without parameters."""
        mock_request.return_value = {'events': []}
        
        client.get_scoreboard()
        
        mock_request.assert_called_once_with("sports/football/nfl/scoreboard", params={})
    
    @patch.object(ESPNClient, 'get_scoreboard')
    def test_get_current_week_success(self, mock_scoreboard, client):
        """Test current week retrieval."""
        mock_scoreboard.return_value = {'week': {'number': 8}}
        
        week = client.get_current_week()
        
        assert week == 8
    
    @patch.object(ESPNClient, 'get_scoreboard')
    def test_get_current_week_missing_data(self, mock_scoreboard, client):
        """Test current week with missing data."""
        mock_scoreboard.return_value = {}
        
        week = client.get_current_week()
        
        assert week == 1  # Default value
    
    @patch.object(ESPNClient, 'get_scoreboard')
    def test_get_current_week_api_error(self, mock_scoreboard, client):
        """Test current week with API error."""
        mock_scoreboard.side_effect = RequestException("API Error")
        
        week = client.get_current_week()
        
        assert week == 1  # Default value on error
    
    @patch.object(ESPNClient, 'get_scoreboard')
    def test_has_current_season_games_true(self, mock_scoreboard, client):
        """Test checking for current season games - games exist."""
        mock_scoreboard.return_value = {
            'events': [
                {'status': {'type': {'name': 'STATUS_FINAL'}}},
                {'status': {'type': {'name': 'STATUS_IN_PROGRESS'}}}
            ]
        }
        
        result = client.has_current_season_games()
        
        assert result is True
    
    @patch.object(ESPNClient, 'get_scoreboard') 
    def test_has_current_season_games_false(self, mock_scoreboard, client):
        """Test checking for current season games - no completed games."""
        mock_scoreboard.return_value = {
            'events': [
                {'status': {'type': {'name': 'STATUS_SCHEDULED'}}},
                {'status': {'type': {'name': 'STATUS_IN_PROGRESS'}}}
            ]
        }
        
        result = client.has_current_season_games()
        
        assert result is False
    
    @patch.object(ESPNClient, 'get_scoreboard')
    def test_has_current_season_games_api_error(self, mock_scoreboard, client):
        """Test checking for current season games with API error."""
        mock_scoreboard.side_effect = RequestException("API Error")
        
        result = client.has_current_season_games()
        
        assert result is False


class TestESPNClientHistoricalData:
    """Test historical data retrieval methods."""
    
    @pytest.fixture
    def client(self):
        return ESPNClient()
    
    @pytest.fixture
    def sample_game_event(self):
        """Sample completed game event."""
        return {
            'id': 'game123',
            'status': {'type': {'name': 'STATUS_FINAL'}},
            'season': {'type': 2},  # Regular season
            'week': {'number': 5},
            'competitions': [{
                'competitors': [
                    {'homeAway': 'home', 'team': {'id': '1'}, 'score': '28'},
                    {'homeAway': 'away', 'team': {'id': '2'}, 'score': '21'}
                ]
            }]
        }
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"games": [{"id": "test"}]}')
    def test_load_verified_complete_dataset_success(self, mock_file, mock_exists, client):
        """Test loading verified dataset successfully."""
        mock_exists.return_value = True
        # Mock 272 games
        games_data = {"games": [{"id": f"game{i}"} for i in range(272)]}
        mock_file.return_value.read.return_value = json.dumps(games_data)
        
        with patch('json.load', return_value=games_data):
            games = client._load_verified_complete_dataset()
        
        assert len(games) == 272
        mock_exists.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_verified_complete_dataset_file_not_found(self, mock_exists, client):
        """Test loading verified dataset when file doesn't exist."""
        mock_exists.return_value = False
        
        games = client._load_verified_complete_dataset()
        
        assert games == []
    
    @patch('os.path.exists')
    @patch('builtins.open', side_effect=IOError("File read error"))
    def test_load_verified_complete_dataset_io_error(self, mock_file, mock_exists, client):
        """Test loading verified dataset with file I/O error."""
        mock_exists.return_value = True
        
        games = client._load_verified_complete_dataset()
        
        assert games == []
    
    @patch.object(ESPNClient, '_make_request')
    def test_fetch_season_by_dates_2024_success(self, mock_request, client, sample_game_event):
        """Test fetching season by dates successfully."""
        # Mock successful API responses for each date
        mock_request.return_value = {'events': [sample_game_event]}
        
        games = client._fetch_season_by_dates_2024()
        
        # Should make requests for all 18 weeks
        assert mock_request.call_count == 18
        assert len(games) == 18  # One game per week
        
        # Check that week numbers are added correctly
        assert all(game.get('week_number') for game in games)
    
    @patch.object(ESPNClient, '_make_request')
    def test_fetch_season_by_dates_2024_partial_failure(self, mock_request, client, sample_game_event):
        """Test fetching season by dates with some API failures."""
        def side_effect(endpoint, params=None):
            if params and params.get('dates') == '20240915':  # Week 2 fails
                raise RequestException("API Error")
            return {'events': [sample_game_event]}
        
        mock_request.side_effect = side_effect
        
        games = client._fetch_season_by_dates_2024()
        
        # Should have 17 games (18 weeks - 1 failed)
        assert len(games) == 17
    
    @patch.object(ESPNClient, '_make_request')
    def test_fetch_season_by_year_2024_success(self, mock_request, client, sample_game_event):
        """Test fetching season by year successfully."""
        # Mock year-based API response with unique game IDs
        events = []
        for i in range(272):
            game = sample_game_event.copy()
            game['id'] = f'game{i}'  # Unique ID for each game
            game['week'] = {'number': (i % 18) + 1}  # Cycle through weeks 1-18
            events.append(game)
        
        mock_request.return_value = {'events': events}
        
        games = client._fetch_season_by_year_2024()
        
        mock_request.assert_called()
        assert len(games) == 272
        assert all(game.get('week_number') for game in games)
    
    @patch.object(ESPNClient, '_make_request')
    def test_fetch_season_by_year_2024_fallback(self, mock_request, client, sample_game_event):
        """Test year-based fetching with fallback to date range."""
        def side_effect(endpoint, params=None):
            if params and 'dates' in params and params['dates'] == '2024':
                raise RequestException("Primary API failed")
            elif params and 'dates' in params and params['dates'] == '20240901-20250107':
                return {'events': [sample_game_event]}
            return {'events': []}
        
        mock_request.side_effect = side_effect
        
        games = client._fetch_season_by_year_2024()
        
        assert len(games) == 1
        assert mock_request.call_count >= 2  # Primary attempt + fallback
    
    @patch.object(ESPNClient, '_make_request')
    def test_fetch_season_extended_dates_2024(self, mock_request, client, sample_game_event):
        """Test extended date approach with duplicate filtering."""
        # Return the same game for multiple dates to test deduplication
        mock_request.return_value = {'events': [sample_game_event]}
        
        games = client._fetch_season_extended_dates_2024()
        
        # Should make many requests (3 dates per week * 18 weeks = 54)
        assert mock_request.call_count > 50
        
        # But should only return unique games (deduplication by game ID)
        unique_ids = set(game.get('id') for game in games)
        assert len(unique_ids) == len(games)
    
    @patch('power_ranking.api.espn_client.requests.Session.get')
    def test_fetch_season_core_api_2024(self, mock_get, client):
        """Test ESPN Core API approach."""
        # Mock Core API response
        core_response = Mock()
        core_response.status_code = 200
        core_response.json.return_value = {
            'items': [
                {'$ref': 'https://api.espn.com/event/1'},
                {'$ref': 'https://api.espn.com/event/2'}
            ]
        }
        
        # Mock individual event responses
        event_response = Mock()
        event_response.json.return_value = {
            'status': {'type': {'name': 'STATUS_FINAL'}},
            'week': {'number': 1}
        }
        
        mock_get.side_effect = [core_response, event_response, event_response]
        
        games = client._fetch_season_core_api_2024()
        
        assert len(games) == 2
        assert all(game.get('week_number') for game in games)
    
    @patch.object(ESPNClient, '_load_verified_complete_dataset')
    @patch.object(ESPNClient, '_fetch_season_by_dates_2024')
    def test_get_last_season_final_rankings_verified_dataset(self, mock_dates, mock_verified, client):
        """Test last season rankings using verified dataset."""
        # Mock verified dataset with exactly 272 games
        mock_verified.return_value = [{'id': f'game{i}'} for i in range(272)]
        
        result = client.get_last_season_final_rankings()
        
        # Should use verified dataset and not call other methods
        mock_verified.assert_called_once()
        mock_dates.assert_not_called()
        
        assert result['total_games'] == 272
        assert result['season']['year'] == 2024
        assert len(result['events']) == 272
    
    @patch.object(ESPNClient, '_load_verified_complete_dataset')
    @patch.object(ESPNClient, '_fetch_season_by_dates_2024')
    @patch.object(ESPNClient, '_fetch_season_extended_dates_2024') 
    @patch.object(ESPNClient, '_fetch_season_core_api_2024')
    @patch.object(ESPNClient, '_fetch_season_by_year_2024')
    def test_get_last_season_rankings_fallback_methods(self, mock_year, mock_core, 
                                                      mock_extended, mock_dates, 
                                                      mock_verified, client):
        """Test fallback method sequence when verified dataset is unavailable."""
        # Mock verified dataset as unavailable
        mock_verified.return_value = []
        
        # Mock each method returning insufficient data until core API provides enough
        mock_dates.return_value = [{'id': f'game{i}'} for i in range(250)]  # < 270
        mock_extended.return_value = [{'id': f'game{i}'} for i in range(260)]  # < 270  
        mock_core.return_value = [{'id': f'game{i}'} for i in range(265)]  # >= 260, stops here
        mock_year.return_value = [{'id': f'game{i}'} for i in range(275)]  # Won't be called
        
        result = client.get_last_season_final_rankings()
        
        # Should try methods in sequence until core API provides enough data
        mock_verified.assert_called_once()
        mock_dates.assert_called_once()
        mock_extended.assert_called_once()  
        mock_core.assert_called_once()
        # Year method should NOT be called since core API provided 265 games (>= 260)
        mock_year.assert_not_called()
        
        # Should use the core API result (265 games)
        assert len(result['events']) == 265
    
    @patch.object(ESPNClient, '_load_verified_complete_dataset')
    @patch.object(ESPNClient, '_fetch_season_by_dates_2024')
    @patch.object(ESPNClient, '_fetch_season_extended_dates_2024') 
    @patch.object(ESPNClient, '_fetch_season_core_api_2024')
    @patch.object(ESPNClient, '_fetch_season_by_year_2024')
    @patch.object(ESPNClient, '_get_sample_season_data')
    def test_get_last_season_rankings_all_methods_fail(self, mock_sample, mock_year, mock_core,
                                                      mock_extended, mock_dates, 
                                                      mock_verified, client):
        """Test fallback to sample data when all methods fail."""
        # Mock all methods returning insufficient data
        mock_verified.return_value = []
        mock_dates.return_value = []
        mock_extended.return_value = []
        mock_core.return_value = []
        mock_year.return_value = []
        mock_sample.return_value = {'events': [], 'sample': True}
        
        result = client.get_last_season_final_rankings()
        
        # Should try all methods before falling back to sample data
        mock_verified.assert_called_once()
        mock_dates.assert_called_once()
        mock_extended.assert_called_once()
        mock_core.assert_called_once()
        mock_year.assert_called_once()
        mock_sample.assert_called_once()
        assert 'sample' in result
    
    def test_get_sample_season_data(self, client):
        """Test sample season data generation."""
        result = client._get_sample_season_data()
        
        assert 'events' in result
        assert 'week' in result
        assert 'season' in result
        assert result['week']['number'] == 18
        assert result['season']['year'] == 2024
        assert len(result['events']) > 0
        
        # Check that events have proper structure
        for event in result['events']:
            assert event['status']['type']['name'] == 'STATUS_FINAL'
            assert 'competitions' in event
            assert len(event['competitions'][0]['competitors']) == 2