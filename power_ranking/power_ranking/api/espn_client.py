import requests
import time
import logging
import json
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ESPNClient:
    def __init__(self, base_url: str = "https://site.api.espn.com/apis/site/v2"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    # Compatibility wrappers to match documentation/examples
    def get_teams_data(self):
        """Compatibility alias for get_teams().

        Returns:
            List[Dict]: Team objects from ESPN API
        """
        return self.get_teams()

    def get_scoreboard_data(self, season: int, week: int):
        """Compatibility alias for get_scoreboard().

        Args:
            season (int): Season year (e.g., 2024)
            week (int): Week number

        Returns:
            Dict: Scoreboard payload from ESPN API
        """
        return self.get_scoreboard(week=week, season=season)
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def get_teams(self) -> List[Dict]:
        try:
            data = self._make_request("sports/football/nfl/teams")
            return data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
        except Exception as e:
            logger.error(f"Failed to fetch teams: {e}")
            raise
    
    def get_scoreboard(self, week: Optional[int] = None, season: Optional[int] = None) -> Dict:
        params = {}
        if week:
            params['week'] = week
        if season:
            params['seasontype'] = '2'  # Regular season
            params['year'] = str(season)
            
        try:
            data = self._make_request("sports/football/nfl/scoreboard", params=params)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch scoreboard for week {week}, season {season}: {e}")
            raise
    
    def get_current_week(self) -> int:
        try:
            data = self.get_scoreboard()
            return data.get('week', {}).get('number', 1)
        except Exception:
            logger.warning("Could not determine current week, defaulting to 1")
            return 1
    
    def get_last_season_final_rankings(self) -> Dict:
        """Get comprehensive data from entire 2024 season using verified complete dataset"""
        try:
            logger.info("Fetching complete 2024 season data for comprehensive rankings...")
            
            # Method 1: Try to use verified complete dataset first
            complete_games = self._load_verified_complete_dataset()
            if complete_games and len(complete_games) == 272:
                logger.info(f"Using verified complete dataset with {len(complete_games)} games")
                return {
                    'events': complete_games,
                    'week': {'number': 18},
                    'season': {'year': 2024},
                    'total_games': len(complete_games)
                }
            
            # Method 2: Try comprehensive date-based approach (most accurate)
            all_games = self._fetch_season_by_dates_2024()
            
            # Method 3: Try extended date approach if basic dates insufficient
            if len(all_games) < 270:  # Expected ~272 games for full season
                logger.info("Basic date approach insufficient, trying extended date range...")
                extended_games = self._fetch_season_extended_dates_2024()
                if len(extended_games) > len(all_games):
                    all_games = extended_games
            
            # Method 4: Fallback to ESPN Core API
            if len(all_games) < 270:
                logger.info("Extended dates insufficient, trying ESPN Core API...")
                core_games = self._fetch_season_core_api_2024()
                if core_games and len(core_games) > len(all_games):
                    all_games = core_games
            
            # Method 5: Final fallback to year-based approach (known to have data quality issues)
            if len(all_games) < 260:
                logger.info("All methods insufficient, trying year-based approach as final fallback...")
                year_games = self._fetch_season_by_year_2024()
                if len(year_games) > len(all_games):
                    all_games = year_games
            
            if not all_games:
                logger.warning("All methods failed, using sample data")
                return self._get_sample_season_data()
            
            logger.info(f"Successfully fetched {len(all_games)} games from 2024 season")
            
            # Return consolidated data structure
            return {
                'events': all_games,
                'week': {'number': 18},
                'season': {'year': 2024},
                'total_games': len(all_games)
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch 2024 season data: {e}")
            return self._get_sample_season_data()
    
    def _fetch_season_by_year_2024(self) -> List[Dict]:
        """Fetch 2024 season using year-based scoreboard approach (most comprehensive)"""
        logger.info("Using year-based scoreboard approach for 2024 season...")
        
        try:
            # Try the comprehensive year-based endpoint
            data = self._make_request("sports/football/nfl/scoreboard", {
                'dates': '2024',
                'seasontype': '2',
                'limit': '1000'
            })
            events = data.get('events', [])
            
            completed_games = [
                event for event in events 
                if (event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL' and
                    event.get('season', {}).get('type', 2) == 2)  # Regular season only
            ]
            
            # Add week numbers to games and filter to regular season weeks (1-18)
            filtered_games = []
            game_ids_seen = set()  # Track unique game IDs to prevent duplicates
            
            for game in completed_games:
                game_id = game.get('id')
                if game_id in game_ids_seen:
                    continue  # Skip duplicate games
                
                week_info = game.get('week', {})
                if week_info:
                    week_num = week_info.get('number', 1)
                    if 1 <= week_num <= 18:  # Regular season weeks only
                        game['week_number'] = week_num
                        filtered_games.append(game)
                        if game_id:
                            game_ids_seen.add(game_id)
                else:
                    # Fallback - assume week 1 for regular season
                    game['week_number'] = 1
                    filtered_games.append(game)
                    if game_id:
                        game_ids_seen.add(game_id)
            
            completed_games = filtered_games
            
            logger.info(f"Year-based approach: {len(completed_games)} games")
            return completed_games
            
        except Exception as e:
            logger.warning(f"Year-based approach failed: {e}")
            
            # Try alternative year-based endpoints
            try:
                # Try date range approach for full 2024 season
                data = self._make_request("sports/football/nfl/scoreboard", {
                    'dates': '20240901-20250107',
                    'limit': '1000'
                })
                events = data.get('events', [])
                
                completed_games = [
                    event for event in events 
                    if (event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL' and
                        event.get('season', {}).get('type', 2) == 2)  # Regular season only
                ]
                
                # Add week numbers and filter to regular season weeks
                filtered_games = []
                game_ids_seen = set()  # Track unique game IDs to prevent duplicates
                
                for game in completed_games:
                    game_id = game.get('id')
                    if game_id in game_ids_seen:
                        continue  # Skip duplicate games
                        
                    week_info = game.get('week', {})
                    if week_info:
                        week_num = week_info.get('number', 1)
                        if 1 <= week_num <= 18:  # Regular season weeks only
                            game['week_number'] = week_num
                            filtered_games.append(game)
                            if game_id:
                                game_ids_seen.add(game_id)
                    else:
                        game['week_number'] = 1
                        filtered_games.append(game)
                        if game_id:
                            game_ids_seen.add(game_id)
                
                completed_games = filtered_games
                
                logger.info(f"Date range approach: {len(completed_games)} games")
                return completed_games
                
            except Exception as e2:
                logger.warning(f"Date range approach also failed: {e2}")
                return []
    
    def _fetch_season_by_dates_2024(self) -> List[Dict]:
        """Fetch 2024 season using comprehensive date-based approach"""
        logger.info("Using comprehensive date-based approach for 2024 season...")
        
        # Complete 2024 NFL season dates (Sundays for each week)
        season_dates = [
            ('20240908', 1),   # Week 1
            ('20240915', 2),   # Week 2  
            ('20240922', 3),   # Week 3
            ('20240929', 4),   # Week 4
            ('20241006', 5),   # Week 5
            ('20241013', 6),   # Week 6
            ('20241020', 7),   # Week 7
            ('20241027', 8),   # Week 8
            ('20241103', 9),   # Week 9
            ('20241110', 10),  # Week 10
            ('20241117', 11),  # Week 11
            ('20241124', 12),  # Week 12 (Thanksgiving)
            ('20241201', 13),  # Week 13
            ('20241208', 14),  # Week 14
            ('20241215', 15),  # Week 15
            ('20241222', 16),  # Week 16
            ('20241229', 17),  # Week 17
            ('20250105', 18),  # Week 18
        ]
        
        all_games = []
        successful_weeks = 0
        
        for date_str, week_num in season_dates:
            try:
                logger.info(f"Fetching week {week_num} ({date_str})...")
                data = self._make_request("sports/football/nfl/scoreboard", {'dates': date_str})
                events = data.get('events', [])
                
                completed_games = [
                    event for event in events 
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'
                ]
                
                if completed_games:
                    # Add week number to each game
                    for game in completed_games:
                        game['week_number'] = week_num
                    
                    all_games.extend(completed_games)
                    successful_weeks += 1
                    logger.info(f"Week {week_num}: Found {len(completed_games)} games")
                else:
                    logger.warning(f"Week {week_num}: No completed games found")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch week {week_num}: {e}")
                continue
        
        logger.info(f"Date-based approach: {len(all_games)} games from {successful_weeks} weeks")
        return all_games
    
    def _fetch_season_core_api_2024(self) -> List[Dict]:
        """Fetch 2024 season using ESPN Core API"""
        try:
            logger.info("Trying ESPN Core API for 2024 season...")
            
            # ESPN Core API endpoint for full season events
            core_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/events"
            response = self.session.get(f"{core_url}?limit=1000", timeout=30)
            response.raise_for_status()
            core_data = response.json()
            
            # Process core API data (different structure)
            events = core_data.get('items', [])
            
            all_games = []
            for event_ref in events[:272]:  # Limit to regular season games
                try:
                    # Get individual event data
                    event_response = self.session.get(event_ref.get('$ref'), timeout=30)
                    event_data = event_response.json()
                    
                    # Convert to scoreboard format
                    if event_data.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        # Add week number
                        week_num = event_data.get('week', {}).get('number', 1)
                        event_data['week_number'] = week_num
                        all_games.append(event_data)
                        
                except Exception as e:
                    logger.debug(f"Failed to fetch individual event: {e}")
                    continue
                    
            logger.info(f"Core API approach: {len(all_games)} games")
            return all_games
            
        except Exception as e:
            logger.warning(f"ESPN Core API failed: {e}")
            return []
    
    def _fetch_season_extended_dates_2024(self) -> List[Dict]:
        """Extended date-based approach with multiple dates per week"""
        logger.info("Using extended date range approach...")
        
        # Include Thursday, Sunday, and Monday games for each week
        extended_dates = [
            # Week 1
            ('20240905', 1), ('20240908', 1), ('20240909', 1),
            # Week 2  
            ('20240912', 2), ('20240915', 2), ('20240916', 2),
            # Week 3
            ('20240919', 3), ('20240922', 3), ('20240923', 3),
            # Week 4
            ('20240926', 4), ('20240929', 4), ('20240930', 4),
            # Week 5
            ('20241003', 5), ('20241006', 5), ('20241007', 5),
            # Week 6
            ('20241010', 6), ('20241013', 6), ('20241014', 6),
            # Week 7
            ('20241017', 7), ('20241020', 7), ('20241021', 7),
            # Week 8
            ('20241024', 8), ('20241027', 8), ('20241028', 8),
            # Week 9
            ('20241031', 9), ('20241103', 9), ('20241104', 9),
            # Week 10
            ('20241107', 10), ('20241110', 10), ('20241111', 10),
            # Week 11
            ('20241114', 11), ('20241117', 11), ('20241118', 11),
            # Week 12
            ('20241121', 12), ('20241124', 12), ('20241125', 12),
            # Week 13
            ('20241128', 13), ('20241201', 13), ('20241202', 13),
            # Week 14
            ('20241205', 14), ('20241208', 14), ('20241209', 14),
            # Week 15
            ('20241212', 15), ('20241215', 15), ('20241216', 15),
            # Week 16
            ('20241219', 16), ('20241222', 16), ('20241223', 16),
            # Week 17
            ('20241226', 17), ('20241229', 17), ('20241230', 17),
            # Week 18
            ('20250102', 18), ('20250105', 18), ('20250106', 18),
        ]
        
        all_games = []
        game_ids = set()  # Prevent duplicates
        
        for date_str, week_num in extended_dates:
            try:
                data = self._make_request("sports/football/nfl/scoreboard", {'dates': date_str})
                events = data.get('events', [])
                
                for event in events:
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        game_id = event.get('id')
                        if game_id not in game_ids:
                            event['week_number'] = week_num
                            all_games.append(event)
                            game_ids.add(game_id)
                            
            except Exception as e:
                logger.debug(f"Failed extended date {date_str}: {e}")
                continue
        
        logger.info(f"Extended date approach: {len(all_games)} unique games")
        return all_games
    
    def _get_sample_season_data(self) -> Dict:
        """Fallback sample data when historical API calls fail"""
        logger.info("Using sample season data for initial rankings")
        
        # Sample game data representing typical end-of-season results
        # This is simplified but shows how the ranking system would work
        sample_events = [
            {
                'status': {'type': {'name': 'STATUS_FINAL'}},
                'week_number': 17,
                'competitions': [{
                    'competitors': [
                        {'homeAway': 'home', 'team': {'id': '1'}, 'score': '31'},  # Bills
                        {'homeAway': 'away', 'team': {'id': '15'}, 'score': '10'}  # Dolphins
                    ]
                }]
            },
            {
                'status': {'type': {'name': 'STATUS_FINAL'}},
                'week_number': 17,
                'competitions': [{
                    'competitors': [
                        {'homeAway': 'home', 'team': {'id': '11'}, 'score': '27'},  # Chiefs
                        {'homeAway': 'away', 'team': {'id': '12'}, 'score': '17'}  # Raiders
                    ]
                }]
            }
        ]
        
        return {
            'events': sample_events,
            'week': {'number': 18},
            'season': {'year': 2024}
        }
    
    def _load_verified_complete_dataset(self) -> List[Dict]:
        """Load the verified complete 272-game dataset if available"""
        try:
            # Try to load our verified filtered dataset
            filtered_file = "output/filtered_272_games.json"
            if os.path.exists(filtered_file):
                logger.info("Loading verified 272-game dataset...")
                with open(filtered_file, 'r') as f:
                    data = json.load(f)
                games = data.get('games', [])
                if len(games) == 272:
                    logger.info(f"Successfully loaded verified dataset with {len(games)} games")
                    return games
                else:
                    logger.warning(f"Filtered dataset has {len(games)} games, expected 272")
            else:
                logger.debug("Verified dataset file not found, will use API collection")
                
        except Exception as e:
            logger.warning(f"Failed to load verified dataset: {e}")
        
        return []
    
    def has_current_season_games(self, week: int = None) -> bool:
        """Check if current season has any completed games"""
        try:
            data = self.get_scoreboard(week=week)
            events = data.get('events', [])
            
            completed_games = [
                event for event in events 
                if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'
            ]
            
            return len(completed_games) > 0
        except Exception:
            return False
