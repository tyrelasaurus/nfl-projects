"""
Async ESPN API Client with concurrent requests and intelligent caching.
This provides significant performance improvements over the synchronous client.
"""

import aiohttp
import asyncio
import time
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class AsyncESPNClient:
    """High-performance async ESPN client with concurrent requests and caching."""
    
    def __init__(self, base_url: str = "https://site.api.espn.com/apis/site/v2", 
                 max_concurrent: int = 10, cache_ttl: int = 300):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.cache_ttl = cache_ttl  # 5 minutes default
        self.cache = {}
        self.session = None
        
        # Request rate limiting
        self.rate_limiter = asyncio.Semaphore(max_concurrent)
        self.min_request_interval = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for request."""
        key_data = f"{endpoint}:{params or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_time, _ = self.cache[cache_key]
        return (time.time() - cached_time) < self.cache_ttl
    
    async def _rate_limited_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make rate-limited request with intelligent backoff."""
        async with self.rate_limiter:
            # Ensure minimum time between requests
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time = time.time()
            return await self._make_request(endpoint, params)
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                          retries: int = 3) -> Dict:
        """Make async HTTP request with exponential backoff."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {endpoint}")
            return self.cache[cache_key][1]
        
        for attempt in range(retries):
            try:
                logger.debug(f"Making async request to {url} (attempt {attempt + 1})")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Cache successful response
                    self.cache[cache_key] = (time.time(), data)
                    logger.debug(f"Cached response for {endpoint}")
                    
                    return data
                    
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def get_teams(self) -> List[Dict]:
        """Get NFL teams asynchronously."""
        try:
            data = await self._rate_limited_request("sports/football/nfl/teams")
            return data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
        except Exception as e:
            logger.error(f"Failed to fetch teams: {e}")
            raise
    
    async def get_scoreboard(self, week: Optional[int] = None, 
                           season: Optional[int] = None) -> Dict:
        """Get scoreboard for specific week/season."""
        params = {}
        if week:
            params['week'] = week
        if season:
            params['seasontype'] = '2'  # Regular season
            params['year'] = str(season)
            
        try:
            data = await self._rate_limited_request("sports/football/nfl/scoreboard", params=params)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch scoreboard for week {week}, season {season}: {e}")
            raise
    
    async def get_multiple_scoreboards(self, requests: List[Tuple[int, int]]) -> List[Dict]:
        """Get multiple scoreboards concurrently."""
        logger.info(f"Fetching {len(requests)} scoreboards concurrently")
        
        async def fetch_scoreboard(week: int, season: int) -> Dict:
            try:
                data = await self.get_scoreboard(week, season)
                data['_request_info'] = {'week': week, 'season': season}
                return data
            except Exception as e:
                logger.error(f"Failed to fetch week {week}, season {season}: {e}")
                return {'events': [], '_request_info': {'week': week, 'season': season, 'error': str(e)}}
        
        # Create tasks for concurrent execution
        tasks = [fetch_scoreboard(week, season) for week, season in requests]
        
        # Execute with controlled concurrency
        results = []
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch request failed: {result}")
                    results.append({'events': [], 'error': str(result)})
                else:
                    results.append(result)
        
        logger.info(f"Completed {len(results)} scoreboard requests")
        return results
    
    async def get_current_week(self) -> int:
        """Get current NFL week."""
        try:
            data = await self.get_scoreboard()
            return data.get('week', {}).get('number', 1)
        except Exception:
            logger.warning("Could not determine current week, defaulting to 1")
            return 1
    
    async def get_last_season_final_rankings(self) -> Dict:
        """Get comprehensive 2024 season data using concurrent requests."""
        try:
            logger.info("Fetching complete 2024 season data with concurrent requests...")
            
            # Method 1: Try to use verified complete dataset first
            complete_games = await self._load_verified_complete_dataset()
            if complete_games and len(complete_games) == 272:
                logger.info(f"Using verified complete dataset with {len(complete_games)} games")
                return {
                    'events': complete_games,
                    'week': {'number': 18},
                    'season': {'year': 2024},
                    'total_games': len(complete_games)
                }
            
            # Method 2: Concurrent date-based approach
            all_games = await self._fetch_season_concurrent_dates_2024()
            
            # Method 3: Fallback to extended concurrent approach
            if len(all_games) < 270:
                logger.info("Basic concurrent approach insufficient, trying extended...")
                extended_games = await self._fetch_season_extended_concurrent_2024()
                if len(extended_games) > len(all_games):
                    all_games = extended_games
            
            if not all_games:
                logger.warning("All concurrent methods failed, using sample data")
                return await self._get_sample_season_data()
            
            logger.info(f"Successfully fetched {len(all_games)} games from 2024 season")
            
            return {
                'events': all_games,
                'week': {'number': 18},
                'season': {'year': 2024},
                'total_games': len(all_games)
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch 2024 season data: {e}")
            return await self._get_sample_season_data()
    
    async def _fetch_season_concurrent_dates_2024(self) -> List[Dict]:
        """Fetch 2024 season using concurrent date-based requests."""
        logger.info("Using concurrent date-based approach for 2024 season...")
        
        # Complete 2024 NFL season dates
        season_dates = [
            ('20240908', 1), ('20240915', 2), ('20240922', 3), ('20240929', 4),
            ('20241006', 5), ('20241013', 6), ('20241020', 7), ('20241027', 8),
            ('20241103', 9), ('20241110', 10), ('20241117', 11), ('20241124', 12),
            ('20241201', 13), ('20241208', 14), ('20241215', 15), ('20241222', 16),
            ('20241229', 17), ('20250105', 18)
        ]
        
        async def fetch_week_data(date_str: str, week_num: int) -> List[Dict]:
            """Fetch data for a specific week."""
            try:
                data = await self._rate_limited_request("sports/football/nfl/scoreboard", 
                                                      {'dates': date_str})
                events = data.get('events', [])
                
                completed_games = []
                for event in events:
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        event['week_number'] = week_num
                        completed_games.append(event)
                
                logger.debug(f"Week {week_num}: Found {len(completed_games)} games")
                return completed_games
                
            except Exception as e:
                logger.warning(f"Failed to fetch week {week_num}: {e}")
                return []
        
        # Create tasks for all weeks
        tasks = [fetch_week_data(date_str, week_num) for date_str, week_num in season_dates]
        
        # Execute with controlled concurrency
        all_results = []
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Week fetch failed: {result}")
                    continue
                else:
                    all_results.extend(result)
        
        logger.info(f"Concurrent date approach: {len(all_results)} games")
        return all_results
    
    async def _fetch_season_extended_concurrent_2024(self) -> List[Dict]:
        """Extended concurrent approach with multiple dates per week."""
        logger.info("Using extended concurrent date approach...")
        
        # Include multiple days per week for comprehensive coverage
        extended_dates = [
            # Week 1
            ('20240905', 1), ('20240908', 1), ('20240909', 1),
            # Week 2  
            ('20240912', 2), ('20240915', 2), ('20240916', 2),
            # Continue for all weeks...
            ('20241003', 5), ('20241006', 5), ('20241007', 5),
            ('20241010', 6), ('20241013', 6), ('20241014', 6),
            ('20241017', 7), ('20241020', 7), ('20241021', 7),
            ('20241024', 8), ('20241027', 8), ('20241028', 8),
            ('20241031', 9), ('20241103', 9), ('20241104', 9),
            ('20241107', 10), ('20241110', 10), ('20241111', 10),
            ('20241114', 11), ('20241117', 11), ('20241118', 11),
            ('20241121', 12), ('20241124', 12), ('20241125', 12),
            ('20241128', 13), ('20241201', 13), ('20241202', 13),
            ('20241205', 14), ('20241208', 14), ('20241209', 14),
            ('20241212', 15), ('20241215', 15), ('20241216', 15),
            ('20241219', 16), ('20241222', 16), ('20241223', 16),
            ('20241226', 17), ('20241229', 17), ('20241230', 17),
            ('20250102', 18), ('20250105', 18), ('20250106', 18),
        ]
        
        async def fetch_date_data(date_str: str, week_num: int) -> List[Dict]:
            """Fetch games for a specific date."""
            try:
                data = await self._rate_limited_request("sports/football/nfl/scoreboard", 
                                                      {'dates': date_str})
                events = data.get('events', [])
                
                completed_games = []
                for event in events:
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        event['week_number'] = week_num
                        completed_games.append(event)
                
                return completed_games
                
            except Exception as e:
                logger.debug(f"Failed extended date {date_str}: {e}")
                return []
        
        # Create tasks for all dates
        tasks = [fetch_date_data(date_str, week_num) for date_str, week_num in extended_dates]
        
        # Execute with controlled concurrency
        all_games = []
        game_ids = set()  # Prevent duplicates
        
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    continue
                
                for game in result:
                    game_id = game.get('id')
                    if game_id and game_id not in game_ids:
                        all_games.append(game)
                        game_ids.add(game_id)
        
        logger.info(f"Extended concurrent approach: {len(all_games)} unique games")
        return all_games
    
    async def _load_verified_complete_dataset(self) -> List[Dict]:
        """Load verified dataset asynchronously."""
        try:
            filtered_file = "output/filtered_272_games.json"
            if os.path.exists(filtered_file):
                logger.info("Loading verified 272-game dataset...")
                
                loop = asyncio.get_event_loop()
                
                def load_file():
                    with open(filtered_file, 'r') as f:
                        return json.load(f)
                
                data = await loop.run_in_executor(None, load_file)
                games = data.get('games', [])
                
                if len(games) == 272:
                    logger.info(f"Successfully loaded verified dataset with {len(games)} games")
                    return games
                else:
                    logger.warning(f"Filtered dataset has {len(games)} games, expected 272")
            else:
                logger.debug("Verified dataset file not found")
                
        except Exception as e:
            logger.warning(f"Failed to load verified dataset: {e}")
        
        return []
    
    async def _get_sample_season_data(self) -> Dict:
        """Get sample season data asynchronously."""
        logger.info("Using sample season data for initial rankings")
        
        sample_events = [
            {
                'status': {'type': {'name': 'STATUS_FINAL'}},
                'week_number': 17,
                'competitions': [{
                    'competitors': [
                        {'homeAway': 'home', 'team': {'id': '1'}, 'score': '31'},
                        {'homeAway': 'away', 'team': {'id': '15'}, 'score': '10'}
                    ]
                }]
            },
            {
                'status': {'type': {'name': 'STATUS_FINAL'}},
                'week_number': 17,
                'competitions': [{
                    'competitors': [
                        {'homeAway': 'home', 'team': {'id': '11'}, 'score': '27'},
                        {'homeAway': 'away', 'team': {'id': '12'}, 'score': '17'}
                    ]
                }]
            }
        ]
        
        return {
            'events': sample_events,
            'week': {'number': 18},
            'season': {'year': 2024}
        }
    
    async def has_current_season_games(self, week: int = None) -> bool:
        """Check if current season has completed games."""
        try:
            data = await self.get_scoreboard(week=week)
            events = data.get('events', [])
            
            completed_games = [
                event for event in events 
                if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'
            ]
            
            return len(completed_games) > 0
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        valid_entries = sum(1 for key in self.cache.keys() if self._is_cache_valid(key))
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'cache_hit_ratio': valid_entries / max(len(self.cache), 1)
        }