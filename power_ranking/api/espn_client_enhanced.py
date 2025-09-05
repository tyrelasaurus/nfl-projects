"""
Enhanced ESPN API client with structured exception handling and error recovery.
Extends the original ESPN client with comprehensive error management.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
import json

# Import custom exceptions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import (
    ESPNAPIError, ESPNRateLimitError, ESPNTimeoutError, ESPNDataError,
    DataProcessingError, DataIncompleteError, handle_exception_with_recovery,
    log_and_handle_error
)

# Import original client
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from espn_client import ESPNClient

logger = logging.getLogger(__name__)

class EnhancedESPNClient(ESPNClient):
    """Enhanced ESPN client with comprehensive error handling."""
    
    def __init__(self, base_url: str = "https://site.api.espn.com/apis/site/v2",
                 max_retries: int = 3, timeout: int = 30, rate_limit_delay: float = 1.0):
        """
        Initialize enhanced ESPN client.
        
        Args:
            base_url: ESPN API base URL
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        super().__init__(base_url)
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.request_count = 0
        self.last_request_time = 0
        
        logger.info(f"Enhanced ESPN client initialized with {max_retries} retries and {timeout}s timeout")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = None) -> Dict:
        """
        Make HTTP request with comprehensive error handling.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            retries: Number of retries (uses instance default if None)
            
        Returns:
            Dict: API response data
            
        Raises:
            ESPNAPIError: For API-related errors
            ESPNRateLimitError: For rate limit errors
            ESPNTimeoutError: For timeout errors
            ESPNDataError: For data validation errors
        """
        if retries is None:
            retries = self.max_retries
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - (current_time - self.last_request_time)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        last_exception = None
        
        for attempt in range(retries):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{retries})")
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # Handle specific HTTP status codes
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise ESPNRateLimitError(
                        f"Rate limit exceeded, retry after {retry_after} seconds",
                        status_code=response.status_code,
                        endpoint=endpoint,
                        context={'retry_after': retry_after, 'attempt': attempt + 1}
                    )
                
                elif response.status_code == 404:
                    raise ESPNAPIError(
                        f"Endpoint not found: {endpoint}",
                        status_code=response.status_code,
                        endpoint=endpoint,
                        context={'url': url, 'params': params}
                    )
                
                elif response.status_code >= 500:
                    # Server errors - retry these
                    raise ESPNAPIError(
                        f"Server error {response.status_code}: {response.reason}",
                        status_code=response.status_code,
                        endpoint=endpoint,
                        context={'attempt': attempt + 1, 'retryable': True}
                    )
                
                # Raise for other HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise ESPNDataError(
                        f"Invalid JSON response from {endpoint}",
                        expected_format="JSON",
                        received_data=response.text[:200],
                        context={'status_code': response.status_code}
                    ) from e
                
                # Validate basic response structure
                self._validate_response_structure(data, endpoint)
                
                logger.debug(f"Successfully retrieved data from {endpoint}")
                return data
                
            except requests.exceptions.Timeout as e:
                last_exception = ESPNTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout} seconds",
                    timeout=self.timeout,
                    endpoint=endpoint,
                    context={'attempt': attempt + 1}
                )
                
            except requests.exceptions.ConnectionError as e:
                last_exception = ESPNAPIError(
                    f"Connection error for {endpoint}: {str(e)}",
                    endpoint=endpoint,
                    context={'attempt': attempt + 1, 'error_type': 'connection'}
                )
                
            except ESPNRateLimitError as e:
                # Don't retry rate limit errors immediately
                if attempt == 0:  # First attempt
                    retry_after = e.context.get('retry_after', 60)
                    logger.warning(f"Rate limited, waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                else:
                    raise e
                    
            except (ESPNAPIError, ESPNDataError) as e:
                # Don't retry client errors (4xx) except rate limits
                if hasattr(e, 'context') and e.context.get('status_code', 0) < 500:
                    raise e
                last_exception = e
                
            except Exception as e:
                last_exception = ESPNAPIError(
                    f"Unexpected error for {endpoint}: {str(e)}",
                    endpoint=endpoint,
                    context={'attempt': attempt + 1, 'error_type': type(e).__name__}
                )
            
            # Exponential backoff for retries
            if attempt < retries - 1:
                wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                logger.warning(f"Request failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise ESPNAPIError(
                f"All {retries} attempts failed for {endpoint}",
                endpoint=endpoint,
                context={'total_attempts': retries}
            )
    
    def _validate_response_structure(self, data: Dict, endpoint: str) -> None:
        """
        Validate basic response structure.
        
        Args:
            data: Response data to validate
            endpoint: API endpoint for context
            
        Raises:
            ESPNDataError: If response structure is invalid
        """
        if not isinstance(data, dict):
            raise ESPNDataError(
                f"Response from {endpoint} is not a dictionary",
                expected_format="dict",
                received_data=type(data).__name__,
                context={'endpoint': endpoint}
            )
        
        # Endpoint-specific validation
        if 'scoreboard' in endpoint:
            self._validate_scoreboard_response(data, endpoint)
        elif 'teams' in endpoint:
            self._validate_teams_response(data, endpoint)
    
    def _validate_scoreboard_response(self, data: Dict, endpoint: str) -> None:
        """Validate scoreboard response structure."""
        if 'events' not in data and 'week' not in data:
            raise ESPNDataError(
                f"Scoreboard response missing required fields",
                expected_format="{'events': [...], 'week': {...}}",
                received_data=list(data.keys()),
                context={'endpoint': endpoint}
            )
    
    def _validate_teams_response(self, data: Dict, endpoint: str) -> None:
        """Validate teams response structure."""
        sports = data.get('sports', [])
        if not sports or not isinstance(sports, list):
            raise ESPNDataError(
                f"Teams response missing 'sports' array",
                expected_format="{'sports': [{'leagues': [...]}]}",
                received_data=list(data.keys()),
                context={'endpoint': endpoint}
            )
    
    @handle_exception_with_recovery
    def get_teams(self) -> List[Dict]:
        """
        Get NFL teams with enhanced error handling.
        
        Returns:
            List[Dict]: List of team data
            
        Raises:
            ESPNAPIError: For API-related errors
            ESPNDataError: For data validation errors
        """
        try:
            data = self._make_request("sports/football/nfl/teams")
            
            # Extract teams with validation
            sports = data.get('sports', [])
            if not sports:
                raise ESPNDataError(
                    "No sports data found in teams response",
                    context={'response_keys': list(data.keys())}
                )
            
            leagues = sports[0].get('leagues', [])
            if not leagues:
                raise ESPNDataError(
                    "No leagues data found in teams response",
                    context={'sports_data': sports[0].keys() if sports else None}
                )
            
            teams = leagues[0].get('teams', [])
            if not teams:
                raise ESPNDataError(
                    "No teams data found in response",
                    context={'leagues_data': leagues[0].keys() if leagues else None}
                )
            
            logger.info(f"Successfully retrieved {len(teams)} NFL teams")
            return teams
            
        except Exception as e:
            log_and_handle_error(e, logger, context={'operation': 'get_teams'})
            raise
    
    @handle_exception_with_recovery
    def get_scoreboard(self, week: Optional[int] = None, season: Optional[int] = None) -> Dict:
        """
        Get scoreboard with enhanced error handling.
        
        Args:
            week: Week number (optional)
            season: Season year (optional)
            
        Returns:
            Dict: Scoreboard data
            
        Raises:
            ESPNAPIError: For API-related errors
            DataProcessingError: For parameter validation errors
        """
        # Validate parameters
        if week is not None and (not isinstance(week, int) or week < 1 or week > 22):
            raise DataProcessingError(
                f"Invalid week number: {week}",
                context={'week': week, 'valid_range': '1-22'}
            )
        
        if season is not None and (not isinstance(season, int) or season < 1990 or season > 2030):
            raise DataProcessingError(
                f"Invalid season year: {season}",
                context={'season': season, 'valid_range': '1990-2030'}
            )
        
        params = {}
        if week:
            params['week'] = week
        if season:
            params['seasontype'] = '2'  # Regular season
            params['year'] = str(season)
        
        try:
            data = self._make_request("sports/football/nfl/scoreboard", params=params)
            
            # Validate we got events
            events = data.get('events', [])
            logger.info(f"Retrieved scoreboard with {len(events)} events for week {week}, season {season}")
            
            return data
            
        except Exception as e:
            context = {'week': week, 'season': season, 'operation': 'get_scoreboard'}
            log_and_handle_error(e, logger, context=context)
            raise
    
    @handle_exception_with_recovery
    def get_current_week(self) -> int:
        """
        Get current NFL week with enhanced error handling.
        
        Returns:
            int: Current week number
        """
        try:
            data = self.get_scoreboard()
            week = data.get('week', {}).get('number')
            
            if not isinstance(week, int):
                logger.warning("Could not determine current week from API, using fallback")
                return self._get_fallback_week()
            
            logger.info(f"Current NFL week: {week}")
            return week
            
        except Exception as e:
            logger.warning(f"Failed to get current week: {e}")
            return self._get_fallback_week()
    
    def _get_fallback_week(self) -> int:
        """
        Get fallback week number based on date.
        
        Returns:
            int: Estimated current week
        """
        import datetime
        
        # Simple fallback based on date
        now = datetime.datetime.now()
        
        # NFL season typically starts first week of September
        if now.month >= 9:
            # Rough calculation: first week of September is week 1
            september_first = datetime.datetime(now.year, 9, 1)
            days_since_start = (now - september_first).days
            week = max(1, min(18, (days_since_start // 7) + 1))
        elif now.month <= 2:
            # Likely playoffs or end of season
            week = 18
        else:
            # Offseason
            week = 1
        
        logger.info(f"Using fallback week calculation: {week}")
        return week
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dict: Error statistics
        """
        return {
            'total_requests': self.request_count,
            'client_config': {
                'max_retries': self.max_retries,
                'timeout': self.timeout,
                'rate_limit_delay': self.rate_limit_delay
            },
            'last_request_time': self.last_request_time
        }