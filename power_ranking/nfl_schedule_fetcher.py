#!/usr/bin/env python3
"""
NFL Schedule Fetcher for 2025/26 Season
Fetches Head-to-Head schedule data from ESPN Hidden API and saves to CSV
Compatible with existing power_ranking tools team IDs
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NFLScheduleFetcher:
    def __init__(self, season_year: int = 2025):
        self.season_year = season_year
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.schedule_data = []
        
    def fetch_week_schedule(self, week: int, season_type: int = 2) -> List[Dict]:
        """
        Fetch schedule for a specific week
        season_type: 1=preseason, 2=regular season, 3=postseason
        """
        url = f"{self.base_url}/scoreboard"
        params = {
            'seasontype': season_type,
            'week': week,
            'dates': self.season_year
        }
        
        logger.info(f"Fetching Week {week} schedule...")
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            events = data.get('events', [])
            
            for event in events:
                game_data = self._extract_game_data(event, week)
                if game_data:
                    games.append(game_data)
                    
            logger.info(f"Found {len(games)} games for Week {week}")
            return games
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Week {week}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for Week {week}: {e}")
            return []
    
    def _extract_game_data(self, event: Dict, week: int) -> Dict[str, Any]:
        """Extract relevant game data from ESPN event"""
        try:
            competitions = event.get('competitions', [])
            if not competitions:
                return None
                
            comp = competitions[0]
            competitors = comp.get('competitors', [])
            
            if len(competitors) != 2:
                return None
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            
            if not home_team or not away_team:
                return None
            
            # Extract team IDs and names
            home_id = home_team.get('team', {}).get('id')
            away_id = away_team.get('team', {}).get('id')
            home_name = home_team.get('team', {}).get('displayName', 'Unknown')
            away_name = away_team.get('team', {}).get('displayName', 'Unknown')
            home_abbrev = home_team.get('team', {}).get('abbreviation', 'UNK')
            away_abbrev = away_team.get('team', {}).get('abbreviation', 'UNK')
            
            # Get game date and time
            game_date = event.get('date', '')
            
            # Get venue information
            venue = comp.get('venue', {})
            venue_name = venue.get('fullName', 'Unknown Venue')
            venue_city = venue.get('address', {}).get('city', 'Unknown')
            venue_state = venue.get('address', {}).get('state', 'Unknown')
            
            return {
                'season': self.season_year,
                'week': week,
                'game_id': event.get('id'),
                'game_date': game_date,
                'home_team_id': str(home_id) if home_id else '',
                'home_team_name': home_name,
                'home_team_abbrev': home_abbrev,
                'away_team_id': str(away_id) if away_id else '',
                'away_team_name': away_name,
                'away_team_abbrev': away_abbrev,
                'venue_name': venue_name,
                'venue_city': venue_city,
                'venue_state': venue_state
            }
            
        except Exception as e:
            logger.error(f"Error extracting game data: {e}")
            return None
    
    def fetch_full_season_schedule(self, start_week: int = 1, end_week: int = 18) -> List[Dict]:
        """Fetch schedule for entire regular season"""
        all_games = []
        
        for week in range(start_week, end_week + 1):
            week_games = self.fetch_week_schedule(week)
            all_games.extend(week_games)
            
            # Be respectful to ESPN's servers
            time.sleep(1)
        
        self.schedule_data = all_games
        logger.info(f"Fetched complete schedule: {len(all_games)} games total")
        return all_games
    
    def save_to_csv(self, filename: str = None) -> str:
        """Save schedule data to CSV file"""
        if not self.schedule_data:
            logger.warning("No schedule data to save")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nfl_schedule_{self.season_year}_{timestamp}.csv"
        
        df = pd.DataFrame(self.schedule_data)
        
        # Ensure consistent column order
        columns = [
            'season', 'week', 'game_id', 'game_date',
            'home_team_id', 'home_team_name', 'home_team_abbrev',
            'away_team_id', 'away_team_name', 'away_team_abbrev',
            'venue_name', 'venue_city', 'venue_state'
        ]
        
        df = df[columns]
        df.to_csv(filename, index=False)
        
        logger.info(f"Schedule saved to {filename}")
        return filename
    
    def get_team_matchups(self, team_id: str) -> List[Dict]:
        """Get all matchups for a specific team"""
        team_games = []
        for game in self.schedule_data:
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                team_games.append(game)
        return team_games
    
    def display_summary(self):
        """Display summary of fetched schedule"""
        if not self.schedule_data:
            print("No schedule data available")
            return
        
        total_games = len(self.schedule_data)
        weeks = set(game['week'] for game in self.schedule_data)
        teams = set()
        
        for game in self.schedule_data:
            teams.add(game['home_team_id'])
            teams.add(game['away_team_id'])
        
        print(f"\n2025 NFL Schedule Summary:")
        print(f"Total games: {total_games}")
        print(f"Weeks: {sorted(weeks)}")
        print(f"Teams: {len(teams)}")
        
        # Show sample games
        print(f"\nSample games:")
        for i, game in enumerate(self.schedule_data[:5]):
            print(f"Week {game['week']}: {game['away_team_name']} @ {game['home_team_name']}")


def main():
    """Main execution function"""
    fetcher = NFLScheduleFetcher(2025)
    
    print("Fetching 2025 NFL Regular Season Schedule...")
    schedule = fetcher.fetch_full_season_schedule()
    
    if schedule:
        fetcher.display_summary()
        filename = fetcher.save_to_csv()
        print(f"\nSchedule saved to: {filename}")
    else:
        print("Failed to fetch schedule data")


if __name__ == "__main__":
    main()