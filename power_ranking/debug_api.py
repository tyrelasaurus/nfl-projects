#!/usr/bin/env python3
"""
Debug script to test different ESPN API endpoints and parameters
to identify the best approach for getting complete 2024 season data
"""
import requests
import json
from datetime import datetime, timedelta

def test_espn_endpoint(endpoint, params=None, description=""):
    """Test an ESPN API endpoint and return results"""
    base_url = "https://site.api.espn.com/apis/site/v2"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"Params: {params}")
    print(f"{'='*60}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Analyze the response
        events = data.get('events', [])
        completed_games = [e for e in events if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL']
        
        print(f"Total events: {len(events)}")
        print(f"Completed games: {len(completed_games)}")
        
        if completed_games:
            print(f"Sample game week: {completed_games[0].get('week', {}).get('number', 'N/A')}")
            print(f"Season info: {data.get('season', {})}")
            print(f"Week info: {data.get('week', {})}")
            
            # Show first few games
            for i, game in enumerate(completed_games[:3]):
                competitors = game.get('competitions', [{}])[0].get('competitors', [])
                if len(competitors) >= 2:
                    home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                    away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                    home_team = home.get('team', {}).get('displayName', 'Unknown')
                    away_team = away.get('team', {}).get('displayName', 'Unknown')
                    home_score = home.get('score', 0)
                    away_score = away.get('score', 0)
                    print(f"  Game {i+1}: {away_team} @ {home_team} ({away_score}-{home_score})")
        
        return len(completed_games), data
    
    except Exception as e:
        print(f"ERROR: {e}")
        return 0, None

def main():
    print("ESPN API Debug Tool - Testing 2024 Season Data Collection")
    print("="*80)
    
    # Test different approaches to get 2024 season data
    test_cases = [
        # Current approach - weekly with season parameter
        ("sports/football/nfl/scoreboard", 
         {'week': 1, 'seasontype': '2', 'year': '2024'}, 
         "Week 1 with season params"),
         
        ("sports/football/nfl/scoreboard", 
         {'week': 10, 'seasontype': '2', 'year': '2024'}, 
         "Week 10 with season params"),
         
        # Try without season type
        ("sports/football/nfl/scoreboard", 
         {'week': 1, 'year': '2024'}, 
         "Week 1 with year only"),
         
        # Try different season types
        ("sports/football/nfl/scoreboard", 
         {'week': 1, 'seasontype': '1', 'year': '2024'}, 
         "Week 1 preseason (type 1)"),
         
        ("sports/football/nfl/scoreboard", 
         {'week': 1, 'seasontype': '3', 'year': '2024'}, 
         "Week 1 playoffs (type 3)"),
         
        # Date-based approaches
        ("sports/football/nfl/scoreboard", 
         {'dates': '20240908'}, 
         "Date-based Week 1 (Sep 8, 2024)"),
         
        ("sports/football/nfl/scoreboard", 
         {'dates': '20241103'}, 
         "Date-based Week 9 (Nov 3, 2024)"),
         
        # Try season endpoint
        ("sports/football/nfl/scoreboard", 
         {'season': '2024', 'seasontype': '2'}, 
         "Season parameter instead of year"),
         
        # Try getting current season info
        ("sports/football/nfl/scoreboard", 
         {}, 
         "Current/default scoreboard"),
    ]
    
    total_games = 0
    successful_endpoints = []
    
    for endpoint, params, description in test_cases:
        games_found, data = test_espn_endpoint(endpoint, params, description)
        total_games += games_found
        
        if games_found > 0:
            successful_endpoints.append((description, games_found, params))
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total games found across all tests: {total_games}")
    print(f"Successful endpoints: {len(successful_endpoints)}")
    
    if successful_endpoints:
        print("\nBest approaches:")
        for desc, games, params in sorted(successful_endpoints, key=lambda x: x[1], reverse=True):
            print(f"  {desc}: {games} games - {params}")
    
    # Now test a comprehensive date range approach
    print(f"\n{'='*60}")
    print("Testing comprehensive date-based approach for full season")
    print(f"{'='*60}")
    
    # Test getting multiple weeks via date ranges
    season_2024_dates = [
        ('20240908', 'Week 1'),
        ('20240915', 'Week 2'), 
        ('20240922', 'Week 3'),
        ('20240929', 'Week 4'),
        ('20241006', 'Week 5'),
        ('20241013', 'Week 6'),
        ('20241020', 'Week 7'),
        ('20241027', 'Week 8'),
        ('20241103', 'Week 9'),
        ('20241110', 'Week 10'),
    ]
    
    total_season_games = 0
    for date_str, week_name in season_2024_dates:
        games, _ = test_espn_endpoint(
            "sports/football/nfl/scoreboard", 
            {'dates': date_str}, 
            f"Date-based {week_name} ({date_str})"
        )
        total_season_games += games
    
    print(f"\nTotal games from date-based approach (10 weeks): {total_season_games}")

if __name__ == "__main__":
    main()