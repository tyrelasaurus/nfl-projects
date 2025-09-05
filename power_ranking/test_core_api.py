#!/usr/bin/env python3
"""
Test ESPN Core API for accurate Week 18 data
"""
import requests
import json

def test_core_api_week18():
    print("Testing ESPN Core API for Week 18 2024 data...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        # Try ESPN Core API for 2024 season events
        core_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/weeks/18/events"
        response = session.get(f"{core_url}?limit=50", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"Core API returned {len(data.get('items', []))} Week 18 events")
        
        # Check each event
        for item in data.get('items', []):
            event_url = item.get('$ref')
            if event_url:
                try:
                    event_response = session.get(event_url, timeout=30)
                    event_data = event_response.json()
                    
                    # Check if this involves Chiefs or Broncos
                    competitors = event_data.get('competitions', [{}])[0].get('competitors', [])
                    team_names = []
                    
                    for comp in competitors:
                        team_name = comp.get('team', {}).get('displayName', '')
                        team_names.append(team_name)
                    
                    if 'Kansas City Chiefs' in team_names or 'Denver Broncos' in team_names:
                        print(f"\nFound relevant game: {' vs '.join(team_names)}")
                        
                        for comp in competitors:
                            team_name = comp.get('team', {}).get('displayName', '')
                            score = comp.get('score', '0')
                            home_away = comp.get('homeAway', '')
                            print(f"  {team_name}: {score} ({home_away})")
                        
                        if 'Kansas City Chiefs' in team_names and 'Denver Broncos' in team_names:
                            print("  ✅ Found Chiefs @ Broncos game!")
                            return True
                            
                except Exception as e:
                    print(f"Error fetching event: {e}")
                    continue
        
        print("❌ Chiefs @ Broncos game not found in Core API either")
        return False
        
    except Exception as e:
        print(f"Core API failed: {e}")
        return False

def test_date_specific_api():
    print("\nTesting date-specific API for January 5, 2025 (Week 18)...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        # Test specific date for Week 18
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        response = session.get(url, params={'dates': '20250105'}, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        events = data.get('events', [])
        
        print(f"Found {len(events)} games on Jan 5, 2025")
        
        for event in events:
            competitions = event.get('competitions', [])
            if competitions:
                competitors = competitions[0].get('competitors', [])
                team_names = [comp.get('team', {}).get('displayName', '') for comp in competitors]
                
                if 'Kansas City Chiefs' in team_names:
                    print(f"\nFound Chiefs game: {' @ '.join(team_names[::-1])}")
                    for comp in competitors:
                        team_name = comp.get('team', {}).get('displayName', '')
                        score = comp.get('score', '0')
                        home_away = comp.get('homeAway', '')
                        print(f"  {team_name}: {score} ({home_away})")
                    
                    if 'Denver Broncos' in team_names:
                        print("  ✅ This is the Chiefs @ Broncos game!")
                        return True
        
        print("❌ Chiefs @ Broncos game not found on Jan 5, 2025 either")
        return False
        
    except Exception as e:
        print(f"Date-specific API failed: {e}")
        return False

if __name__ == "__main__":
    found_core = test_core_api_week18()
    if not found_core:
        found_date = test_date_specific_api()
        
    if not (found_core or found_date):
        print("\n❌ CRITICAL: Neither API source has accurate Chiefs @ Broncos Week 18 data")
        print("This suggests ESPN's API data is incomplete or incorrect for this game.")