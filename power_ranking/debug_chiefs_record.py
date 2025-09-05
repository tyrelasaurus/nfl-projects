#!/usr/bin/env python3
"""
Debug script to examine Chiefs' 2024 season record and validate data quality
"""
import sys
sys.path.append('.')

from api.espn_client import ESPNClient
import json

def debug_chiefs_games():
    client = ESPNClient()
    
    print("Fetching 2024 season data...")
    season_data = client.get_last_season_final_rankings()
    events = season_data.get('events', [])
    
    print(f"Total games found: {len(events)}")
    
    # Find all team IDs and look for Chiefs
    chiefs_team_ids = set()
    team_names = {}
    
    for event in events:  # Check ALL events, not just first 10
        competitions = event.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            for competitor in competitors:
                team = competitor.get('team', {})
                team_id = team.get('id')
                team_name = team.get('displayName', '')
                if team_id:
                    team_names[team_id] = team_name
                    if 'Chiefs' in team_name or 'Kansas City' in team_name:
                        chiefs_team_ids.add(team_id)
    
    print("\nTeam names found:")
    for team_id, name in sorted(team_names.items()):
        print(f"  {team_id}: {name}")
    
    print(f"\nChiefs team IDs found: {chiefs_team_ids}")
    
    # Find all Chiefs games
    chiefs_games = []
    for event in events:
        competitions = event.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            team_ids = [comp.get('team', {}).get('id') for comp in competitors]
            
            # Check if Chiefs are in this game
            for chiefs_id in chiefs_team_ids:
                if chiefs_id in team_ids:
                    chiefs_games.append(event)
                    break
    
    print(f"\nFound {len(chiefs_games)} Chiefs games")
    
    # Analyze Chiefs games
    wins = 0
    losses = 0
    
    print("\nChiefs 2024 Regular Season Games:")
    print("=" * 80)
    
    for i, game in enumerate(sorted(chiefs_games, key=lambda x: x.get('week_number', 0)), 1):
        competitions = game.get('competitions', [])
        if not competitions:
            continue
            
        competitors = competitions[0].get('competitors', [])
        week = game.get('week_number', '?')
        
        # Find Chiefs and opponent
        chiefs_competitor = None
        opponent_competitor = None
        
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if 'Chiefs' in team_name:
                chiefs_competitor = comp
            else:
                opponent_competitor = comp
        
        if not (chiefs_competitor and opponent_competitor):
            continue
            
        chiefs_score = int(chiefs_competitor.get('score', '0'))
        opponent_score = int(opponent_competitor.get('score', '0'))
        opponent_name = opponent_competitor.get('team', {}).get('displayName', 'Unknown')
        
        is_home = chiefs_competitor.get('homeAway') == 'home'
        location = "vs" if is_home else "@"
        
        result = "W" if chiefs_score > opponent_score else "L"
        if result == "W":
            wins += 1
        else:
            losses += 1
            
        print(f"Week {week:2d}: {result} {chiefs_score}-{opponent_score} {location} {opponent_name}")
    
    print("=" * 80)
    print(f"Chiefs Record: {wins}-{losses}")
    print(f"Expected: 15-2")
    
    if wins != 15 or losses != 2:
        print(f"❌ MISMATCH! Expected 15-2, got {wins}-{losses}")
        
        # Check for potential issues
        print("\nDiagnosing issues...")
        
        # Check for duplicate games
        game_ids = [game.get('id') for game in chiefs_games]
        unique_game_ids = set(game_ids)
        if len(game_ids) != len(unique_game_ids):
            print(f"❌ Found duplicate games: {len(game_ids)} total, {len(unique_game_ids)} unique")
            
        # Check week distribution
        weeks = [game.get('week_number') for game in chiefs_games]
        week_counts = {}
        for week in weeks:
            week_counts[week] = week_counts.get(week, 0) + 1
        
        print(f"Week distribution: {dict(sorted(week_counts.items()))}")
        
        # Check for any playoff games that slipped through
        playoff_games = []
        for game in chiefs_games:
            week = game.get('week_number', 0)
            if week > 18:
                playoff_games.append(game)
        
        if playoff_games:
            print(f"❌ Found {len(playoff_games)} playoff games that slipped through filter")
            
    else:
        print("✅ Record matches expected 15-2")

if __name__ == "__main__":
    debug_chiefs_games()