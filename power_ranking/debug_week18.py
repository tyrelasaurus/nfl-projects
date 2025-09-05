#!/usr/bin/env python3
"""
Debug script to examine Week 18 games and validate data accuracy
"""
import sys
sys.path.append('.')

from api.espn_client import ESPNClient
import json

def debug_week18_games():
    client = ESPNClient()
    
    print("Fetching 2024 season data...")
    season_data = client.get_last_season_final_rankings()
    events = season_data.get('events', [])
    
    print(f"Total games found: {len(events)}")
    
    # Find all Week 18 games
    week18_games = []
    for event in events:
        week_num = event.get('week_number')
        if week_num == 18:
            week18_games.append(event)
    
    print(f"\nFound {len(week18_games)} Week 18 games")
    print("\nWeek 18 Games:")
    print("=" * 100)
    
    chiefs_week18 = None
    broncos_week18 = None
    
    for game in week18_games:
        competitions = game.get('competitions', [])
        if not competitions:
            continue
            
        competitors = competitions[0].get('competitors', [])
        if len(competitors) != 2:
            continue
            
        home_team = away_team = None
        home_score = away_score = None
        
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            score = comp.get('score', '0')
            
            if comp.get('homeAway') == 'home':
                home_team = team_name
                home_score = score
            else:
                away_team = team_name
                away_score = score
        
        game_desc = f"{away_team} @ {home_team} ({away_score}-{home_score})"
        print(f"  {game_desc}")
        
        # Check for Chiefs and Broncos games
        if 'Chiefs' in away_team or 'Chiefs' in home_team:
            chiefs_week18 = game
            print(f"    ^^^ CHIEFS GAME: {game_desc}")
            
        if 'Broncos' in away_team or 'Broncos' in home_team:
            broncos_week18 = game
            if 'Chiefs' in away_team or 'Chiefs' in home_team:
                print(f"    ^^^ CHIEFS vs BRONCOS GAME!")
    
    print("=" * 100)
    
    # Analyze the Chiefs Week 18 game
    if chiefs_week18:
        print(f"\nChiefs Week 18 Game Analysis:")
        print(f"Game ID: {chiefs_week18.get('id')}")
        
        competitions = chiefs_week18.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            
            print("Competitors:")
            for comp in competitors:
                team = comp.get('team', {})
                print(f"  Team ID: {team.get('id')}, Name: {team.get('displayName')}")
                print(f"  Score: {comp.get('score')}, Home/Away: {comp.get('homeAway')}")
    
    # Check if we're missing the actual Chiefs @ Broncos game
    print(f"\nChecking for Chiefs @ Broncos Week 18...")
    found_chiefs_broncos = False
    
    for game in week18_games:
        competitions = game.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            team_names = [comp.get('team', {}).get('displayName', '') for comp in competitors]
            
            has_chiefs = any('Chiefs' in name for name in team_names)
            has_broncos = any('Broncos' in name for name in team_names)
            
            if has_chiefs and has_broncos:
                found_chiefs_broncos = True
                print(f"✅ Found Chiefs @ Broncos game!")
                for comp in competitors:
                    team_name = comp.get('team', {}).get('displayName', '')
                    score = comp.get('score', '0')
                    home_away = comp.get('homeAway')
                    print(f"  {team_name}: {score} ({home_away})")
                break
    
    if not found_chiefs_broncos:
        print(f"❌ Chiefs @ Broncos Week 18 game NOT FOUND in our data!")
        print("This suggests the ESPN API is not returning the correct Week 18 data.")

if __name__ == "__main__":
    debug_week18_games()