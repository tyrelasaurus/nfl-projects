#!/usr/bin/env python3
"""
Analyze the extra 16 games in our 288-game dataset to identify duplicates or non-regular season games
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

def analyze_extra_games():
    # Load the debug results
    with open('/Users/tyrelshaw/Projects/power_ranking/output/debug_full_season_20250831_212411.json', 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"Total games loaded: {len(games)}")
    print(f"Expected regular season games: 272")
    print(f"Extra games: {len(games) - 272}")
    print("=" * 80)
    
    # Analyze game types and seasons
    season_types = Counter()
    week_counts = Counter()
    game_ids = []
    duplicate_ids = []
    
    for game in games:
        game_id = game.get('id')
        if game_id:
            if game_id in game_ids:
                duplicate_ids.append(game_id)
            else:
                game_ids.append(game_id)
        
        # Check season info
        season = game.get('season', {})
        season_type = season.get('type', 'Unknown')
        season_year = season.get('year', 'Unknown')
        
        # Check week info
        week_num = game.get('week', {}).get('number', game.get('week_number', 'Unknown'))
        
        season_types[f"{season_year}-Type{season_type}"] += 1
        week_counts[week_num] += 1
    
    print("DUPLICATE GAME IDS:")
    print(f"Found {len(duplicate_ids)} duplicate game IDs: {duplicate_ids}")
    print()
    
    print("SEASON TYPE BREAKDOWN:")
    for season_type, count in season_types.most_common():
        print(f"  {season_type}: {count} games")
    print()
    
    print("WEEK BREAKDOWN:")
    total_regular_weeks = 0
    for week in sorted(week_counts.keys()):
        count = week_counts[week]
        if isinstance(week, int) and 1 <= week <= 18:
            total_regular_weeks += count
            print(f"  Week {week:2d}: {count} games")
        else:
            print(f"  Week {week}: {count} games ⚠️ (Non-regular season)")
    
    print(f"\nTotal regular season weeks (1-18): {total_regular_weeks} games")
    print(f"Non-regular season games: {len(games) - total_regular_weeks}")
    print()
    
    # Look for specific problematic games
    print("DETAILED ANALYSIS OF POTENTIALLY EXTRA GAMES:")
    print("=" * 80)
    
    # Find games that might be duplicates or non-regular season
    suspicious_games = []
    
    for game in games:
        season = game.get('season', {})
        season_type = season.get('type')
        week_num = game.get('week', {}).get('number', game.get('week_number'))
        game_id = game.get('id')
        
        # Flag suspicious games
        is_suspicious = False
        reasons = []
        
        # Non-regular season type
        if season_type != 2:
            is_suspicious = True
            reasons.append(f"Season type {season_type} (not regular season)")
        
        # Week outside 1-18 range
        if not (isinstance(week_num, int) and 1 <= week_num <= 18):
            is_suspicious = True
            reasons.append(f"Week {week_num} (outside 1-18)")
        
        # Duplicate game ID
        if duplicate_ids and game_id in duplicate_ids:
            is_suspicious = True
            reasons.append("Duplicate game ID")
        
        if is_suspicious:
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            team_names = []
            for comp in competitors:
                team_name = comp.get('team', {}).get('displayName', 'Unknown')
                team_names.append(team_name)
            
            game_name = ' vs '.join(team_names) if len(team_names) >= 2 else game.get('name', 'Unknown')
            
            suspicious_games.append({
                'id': game_id,
                'name': game_name,
                'week': week_num,
                'season_type': season_type,
                'reasons': reasons,
                'date': game.get('date', 'Unknown')
            })
    
    print(f"Found {len(suspicious_games)} suspicious games:")
    for i, game in enumerate(suspicious_games[:20]):  # Show first 20
        print(f"{i+1:2d}. {game['name']}")
        print(f"    ID: {game['id']}, Week: {game['week']}, Type: {game['season_type']}")
        print(f"    Date: {game['date']}")
        print(f"    Issues: {', '.join(game['reasons'])}")
        print()
    
    if len(suspicious_games) > 20:
        print(f"... and {len(suspicious_games) - 20} more suspicious games")
    
    # Analyze Week 18 specifically (it should have 16 games, not 32)
    week18_games = [g for g in games if g.get('week', {}).get('number', g.get('week_number')) == 18]
    print(f"\nWEEK 18 ANALYSIS:")
    print(f"Week 18 games found: {len(week18_games)} (expected: 16)")
    
    if len(week18_games) > 16:
        print("Week 18 games (showing duplicates/extras):")
        game_matchups = {}
        for game in week18_games:
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            if len(competitors) >= 2:
                teams = sorted([comp.get('team', {}).get('displayName', '') for comp in competitors])
                matchup = f"{teams[0]} vs {teams[1]}"
                
                if matchup not in game_matchups:
                    game_matchups[matchup] = []
                game_matchups[matchup].append({
                    'id': game.get('id'),
                    'date': game.get('date'),
                    'season_type': game.get('season', {}).get('type')
                })
        
        for matchup, games_list in game_matchups.items():
            if len(games_list) > 1:
                print(f"  DUPLICATE: {matchup} ({len(games_list)} instances)")
                for g in games_list:
                    print(f"    - ID: {g['id']}, Date: {g['date']}, Type: {g['season_type']}")
            elif len(games_list) == 1:
                g = games_list[0]
                print(f"  {matchup} - ID: {g['id']}, Type: {g['season_type']}")
    
    return suspicious_games

def create_filtered_dataset():
    """Create a filtered dataset with exactly 272 regular season games"""
    print("\n" + "=" * 80)
    print("CREATING FILTERED DATASET (272 REGULAR SEASON GAMES)")
    print("=" * 80)
    
    # Load the debug results
    with open('/Users/tyrelshaw/Projects/power_ranking/output/debug_full_season_20250831_212411.json', 'r') as f:
        data = json.load(f)
    
    games = data['games']
    filtered_games = []
    seen_game_ids = set()
    week_counts = defaultdict(int)
    
    # Filter to regular season games only, removing duplicates
    for game in games:
        game_id = game.get('id')
        season = game.get('season', {})
        season_type = season.get('type')
        season_year = season.get('year')
        week_num = game.get('week', {}).get('number', game.get('week_number'))
        
        # Only include regular season games (type 2) from 2024, weeks 1-18, no duplicates
        if (season_type == 2 and 
            season_year == 2024 and 
            isinstance(week_num, int) and 
            1 <= week_num <= 18 and
            game_id not in seen_game_ids):
            
            filtered_games.append(game)
            seen_game_ids.add(game_id)
            week_counts[week_num] += 1
    
    print(f"Filtered games: {len(filtered_games)}")
    print("Week breakdown:")
    total = 0
    for week in range(1, 19):
        count = week_counts[week]
        total += count
        print(f"  Week {week:2d}: {count} games")
    
    print(f"Total: {total} games")
    
    # Save filtered dataset
    filtered_data = {
        'timestamp': data['timestamp'] + '_filtered',
        'total_games': len(filtered_games),
        'filter_applied': 'Regular season 2024 only, no duplicates',
        'games': filtered_games
    }
    
    output_file = '/Users/tyrelshaw/Projects/power_ranking/output/filtered_272_games.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\nFiltered dataset saved to: {output_file}")
    return filtered_games

if __name__ == "__main__":
    suspicious_games = analyze_extra_games()
    filtered_games = create_filtered_dataset()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original games: 288")
    print(f"Suspicious games: {len(suspicious_games)}")
    print(f"Filtered games: {len(filtered_games)}")
    print(f"Expected regular season: 272")
    print(f"Match expected: {'✅ YES' if len(filtered_games) == 272 else '❌ NO'}")