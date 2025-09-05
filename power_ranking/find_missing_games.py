#!/usr/bin/env python3
"""
Analyze the missing 11 games to identify specific matchups that weren't collected
"""
import json
import sys
from collections import defaultdict
from datetime import datetime

def analyze_missing_games():
    print("ANALYZING MISSING GAMES")
    print("=" * 80)
    
    # Load our filtered dataset
    with open('/Users/tyrelshaw/Projects/power_ranking/output/filtered_272_games.json', 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"Current games in filtered dataset: {len(games)}")
    print(f"Expected games: 272")
    print(f"Missing games: {272 - len(games)}")
    print()
    
    # Analyze games by week
    weeks_analysis = {}
    for week in range(1, 19):
        week_games = [g for g in games if g.get('week', {}).get('number', g.get('week_number')) == week]
        weeks_analysis[week] = {
            'count': len(week_games),
            'expected': 16 if week != 18 else 16,  # All weeks should have 16 games
            'missing': max(0, 16 - len(week_games)),
            'games': week_games
        }
    
    print("GAMES PER WEEK ANALYSIS:")
    total_missing = 0
    problem_weeks = []
    
    for week in range(1, 19):
        analysis = weeks_analysis[week]
        missing = analysis['missing']
        total_missing += missing
        
        status = "✓" if missing == 0 else f"⚠ (-{missing})"
        print(f"  Week {week:2d}: {analysis['count']:2d}/16 games {status}")
        
        if missing > 0:
            problem_weeks.append(week)
    
    print(f"\nTotal missing across all weeks: {total_missing}")
    print(f"Problem weeks: {problem_weeks}")
    
    # For each problem week, let's see what teams are missing
    print(f"\nDETAILED ANALYSIS OF PROBLEM WEEKS:")
    print("=" * 60)
    
    # Get all NFL teams (32 teams)
    all_teams = set()
    for game in games:
        competitors = game.get('competitions', [{}])[0].get('competitors', [])
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if team_name:
                all_teams.add(team_name)
    
    all_teams = sorted(list(all_teams))
    print(f"Total teams found in dataset: {len(all_teams)}")
    
    for week in problem_weeks:
        print(f"\nWEEK {week} ANALYSIS:")
        week_games = weeks_analysis[week]['games']
        print(f"Found {len(week_games)} games (missing {weeks_analysis[week]['missing']})")
        
        # Track which teams played this week
        teams_played = set()
        matchups = []
        
        for game in week_games:
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            if len(competitors) >= 2:
                team1 = competitors[0].get('team', {}).get('displayName', '')
                team2 = competitors[1].get('team', {}).get('displayName', '')
                if team1 and team2:
                    teams_played.add(team1)
                    teams_played.add(team2)
                    home_team = next((t for i, t in enumerate([team1, team2]) 
                                    if competitors[i].get('homeAway') == 'home'), team2)
                    away_team = next((t for i, t in enumerate([team1, team2]) 
                                    if competitors[i].get('homeAway') == 'away'), team1)
                    matchups.append(f"{away_team} @ {home_team}")
        
        teams_missing = sorted(set(all_teams) - teams_played)
        
        print(f"Teams that played ({len(teams_played)}):")
        for i, team in enumerate(sorted(teams_played), 1):
            print(f"  {i:2d}. {team}")
        
        if teams_missing:
            print(f"\nTeams that didn't play ({len(teams_missing)}):")
            for i, team in enumerate(teams_missing, 1):
                print(f"  {i:2d}. {team}")
        
        print(f"\nMatchups found:")
        for i, matchup in enumerate(matchups, 1):
            print(f"  {i:2d}. {matchup}")
    
    return problem_weeks, weeks_analysis

def find_missing_games_in_original():
    """Check if missing games exist in the original 288-game dataset"""
    print(f"\n" + "=" * 80)
    print("CHECKING ORIGINAL 288-GAME DATASET FOR MISSING GAMES")
    print("=" * 80)
    
    # Load original dataset
    with open('/Users/tyrelshaw/Projects/power_ranking/output/debug_full_season_20250831_212411.json', 'r') as f:
        original_data = json.load(f)
    
    # Load filtered dataset  
    with open('/Users/tyrelshaw/Projects/power_ranking/output/filtered_272_games.json', 'r') as f:
        filtered_data = json.load(f)
    
    original_games = original_data['games']
    filtered_games = filtered_data['games']
    
    # Get game IDs from filtered set
    filtered_ids = {g.get('id') for g in filtered_games}
    
    # Find 2024 regular season games that were filtered out
    missing_games = []
    for game in original_games:
        game_id = game.get('id')
        season = game.get('season', {})
        season_type = season.get('type')
        season_year = season.get('year')
        week_num = game.get('week', {}).get('number', game.get('week_number'))
        
        # If it's a 2024 regular season game but not in filtered set
        if (season_type == 2 and 
            season_year == 2024 and 
            isinstance(week_num, int) and 
            1 <= week_num <= 18 and
            game_id not in filtered_ids):
            
            missing_games.append(game)
    
    print(f"2024 regular season games in original: {sum(1 for g in original_games if g.get('season', {}).get('year') == 2024 and g.get('season', {}).get('type') == 2)}")
    print(f"2024 regular season games in filtered: {len(filtered_games)}")
    print(f"Games that were filtered out: {len(missing_games)}")
    
    if missing_games:
        print(f"\nGAMES THAT WERE INCORRECTLY FILTERED OUT:")
        for i, game in enumerate(missing_games, 1):
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            team_names = []
            for comp in competitors:
                team_name = comp.get('team', {}).get('displayName', '')
                team_names.append(team_name)
            
            matchup = ' vs '.join(team_names) if len(team_names) >= 2 else 'Unknown matchup'
            week = game.get('week', {}).get('number', game.get('week_number'))
            game_id = game.get('id')
            date = game.get('date', 'Unknown')
            
            print(f"  {i:2d}. Week {week}: {matchup}")
            print(f"      ID: {game_id}, Date: {date}")
    
    # Check if any games are truly missing from both datasets
    print(f"\nCHECKING FOR GAMES MISSING FROM ORIGINAL DATASET...")
    
    # We expect 272 games total (32 teams × 17 games ÷ 2)
    # Let's verify this is mathematically correct
    
    return missing_games

def suggest_data_collection_improvements():
    """Suggest improvements to data collection to capture missing games"""
    print(f"\n" + "=" * 80)
    print("DATA COLLECTION IMPROVEMENT SUGGESTIONS")
    print("=" * 80)
    
    suggestions = [
        "1. Try additional date ranges for problem weeks:",
        "   - Include Tuesday/Wednesday games (London/international games)",
        "   - Check Saturday games during certain weeks",
        "   - Verify Thursday Night Football games",
        "",
        "2. Use ESPN's games endpoint with different parameters:",
        "   - Try /games endpoint instead of /scoreboard",
        "   - Use specific date ranges: YYYYMMDD-YYYYMMDD",
        "   - Check if any games are marked as postponed/rescheduled",
        "",
        "3. Cross-reference with other data sources:",
        "   - NFL.com official schedule",
        "   - Pro Football Reference",
        "   - ESPN's team schedules endpoint",
        "",
        "4. Check for data quality issues:",
        "   - Games with incorrect status (not marked as final)",
        "   - Games with missing competitor data",
        "   - Duplicate games with different IDs"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

if __name__ == "__main__":
    problem_weeks, weeks_analysis = analyze_missing_games()
    missing_from_original = find_missing_games_in_original()
    suggest_data_collection_improvements()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Current dataset: {261 if len(sys.argv) == 1 else 'Unknown'} games")
    print(f"Expected: 272 games") 
    print(f"Missing: {272 - 261} games")
    print(f"Problem weeks: {problem_weeks}")
    print(f"Games filtered out incorrectly: {len(missing_from_original) if 'missing_from_original' in locals() else 'Unknown'}")