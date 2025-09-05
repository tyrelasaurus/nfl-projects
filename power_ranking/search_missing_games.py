#!/usr/bin/env python3
"""
Search for the specific missing games using targeted API calls
"""
import requests
import json
import time
from datetime import datetime, timedelta

def search_missing_games():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    base_url = "https://site.api.espn.com/apis/site/v2"
    
    # Missing matchups we need to find
    missing_games = {
        5: [
            ("Detroit Lions", "Philadelphia Eagles"),  # One of these pairs should play each other
            ("Los Angeles Chargers", "Tennessee Titans")
        ],
        6: [
            ("Kansas City Chiefs", "Miami Dolphins"),  # Bye week teams should have games somewhere
            ("Los Angeles Rams", "Minnesota Vikings") 
        ],
        7: [
            ("Chicago Bears", "Dallas Cowboys")  # Missing 1 game
        ],
        9: [
            ("Pittsburgh Steelers", "San Francisco 49ers")  # Missing 1 game
        ],
        10: [
            ("Cleveland Browns", "Green Bay Packers"),
            ("Las Vegas Raiders", "Seattle Seahawks")
        ],
        11: [
            ("Arizona Cardinals", "Carolina Panthers"),
            ("New York Giants", "Tampa Bay Buccaneers")
        ],
        12: [
            ("Atlanta Falcons", "New Orleans Saints"),  # NFC South rivalry
            ("Buffalo Bills", "Cincinnati Bengals"),
            ("Jacksonville Jaguars", "New York Jets")
        ],
        14: [
            ("Baltimore Ravens", "Houston Texans"),
            ("Denver Broncos", "Indianapolis Colts"),
            ("New England Patriots", "Washington Commanders")
        ]
    }
    
    # Week-specific date ranges to search more thoroughly
    extended_search_dates = {
        5: [
            "20241003", "20241004", "20241005", "20241006", "20241007", "20241008"  # Thu-Tue Week 5
        ],
        6: [
            "20241010", "20241011", "20241012", "20241013", "20241014", "20241015"  # Thu-Tue Week 6
        ],
        7: [
            "20241017", "20241018", "20241019", "20241020", "20241021", "20241022"  # Thu-Tue Week 7
        ],
        9: [
            "20241031", "20241101", "20241102", "20241103", "20241104", "20241105"  # Thu-Tue Week 9
        ],
        10: [
            "20241107", "20241108", "20241109", "20241110", "20241111", "20241112"  # Thu-Tue Week 10
        ],
        11: [
            "20241114", "20241115", "20241116", "20241117", "20241118", "20241119"  # Thu-Tue Week 11
        ],
        12: [
            "20241121", "20241122", "20241123", "20241124", "20241125", "20241126",  # Thu-Tue Week 12 (Thanksgiving)
            "20241127", "20241128", "20241129"  # Extended for Thanksgiving week
        ],
        14: [
            "20241205", "20241206", "20241207", "20241208", "20241209", "20241210"  # Thu-Tue Week 14
        ]
    }
    
    found_games = []
    
    print("SEARCHING FOR MISSING GAMES")
    print("=" * 80)
    
    for week, dates in extended_search_dates.items():
        print(f"\nSEARCHING WEEK {week}:")
        week_games = []
        
        for date_str in dates:
            try:
                print(f"  Checking {date_str}...")
                url = f"{base_url}/sports/football/nfl/scoreboard"
                response = session.get(url, params={'dates': date_str}, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        # Check if this is a 2024 regular season game
                        season = event.get('season', {})
                        if season.get('year') == 2024 and season.get('type') == 2:
                            
                            # Get team names
                            competitors = event.get('competitions', [{}])[0].get('competitors', [])
                            if len(competitors) >= 2:
                                team1 = competitors[0].get('team', {}).get('displayName', '')
                                team2 = competitors[1].get('team', {}).get('displayName', '')
                                game_id = event.get('id')
                                
                                # Check if this involves teams that were missing from this week
                                missing_teams_this_week = set()
                                for pair in missing_games.get(week, []):
                                    missing_teams_this_week.update(pair)
                                
                                if team1 in missing_teams_this_week or team2 in missing_teams_this_week:
                                    print(f"    ✅ FOUND: {team1} vs {team2} (ID: {game_id})")
                                    week_games.append({
                                        'game': event,
                                        'matchup': f"{team1} vs {team2}",
                                        'date': date_str,
                                        'id': game_id
                                    })
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Error checking {date_str}: {e}")
                continue
        
        found_games.extend(week_games)
        print(f"  Week {week} found: {len(week_games)} additional games")
    
    # Try alternative approaches for persistent missing games
    print(f"\n" + "=" * 60)
    print("TRYING ALTERNATIVE SEARCH METHODS")
    print("=" * 60)
    
    # Method 1: Try broader date ranges
    print("\n1. Trying broader date ranges...")
    broad_ranges = [
        ("20241001-20241008", "Week 5 range"),
        ("20241008-20241015", "Week 6 range"), 
        ("20241015-20241022", "Week 7 range"),
        ("20241029-20241105", "Week 9 range"),
        ("20241105-20241112", "Week 10 range"),
        ("20241112-20241119", "Week 11 range"),
        ("20241119-20241129", "Week 12 range"),
        ("20241203-20241210", "Week 14 range")
    ]
    
    for date_range, description in broad_ranges:
        try:
            print(f"  Checking {description} ({date_range})...")
            url = f"{base_url}/sports/football/nfl/scoreboard"
            response = session.get(url, params={'dates': date_range}, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            events = data.get('events', [])
            completed_games = [e for e in events if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL']
            
            print(f"    Found {len(completed_games)} completed games in {description}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"    Error with {description}: {e}")
    
    # Method 2: Try team-specific searches for bye weeks
    print(f"\n2. Checking for bye week inconsistencies...")
    
    # Some of these "missing" games might actually be bye weeks
    # Let's verify by checking if these teams actually had 17 games total
    
    print(f"\nSUMMARY:")
    print(f"Additional games found through extended search: {len(found_games)}")
    
    if found_games:
        print(f"\nGames found:")
        for i, game_info in enumerate(found_games, 1):
            print(f"  {i}. {game_info['matchup']} on {game_info['date']} (ID: {game_info['id']})")
    
    return found_games

def verify_bye_weeks():
    """Verify that the missing games are actually bye weeks, not data collection issues"""
    print(f"\n" + "=" * 60)
    print("VERIFYING BYE WEEKS")
    print("=" * 60)
    
    # In NFL, each team plays 17 games, so they have 1 bye week
    # Let's verify our dataset has the right number of games per team
    
    with open('/Users/tyrelshaw/Projects/power_ranking/output/filtered_272_games.json', 'r') as f:
        data = json.load(f)
    
    games = data['games']
    
    # Count games per team
    team_game_counts = {}
    
    for game in games:
        competitors = game.get('competitions', [{}])[0].get('competitors', [])
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if team_name:
                if team_name not in team_game_counts:
                    team_game_counts[team_name] = 0
                team_game_counts[team_name] += 1
    
    print(f"Games per team in our dataset:")
    teams_with_17_games = 0
    teams_with_fewer_games = []
    
    for team, count in sorted(team_game_counts.items()):
        status = "✓" if count == 17 else f"⚠ (-{17-count})"
        print(f"  {team:<25}: {count:2d} games {status}")
        
        if count == 17:
            teams_with_17_games += 1
        else:
            teams_with_fewer_games.append((team, count))
    
    print(f"\nSummary:")
    print(f"  Teams with 17 games: {teams_with_17_games}/32")
    print(f"  Teams with fewer games: {len(teams_with_fewer_games)}")
    
    if teams_with_fewer_games:
        print(f"\nTeams missing games:")
        for team, count in teams_with_fewer_games:
            print(f"  {team}: {count} games (missing {17-count})")
    
    # Calculate total expected games: 32 teams × 17 games ÷ 2 = 272 games
    total_games_in_dataset = len(games)
    expected_total = 32 * 17 // 2
    
    print(f"\nDataset validation:")
    print(f"  Games in dataset: {total_games_in_dataset}")
    print(f"  Expected games: {expected_total}")
    print(f"  Match: {'✅ YES' if total_games_in_dataset == expected_total else '❌ NO'}")
    
    return team_game_counts

if __name__ == "__main__":
    found_games = search_missing_games()
    team_counts = verify_bye_weeks()
    
    print(f"\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    
    if len([count for count in team_counts.values() if count == 17]) == 32:
        print("✅ ALL TEAMS HAVE 17 GAMES - No games are actually missing!")
        print("The 'missing' games were actually bye weeks or data collection artifacts.")
    else:
        print("❌ Some teams have fewer than 17 games - genuine data collection issue.")
        print("Additional data collection needed.")