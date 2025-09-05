#!/usr/bin/env python3
"""
Debug script to find which weeks are missing data
"""
import sys
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')

from api.espn_client import ESPNClient

def debug_missing_weeks():
    client = ESPNClient()
    
    print("Fetching 2024 season data...")
    season_data = client.get_last_season_final_rankings()
    events = season_data.get('events', [])
    
    print(f"Total games found: {len(events)}")
    
    # Count games per week
    week_counts = {}
    for event in events:
        week_num = event.get('week_number', 0)
        week_counts[week_num] = week_counts.get(week_num, 0) + 1
    
    print("\nGames per week:")
    for week in range(1, 19):
        count = week_counts.get(week, 0)
        status = "✅" if count >= 12 else "❌"  # Expect ~16 games per week (32 teams / 2)
        if week == 6:  # Bye week - fewer games expected
            status = "✅" if count >= 10 else "❌"
        print(f"Week {week:2d}: {count:2d} games {status}")

if __name__ == "__main__":
    debug_missing_weeks()