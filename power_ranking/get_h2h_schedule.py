
import csv
import logging
import os
from api.espn_client import ESPNClient

def get_season_schedule(season: int):
    """
    Fetches the schedule for an entire NFL season.

    Args:
        season: The year of the season to fetch (e.g., 2025).

    Returns:
        A list of game data dictionaries.
    """
    client = ESPNClient()
    all_games = []
    for week in range(1, 19):  # Weeks 1-18
        try:
            logging.info(f"Fetching schedule for week {week} of the {season} season...")
            scoreboard = client.get_scoreboard(week=week, season=season)
            all_games.extend(scoreboard.get('events', []))
        except Exception as e:
            logging.error(f"Failed to fetch schedule for week {week}: {e}")
    return all_games

def save_schedule_to_csv(schedule: list, filename: str):
    """
    Saves the H2H schedule to a CSV file.

    Args:
        schedule: A list of game data dictionaries.
        filename: The name of the CSV file to save.
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['week', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for game in schedule:
            week = game.get('week', {}).get('number')
            competition = game.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            if len(competitors) == 2:
                home_team = None
                away_team = None
                for competitor in competitors:
                    if competitor.get('homeAway') == 'home':
                        home_team = competitor.get('team')
                    else:
                        away_team = competitor.get('team')

                if home_team and away_team:
                    writer.writerow({
                        'week': week,
                        'home_team_id': home_team.get('id'),
                        'home_team_name': home_team.get('displayName'),
                        'away_team_id': away_team.get('id'),
                        'away_team_name': away_team.get('displayName')
                    })

def main():
    """Main function to fetch and save the NFL schedule."""
    logging.basicConfig(level=logging.INFO)
    
    season = 2025  # 2025-2026 season
    schedule = get_season_schedule(season)
    
    if schedule:
        output_filename = f"output/h2h/h2h_schedule_{season}_{season+1}.csv"
        save_schedule_to_csv(schedule, output_filename)
        logging.info(f"Schedule saved to {output_filename}")
    else:
        logging.warning("No schedule data fetched.")

if __name__ == "__main__":
    main()
