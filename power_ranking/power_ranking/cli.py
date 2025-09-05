#!/usr/bin/env python3
import argparse
import logging
import yaml
import sys
import os
from power_ranking.api.espn_client import ESPNClient
from power_ranking.models.power_rankings import PowerRankModel
from power_ranking.export.csv_exporter import CSVExporter
from power_ranking.export.data_exporter import DataExporter


def load_config(config_path: str) -> dict:
    """Load YAML config with sensible fallbacks.

    Tries the provided path first; if missing, falls back to the
    package default config bundled in power_ranking/.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load config from {config_path}: {e}")
        # Fallback to package config
        pkg_dir = os.path.dirname(__file__)
        pkg_default = os.path.join(pkg_dir, 'config.yaml')
        try:
            with open(pkg_default, 'r') as f:
                logging.info(f"Using package default config: {pkg_default}")
                return yaml.safe_load(f)
        except Exception as e2:
            # Try project-level package directory (one level up)
            alt_default = os.path.join(os.path.dirname(pkg_dir), 'config.yaml')
            try:
                with open(alt_default, 'r') as f:
                    logging.info(f"Using alternate default config: {alt_default}")
                    return yaml.safe_load(f)
            except Exception as e3:
                logging.error(f"Failed to load fallback config from {pkg_default} and {alt_default}: {e3}")
                sys.exit(1)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description="NFL Power Rankings Calculator")
    parser.add_argument('--week', type=int, help='NFL week number to process')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--output', help='Directory to save CSV (overrides config)')
    parser.add_argument('--dry-run', action='store_true', help='Run without writing CSV')
    parser.add_argument('--debug', action='store_true', help='Show detailed team metrics for analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        espn_client = ESPNClient(config.get('espn_api_base'))
        power_model = PowerRankModel(config.get('weights'))
        
        output_dir = args.output or config.get('output_dir', './output')
        csv_exporter = CSVExporter(output_dir)
        data_exporter = DataExporter(output_dir)
        
        # Determine week to process
        week = args.week
        if not week:
            week = espn_client.get_current_week()
            logger.info(f"Using current week: {week}")
        
        logger.info(f"Processing NFL power rankings for week {week}")
        
        # Fetch data
        logger.info("Fetching teams data...")
        teams_data = espn_client.get_teams()
        
        logger.info("Fetching scoreboard data...")
        scoreboard_data = espn_client.get_scoreboard(week=week)
        
        # Check if current season has games, and optionally merge with last season
        use_last_season = False
        last_season_data = None
        try:
            last_season_data = espn_client.get_last_season_final_rankings()
        except Exception:
            last_season_data = None

        if not espn_client.has_current_season_games(week=week):
            logger.info("No completed games found in current season, using last season's final standings...")
            scoreboard_data = last_season_data or scoreboard_data
            use_last_season = True
            week = "Initial (based on 2024 season)"
        else:
            # Merge current season week(s) with prior season to enable last-17-games logic
            if last_season_data and last_season_data.get('events'):
                merged_ids = set()
                merged_events = []
                for ev in (scoreboard_data.get('events') or []):
                    eid = str(ev.get('id'))
                    if eid not in merged_ids:
                        merged_events.append(ev)
                        merged_ids.add(eid)
                for ev in (last_season_data.get('events') or []):
                    eid = str(ev.get('id'))
                    if eid not in merged_ids:
                        merged_events.append(ev)
                        merged_ids.add(eid)
                scoreboard_data = {
                    'events': merged_events,
                    'week': scoreboard_data.get('week') or {'number': week},
                    'season': scoreboard_data.get('season') or last_season_data.get('season')
                }
        
        # Compute power rankings
        logger.info("Computing power rankings...")
        # Compute power rankings constrained to each team's last 17 games across seasons
        rankings, computation_data = power_model.compute(scoreboard_data, teams_data, last_n_games=17)
        
        if not rankings:
            logger.warning("No rankings computed - unable to fetch game data")
            return
        
        logger.info(f"Computed rankings for {len(rankings)} teams")
        
        # Display top 10 rankings
        print(f"\nNFL Power Rankings - Week {week}")
        print("=" * 50)
        for i, (team_id, team_name, score) in enumerate(rankings[:10], 1):
            print(f"{i:2d}. {team_name:<25} {score:6.3f}")
        
        # Debug mode: show detailed metrics for specific teams
        if args.debug and computation_data:
            print(f"\n" + "="*80)
            print("DETAILED TEAM ANALYSIS")
            print("="*80)
            
            # Focus on the teams mentioned by user
            focus_teams = ['Kansas City Chiefs', 'Detroit Lions', 'Philadelphia Eagles', 'Buffalo Bills',
                          'Denver Broncos', 'Tampa Bay Buccaneers', 'Houston Texans']
            
            for team_id, team_name, power_score in rankings:
                if team_name in focus_teams:
                    season_stats = computation_data['season_stats'].get(team_id, {})
                    rolling_stats = computation_data['rolling_stats'].get(team_id, {})
                    sos = computation_data['sos_scores'].get(team_id, 0)
                    
                    print(f"\n{team_name} (Rank: {rankings.index((team_id, team_name, power_score)) + 1})")
                    print(f"  Power Score: {power_score:.3f}")
                    print(f"  Season Stats:")
                    print(f"    Games: {season_stats.get('games_played', 0)}")
                    print(f"    Record: {season_stats.get('wins', 0)}-{season_stats.get('losses', 0)} ({season_stats.get('win_pct', 0):.3f})")
                    print(f"    Avg Margin: {season_stats.get('avg_margin', 0):.2f}")
                    print(f"    PPG: {season_stats.get('avg_points_for', 0):.1f}")
                    print(f"    Opp PPG: {season_stats.get('avg_points_against', 0):.1f}")
                    print(f"  Rolling 5-Week:")
                    print(f"    Games: {rolling_stats.get('games_played', 0)}")
                    print(f"    Avg Margin: {rolling_stats.get('avg_margin', 0):.2f}")
                    print(f"    Win Pct: {rolling_stats.get('win_pct', 0):.3f}")
                    print(f"  Strength of Schedule: {sos:.3f}")
        
        # Export data
        if not args.dry_run:
            # Use appropriate week value for filename
            export_week = "initial_adjusted" if use_last_season else week
            
            # Export power rankings CSV
            rankings_filepath = csv_exporter.export_rankings(rankings, export_week)
            print(f"\nRankings exported to: {rankings_filepath}")
            
            # Export comprehensive data for H2H modeling
            logger.info("Exporting comprehensive data for H2H modeling...")
            exported_files = data_exporter.export_comprehensive_data(
                scoreboard_data, teams_data, computation_data, export_week
            )
            
            print(f"\nComprehensive data exported:")
            for data_type, filepath in exported_files.items():
                print(f"  {data_type}: {filepath}")
                
        else:
            print("\nDry run - no files created")
            
    except Exception as e:
        logger.error(f"Failed to generate power rankings: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
