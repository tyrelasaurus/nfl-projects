#!/usr/bin/env python3
"""
Comprehensive debug script for full NFL 2024/25 season data collection
Tests all available methods to ensure we capture all 272 regular season games
"""
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.espn_client import ESPNClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveSeasonDebugger:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Track all games found across methods
        self.all_games_found = {}  # game_id -> game_data
        self.method_results = {}
        
        # NFL 2024 season dates for each week
        self.season_dates = [
            ('20240908', 1),   # Week 1
            ('20240915', 2),   # Week 2  
            ('20240922', 3),   # Week 3
            ('20240929', 4),   # Week 4
            ('20241006', 5),   # Week 5
            ('20241013', 6),   # Week 6
            ('20241020', 7),   # Week 7
            ('20241027', 8),   # Week 8
            ('20241103', 9),   # Week 9
            ('20241110', 10),  # Week 10
            ('20241117', 11),  # Week 11
            ('20241124', 12),  # Week 12 (Thanksgiving)
            ('20241201', 13),  # Week 13
            ('20241208', 14),  # Week 14
            ('20241215', 15),  # Week 15
            ('20241222', 16),  # Week 16
            ('20241229', 17),  # Week 17
            ('20250105', 18),  # Week 18
        ]
        
        # Extended dates including Thursday/Monday games
        self.extended_dates = []
        for week_start, week_num in self.season_dates:
            # Add Thursday (3 days before), Sunday, and Monday games
            base_date = datetime.strptime(week_start, '%Y%m%d')
            thursday = base_date - timedelta(days=3)
            monday = base_date + timedelta(days=1)
            
            self.extended_dates.extend([
                (thursday.strftime('%Y%m%d'), week_num),
                (week_start, week_num),
                (monday.strftime('%Y%m%d'), week_num)
            ])
    
    def test_method_1_espn_client(self) -> Dict:
        """Test the existing ESPNClient comprehensive approach"""
        logger.info("=" * 60)
        logger.info("METHOD 1: ESPNClient get_last_season_final_rankings()")
        logger.info("=" * 60)
        
        try:
            espn_client = ESPNClient()
            season_data = espn_client.get_last_season_final_rankings()
            
            games = season_data.get('events', [])
            completed_games = [g for g in games if g.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL']
            
            # Add games to our master list
            games_added = 0
            for game in completed_games:
                game_id = game.get('id')
                if game_id and game_id not in self.all_games_found:
                    self.all_games_found[game_id] = game
                    games_added += 1
            
            result = {
                'method': 'ESPNClient',
                'games_found': len(completed_games),
                'games_added': games_added,
                'success': len(completed_games) > 200
            }
            
            logger.info(f"ESPNClient found {len(completed_games)} games, added {games_added} new games")
            return result
            
        except Exception as e:
            logger.error(f"ESPNClient method failed: {e}")
            return {'method': 'ESPNClient', 'games_found': 0, 'games_added': 0, 'success': False, 'error': str(e)}
    
    def test_method_2_date_based(self) -> Dict:
        """Test comprehensive date-based approach"""
        logger.info("=" * 60)
        logger.info("METHOD 2: Date-based API calls (all 18 weeks)")
        logger.info("=" * 60)
        
        try:
            base_url = "https://site.api.espn.com/apis/site/v2"
            total_games = 0
            games_added = 0
            successful_weeks = 0
            
            for date_str, week_num in self.season_dates:
                try:
                    logger.info(f"Fetching week {week_num} ({date_str})...")
                    url = f"{base_url}/sports/football/nfl/scoreboard"
                    response = self.session.get(url, params={'dates': date_str}, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    events = data.get('events', [])
                    
                    completed_games = [
                        event for event in events 
                        if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL'
                    ]
                    
                    if completed_games:
                        for game in completed_games:
                            game_id = game.get('id')
                            if game_id and game_id not in self.all_games_found:
                                game['week_number'] = week_num
                                self.all_games_found[game_id] = game
                                games_added += 1
                        
                        total_games += len(completed_games)
                        successful_weeks += 1
                        logger.info(f"Week {week_num}: Found {len(completed_games)} games")
                    else:
                        logger.warning(f"Week {week_num}: No games found")
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Week {week_num} failed: {e}")
                    continue
            
            result = {
                'method': 'Date-based',
                'games_found': total_games,
                'games_added': games_added,
                'successful_weeks': successful_weeks,
                'success': total_games > 250
            }
            
            logger.info(f"Date-based found {total_games} games across {successful_weeks} weeks, added {games_added} new")
            return result
            
        except Exception as e:
            logger.error(f"Date-based method failed: {e}")
            return {'method': 'Date-based', 'games_found': 0, 'games_added': 0, 'success': False, 'error': str(e)}
    
    def test_method_3_extended_dates(self) -> Dict:
        """Test extended date approach with Thu/Sun/Mon games"""
        logger.info("=" * 60)
        logger.info("METHOD 3: Extended dates (Thu/Sun/Mon for each week)")
        logger.info("=" * 60)
        
        try:
            base_url = "https://site.api.espn.com/apis/site/v2"
            total_games = 0
            games_added = 0
            
            for date_str, week_num in self.extended_dates:
                try:
                    url = f"{base_url}/sports/football/nfl/scoreboard"
                    response = self.session.get(url, params={'dates': date_str}, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    events = data.get('events', [])
                    
                    for event in events:
                        if event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                            game_id = event.get('id')
                            if game_id and game_id not in self.all_games_found:
                                event['week_number'] = week_num
                                self.all_games_found[game_id] = event
                                games_added += 1
                                total_games += 1
                    
                    time.sleep(0.2)  # Light rate limiting
                    
                except Exception as e:
                    logger.debug(f"Extended date {date_str} failed: {e}")
                    continue
            
            result = {
                'method': 'Extended dates',
                'games_found': total_games,
                'games_added': games_added,
                'success': games_added > 10
            }
            
            logger.info(f"Extended dates added {games_added} new games")
            return result
            
        except Exception as e:
            logger.error(f"Extended dates method failed: {e}")
            return {'method': 'Extended dates', 'games_found': 0, 'games_added': 0, 'success': False, 'error': str(e)}
    
    def test_method_4_core_api(self) -> Dict:
        """Test ESPN Core API approach"""
        logger.info("=" * 60)
        logger.info("METHOD 4: ESPN Core API")
        logger.info("=" * 60)
        
        try:
            core_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/events"
            response = self.session.get(f"{core_url}?limit=300", timeout=30)
            response.raise_for_status()
            
            core_data = response.json()
            events_refs = core_data.get('items', [])
            
            total_games = 0
            games_added = 0
            
            logger.info(f"Core API returned {len(events_refs)} event references")
            
            for i, event_ref in enumerate(events_refs[:280]):  # Limit to avoid timeout
                try:
                    event_url = event_ref.get('$ref')
                    if not event_url:
                        continue
                    
                    event_response = self.session.get(event_url, timeout=30)
                    event_response.raise_for_status()
                    event_data = event_response.json()
                    
                    if event_data.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL':
                        game_id = event_data.get('id')
                        week_num = event_data.get('week', {}).get('number', 1)
                        
                        if game_id and game_id not in self.all_games_found and 1 <= week_num <= 18:
                            event_data['week_number'] = week_num
                            self.all_games_found[game_id] = event_data
                            games_added += 1
                        
                        total_games += 1
                    
                    if (i + 1) % 20 == 0:
                        logger.info(f"Processed {i + 1} events, found {games_added} new games")
                        time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch event {i}: {e}")
                    continue
            
            result = {
                'method': 'Core API',
                'games_found': total_games,
                'games_added': games_added,
                'success': games_added > 10
            }
            
            logger.info(f"Core API found {total_games} games, added {games_added} new")
            return result
            
        except Exception as e:
            logger.error(f"Core API method failed: {e}")
            return {'method': 'Core API', 'games_found': 0, 'games_added': 0, 'success': False, 'error': str(e)}
    
    def test_method_5_year_range(self) -> Dict:
        """Test year/date range approach"""
        logger.info("=" * 60)
        logger.info("METHOD 5: Year/Date Range approach")
        logger.info("=" * 60)
        
        try:
            base_url = "https://site.api.espn.com/apis/site/v2"
            
            test_params = [
                {'dates': '2024', 'seasontype': '2', 'limit': '1000'},
                {'dates': '20240901-20250107', 'limit': '1000'},
                {'year': '2024', 'seasontype': '2', 'limit': '1000'},
            ]
            
            best_result = {'method': 'Year range', 'games_found': 0, 'games_added': 0, 'success': False}
            
            for i, params in enumerate(test_params):
                try:
                    logger.info(f"Testing year range approach {i+1}: {params}")
                    
                    url = f"{base_url}/sports/football/nfl/scoreboard"
                    response = self.session.get(url, params=params, timeout=60)
                    response.raise_for_status()
                    
                    data = response.json()
                    events = data.get('events', [])
                    
                    completed_games = [
                        event for event in events 
                        if (event.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL' and
                            event.get('season', {}).get('type', 2) == 2)
                    ]
                    
                    games_added = 0
                    for game in completed_games:
                        game_id = game.get('id')
                        week_num = game.get('week', {}).get('number', 1)
                        
                        if game_id and game_id not in self.all_games_found and 1 <= week_num <= 18:
                            game['week_number'] = week_num
                            self.all_games_found[game_id] = game
                            games_added += 1
                    
                    logger.info(f"Approach {i+1}: Found {len(completed_games)} games, added {games_added} new")
                    
                    if len(completed_games) > best_result['games_found']:
                        best_result = {
                            'method': f'Year range {i+1}',
                            'games_found': len(completed_games),
                            'games_added': games_added,
                            'success': len(completed_games) > 200,
                            'params': params
                        }
                    
                    time.sleep(2)  # Rate limiting between attempts
                    
                except Exception as e:
                    logger.warning(f"Year range approach {i+1} failed: {e}")
                    continue
            
            return best_result
            
        except Exception as e:
            logger.error(f"Year range method failed: {e}")
            return {'method': 'Year range', 'games_found': 0, 'games_added': 0, 'success': False, 'error': str(e)}
    
    def analyze_game_coverage(self) -> Dict:
        """Analyze the coverage of games found"""
        logger.info("=" * 60)
        logger.info("ANALYZING GAME COVERAGE")
        logger.info("=" * 60)
        
        week_coverage = {}
        team_games = {}
        
        for game_id, game in self.all_games_found.items():
            week_num = game.get('week_number', 1)
            
            # Track games per week
            if week_num not in week_coverage:
                week_coverage[week_num] = 0
            week_coverage[week_num] += 1
            
            # Track games per team
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            for comp in competitors:
                team_id = comp.get('team', {}).get('id')
                if team_id:
                    if team_id not in team_games:
                        team_games[team_id] = 0
                    team_games[team_id] += 1
        
        # Analysis
        total_games = len(self.all_games_found)
        expected_games = 272  # 32 teams * 17 games / 2
        coverage_pct = (total_games / expected_games) * 100
        
        missing_weeks = [w for w in range(1, 19) if w not in week_coverage]
        incomplete_weeks = [w for w, count in week_coverage.items() if count < 15]  # Expect ~16 games per week
        
        logger.info(f"Total unique games found: {total_games}")
        logger.info(f"Expected games (272): Coverage = {coverage_pct:.1f}%")
        logger.info(f"Weeks with data: {len(week_coverage)}/18")
        logger.info(f"Missing weeks: {missing_weeks}")
        logger.info(f"Weeks with <15 games: {incomplete_weeks}")
        
        # Show week-by-week breakdown
        logger.info("\nWeek-by-week game counts:")
        for week in range(1, 19):
            count = week_coverage.get(week, 0)
            status = "✓" if count >= 15 else "⚠" if count > 0 else "✗"
            logger.info(f"  Week {week:2d}: {count:2d} games {status}")
        
        # Team coverage analysis
        teams_with_17_games = sum(1 for count in team_games.values() if count == 17)
        teams_with_missing_games = sum(1 for count in team_games.values() if count < 17)
        
        logger.info(f"\nTeam coverage:")
        logger.info(f"  Teams with all 17 games: {teams_with_17_games}/32")
        logger.info(f"  Teams with missing games: {teams_with_missing_games}")
        
        return {
            'total_games': total_games,
            'coverage_percentage': coverage_pct,
            'weeks_covered': len(week_coverage),
            'missing_weeks': missing_weeks,
            'incomplete_weeks': incomplete_weeks,
            'teams_complete': teams_with_17_games,
            'teams_incomplete': teams_with_missing_games
        }
    
    def export_results(self) -> str:
        """Export all found games to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/tyrelshaw/Projects/power_ranking/output/debug_full_season_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        export_data = {
            'timestamp': timestamp,
            'total_games': len(self.all_games_found),
            'method_results': self.method_results,
            'games': list(self.all_games_found.values())
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to: {filename}")
        return filename
    
    def run_comprehensive_test(self) -> Dict:
        """Run all test methods and provide comprehensive analysis"""
        logger.info("STARTING COMPREHENSIVE NFL 2024/25 SEASON DATA COLLECTION DEBUG")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run all test methods
        self.method_results['espn_client'] = self.test_method_1_espn_client()
        self.method_results['date_based'] = self.test_method_2_date_based()
        self.method_results['extended_dates'] = self.test_method_3_extended_dates()
        self.method_results['core_api'] = self.test_method_4_core_api()
        self.method_results['year_range'] = self.test_method_5_year_range()
        
        # Analyze coverage
        coverage_analysis = self.analyze_game_coverage()
        
        # Export results
        export_file = self.export_results()
        
        # Generate summary
        total_runtime = datetime.now() - start_time
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE DEBUG SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total runtime: {total_runtime}")
        logger.info(f"Total unique games found: {len(self.all_games_found)}")
        logger.info(f"Target: 272 games (100% regular season)")
        logger.info(f"Coverage: {coverage_analysis['coverage_percentage']:.1f}%")
        
        logger.info("\nMethod Performance:")
        for method, results in self.method_results.items():
            success = "✓" if results.get('success', False) else "✗"
            games_added = results.get('games_added', 0)
            logger.info(f"  {method:15s}: +{games_added:3d} games {success}")
        
        # Determine if we have sufficient data
        sufficient_data = coverage_analysis['coverage_percentage'] >= 95
        
        final_summary = {
            'success': sufficient_data,
            'total_games': len(self.all_games_found),
            'coverage_percentage': coverage_analysis['coverage_percentage'],
            'method_results': self.method_results,
            'coverage_analysis': coverage_analysis,
            'export_file': export_file,
            'runtime_seconds': total_runtime.total_seconds()
        }
        
        if sufficient_data:
            logger.info("✅ SUCCESS: Sufficient data collected for full power ranking analysis")
        else:
            logger.info("❌ WARNING: Insufficient data - may need additional collection strategies")
        
        return final_summary

def main():
    debugger = ComprehensiveSeasonDebugger()
    results = debugger.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Data Collection: {'SUCCESS' if results['success'] else 'PARTIAL'}")
    print(f"Games Found: {results['total_games']}/272 ({results['coverage_percentage']:.1f}%)")
    print(f"Export File: {results['export_file']}")
    print(f"Runtime: {results['runtime_seconds']:.1f} seconds")
    
    return 0 if results['success'] else 1

if __name__ == "__main__":
    sys.exit(main())