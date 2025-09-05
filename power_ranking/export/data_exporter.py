import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple


class DataExporter:
    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for organized data storage
        subdirs = ['data', 'analysis', 'h2h']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def export_comprehensive_data(self, scoreboard_data: Dict, teams_data: List[Dict], 
                                computation_data: Dict, week: Any) -> Dict[str, str]:
        """
        Export comprehensive data for H2H modeling and analysis
        Returns dict of exported file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        # 1. Export raw game data (JSON and CSV)
        game_data_json = self._export_game_data_json(scoreboard_data, week, timestamp)
        game_data_csv = self._export_game_data_csv(scoreboard_data, week, timestamp)
        exported_files.update({'games_json': game_data_json, 'games_csv': game_data_csv})
        
        # 2. Export team statistics
        team_stats_file = self._export_team_statistics(computation_data, week, timestamp)
        exported_files['team_stats'] = team_stats_file
        
        # 3. Export head-to-head matchup data
        h2h_file = self._export_h2h_data(scoreboard_data, week, timestamp)
        exported_files['h2h_matchups'] = h2h_file
        
        # 4. Export performance matrix
        performance_file = self._export_performance_matrix(computation_data, teams_data, week, timestamp)
        exported_files['performance_matrix'] = performance_file
        
        # 5. Export metadata
        metadata_file = self._export_metadata(scoreboard_data, computation_data, week, timestamp)
        exported_files['metadata'] = metadata_file
        
        return exported_files
    
    def _export_game_data_json(self, scoreboard_data: Dict, week: Any, timestamp: str) -> str:
        """Export raw game data as JSON"""
        filename = f"game_data_{week}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, 'data', filename)
        
        # Clean and structure the game data
        games = scoreboard_data.get('events', [])
        structured_games = []
        
        for game in games:
            game_info = {
                'id': game.get('id'),
                'week': game.get('week_number'),
                'date': game.get('date'),
                'status': game.get('status', {}).get('type', {}).get('name'),
                'competitions': []
            }
            
            for competition in game.get('competitions', []):
                comp_info = {
                    'id': competition.get('id'),
                    'home_advantage': competition.get('neutralSite', False),
                    'competitors': []
                }
                
                for competitor in competition.get('competitors', []):
                    comp_data = {
                        'team_id': competitor.get('team', {}).get('id'),
                        'team_name': competitor.get('team', {}).get('displayName'),
                        'team_abbreviation': competitor.get('team', {}).get('abbreviation'),
                        'home_away': competitor.get('homeAway'),
                        'score': int(competitor.get('score', 0)),
                        'winner': competitor.get('winner', False),
                        'record': competitor.get('record', [{}])[0] if competitor.get('record') else {}
                    }
                    comp_info['competitors'].append(comp_data)
                
                game_info['competitions'].append(comp_info)
            
            structured_games.append(game_info)
        
        with open(filepath, 'w') as f:
            json.dump(structured_games, f, indent=2)
        
        return filepath
    
    def _export_game_data_csv(self, scoreboard_data: Dict, week: Any, timestamp: str) -> str:
        """Export game data as CSV for easy analysis"""
        filename = f"game_data_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, 'data', filename)
        
        games = scoreboard_data.get('events', [])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'week', 'date', 'home_team_id', 'home_team_name', 'home_score',
                'away_team_id', 'away_team_name', 'away_score', 'home_win', 'margin',
                'total_points'
            ])
            
            for game in games:
                if game.get('status', {}).get('type', {}).get('name') != 'STATUS_FINAL':
                    continue
                    
                for competition in game.get('competitions', []):
                    competitors = competition.get('competitors', [])
                    if len(competitors) != 2:
                        continue
                    
                    home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                    away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                    
                    if home_team and away_team:
                        home_score = int(home_team.get('score', 0))
                        away_score = int(away_team.get('score', 0))
                        margin = home_score - away_score
                        
                        writer.writerow([
                            game.get('id'),
                            game.get('week_number'),
                            game.get('date'),
                            home_team.get('team', {}).get('id'),
                            home_team.get('team', {}).get('displayName'),
                            home_score,
                            away_team.get('team', {}).get('id'), 
                            away_team.get('team', {}).get('displayName'),
                            away_score,
                            home_score > away_score,
                            margin,
                            home_score + away_score
                        ])
        
        return filepath
    
    def _export_h2h_data(self, scoreboard_data: Dict, week: Any, timestamp: str) -> str:
        """Export head-to-head matchup data"""
        filename = f"h2h_matchups_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, 'h2h', filename)
        
        games = scoreboard_data.get('events', [])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'team_a_id', 'team_a_name', 'team_b_id', 'team_b_name',
                'team_a_score', 'team_b_score', 'winner_id', 'margin',
                'week', 'game_id', 'home_team_id'
            ])
            
            for game in games:
                if game.get('status', {}).get('type', {}).get('name') != 'STATUS_FINAL':
                    continue
                    
                for competition in game.get('competitions', []):
                    competitors = competition.get('competitors', [])
                    if len(competitors) != 2:
                        continue
                    
                    team_a = competitors[0]
                    team_b = competitors[1]
                    
                    team_a_score = int(team_a.get('score', 0))
                    team_b_score = int(team_b.get('score', 0))
                    
                    # Determine winner and margin
                    if team_a_score > team_b_score:
                        winner_id = team_a.get('team', {}).get('id')
                        margin = team_a_score - team_b_score
                    else:
                        winner_id = team_b.get('team', {}).get('id')
                        margin = team_b_score - team_a_score
                    
                    # Determine home team
                    home_team_id = None
                    for comp in competitors:
                        if comp.get('homeAway') == 'home':
                            home_team_id = comp.get('team', {}).get('id')
                            break
                    
                    writer.writerow([
                        team_a.get('team', {}).get('id'),
                        team_a.get('team', {}).get('displayName'),
                        team_b.get('team', {}).get('id'),
                        team_b.get('team', {}).get('displayName'),
                        team_a_score,
                        team_b_score,
                        winner_id,
                        margin,
                        game.get('week_number'),
                        game.get('id'),
                        home_team_id
                    ])
        
        return filepath
    
    def _export_team_statistics(self, computation_data: Dict, week: Any, timestamp: str) -> str:
        """Export detailed team statistics"""
        filename = f"team_statistics_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, 'analysis', filename)
        
        season_stats = computation_data.get('season_stats', {})
        rolling_stats = computation_data.get('rolling_stats', {})
        sos_scores = computation_data.get('sos_scores', {})
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'team_id', 'season_games', 'season_wins', 'season_losses', 'season_win_pct',
                'season_avg_points_for', 'season_avg_points_against', 'season_avg_margin',
                'rolling_games', 'rolling_win_pct', 'rolling_avg_margin',
                'strength_of_schedule'
            ])
            
            all_team_ids = set(season_stats.keys()) | set(rolling_stats.keys())
            
            for team_id in all_team_ids:
                season = season_stats.get(team_id, {})
                rolling = rolling_stats.get(team_id, {})
                sos = sos_scores.get(team_id, 0.0)
                
                writer.writerow([
                    team_id,
                    season.get('games_played', 0),
                    season.get('wins', 0),
                    season.get('losses', 0),
                    season.get('win_pct', 0.0),
                    season.get('avg_points_for', 0.0),
                    season.get('avg_points_against', 0.0),
                    season.get('avg_margin', 0.0),
                    rolling.get('games_played', 0),
                    rolling.get('win_pct', 0.0),
                    rolling.get('avg_margin', 0.0),
                    sos
                ])
        
        return filepath
    
    def _export_performance_matrix(self, computation_data: Dict, teams_data: List[Dict], 
                                 week: Any, timestamp: str) -> str:
        """Export performance matrix for ML modeling"""
        filename = f"performance_matrix_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, 'analysis', filename)
        
        season_stats = computation_data.get('season_stats', {})
        rolling_stats = computation_data.get('rolling_stats', {})
        sos_scores = computation_data.get('sos_scores', {})
        
        # Create team name mapping
        team_names = {}
        for team in teams_data:
            team_names[team.get('team', {}).get('id')] = team.get('team', {}).get('displayName', 'Unknown')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'team_id', 'team_name', 'offensive_efficiency', 'defensive_efficiency',
                'recent_form', 'consistency', 'sos_adjusted_margin', 'home_field_factor'
            ])
            
            for team_id, season in season_stats.items():
                rolling = rolling_stats.get(team_id, {})
                sos = sos_scores.get(team_id, 0.0)
                
                # Calculate derived metrics
                offensive_eff = season.get('avg_points_for', 0.0)
                defensive_eff = -season.get('avg_points_against', 0.0)  # Negative because lower is better
                recent_form = rolling.get('avg_margin', 0.0)
                consistency = abs(season.get('avg_margin', 0.0) - rolling.get('avg_margin', 0.0))
                sos_adjusted_margin = season.get('avg_margin', 0.0) + (sos * 0.1)
                
                # Placeholder for home field factor (would need home/away split data)
                home_field_factor = 0.0
                
                writer.writerow([
                    team_id,
                    team_names.get(team_id, f'Team_{team_id}'),
                    offensive_eff,
                    defensive_eff, 
                    recent_form,
                    consistency,
                    sos_adjusted_margin,
                    home_field_factor
                ])
        
        return filepath
    
    def _export_metadata(self, scoreboard_data: Dict, computation_data: Dict, 
                        week: Any, timestamp: str) -> str:
        """Export metadata about the data collection and computation"""
        filename = f"metadata_{week}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, 'data', filename)
        
        games = scoreboard_data.get('events', [])
        
        metadata = {
            'export_timestamp': timestamp,
            'week': str(week),
            'season_year': scoreboard_data.get('season', {}).get('year', 2024),
            'total_games': len(games),
            'completed_games': len([g for g in games if g.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL']),
            'teams_analyzed': len(computation_data.get('season_stats', {})),
            'weeks_covered': list(set([g.get('week_number') for g in games if g.get('week_number')])),
            'data_quality': {
                'missing_weeks': [],  # Would calculate based on expected weeks
                'incomplete_teams': [],  # Teams with < expected games
                'data_source': 'ESPN API'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath