import csv
import os
from typing import List
from datetime import datetime
from .spread_model import MatchupResult


class CSVExporter:
    """Exports spread calculations to CSV files."""
    
    def __init__(self, output_directory: str = "output"):
        """
        Initialize CSVExporter.
        
        Args:
            output_directory: Directory to save CSV files (default: "output")
        """
        self.output_directory = output_directory
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    
    def export_week_spreads(self, results: List[MatchupResult], week: int) -> str:
        """
        Export weekly spread calculations to CSV.
        
        Args:
            results: List of MatchupResult objects
            week: Week number
            
        Returns:
            Path to the created CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"projected_spreads_week_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_directory, filename)
        
        # CSV headers as specified in PRD
        headers = [
            'week',
            'home_team', 
            'away_team',
            'home_power',
            'away_power', 
            'neutral_diff',
            'home_field_adj',
            'projected_spread',
            'game_date',
            'betting_line'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for result in results:
                # Format betting line
                if result.projected_spread > 0:
                    betting_line = f"{result.home_team} -{abs(result.projected_spread):.1f}"
                elif result.projected_spread < 0:
                    betting_line = f"{result.home_team} +{abs(result.projected_spread):.1f}"
                else:
                    betting_line = f"{result.home_team} PK"
                
                row = [
                    result.week,
                    result.home_team,
                    result.away_team,
                    round(result.home_power, 3),
                    round(result.away_power, 3),
                    round(result.neutral_diff, 3),
                    round(result.home_field_adj, 1),
                    round(result.projected_spread, 1),
                    result.game_date,
                    betting_line
                ]
                writer.writerow(row)
        
        return filepath
    
    def export_summary_stats(self, results: List[MatchupResult], week: int) -> str:
        """
        Export summary statistics for the week's spreads.
        
        Args:
            results: List of MatchupResult objects
            week: Week number
            
        Returns:
            Path to the created summary CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spread_summary_week_{week}_{timestamp}.csv"
        filepath = os.path.join(self.output_directory, filename)
        
        if not results:
            return filepath
        
        # Calculate summary statistics
        spreads = [r.projected_spread for r in results]
        power_diffs = [r.neutral_diff for r in results]
        
        avg_spread = sum(spreads) / len(spreads)
        max_spread = max(spreads)
        min_spread = min(spreads)
        avg_power_diff = sum(power_diffs) / len(power_diffs)
        
        # Count favorites
        home_favorites = sum(1 for s in spreads if s > 0)
        away_favorites = sum(1 for s in spreads if s < 0)
        pick_ems = sum(1 for s in spreads if s == 0)
        
        headers = [
            'metric',
            'value'
        ]
        
        summary_data = [
            ('total_games', len(results)),
            ('avg_projected_spread', round(avg_spread, 2)),
            ('max_projected_spread', round(max_spread, 1)),
            ('min_projected_spread', round(min_spread, 1)),
            ('avg_neutral_diff', round(avg_power_diff, 2)),
            ('home_favorites', home_favorites),
            ('away_favorites', away_favorites), 
            ('pick_ems', pick_ems),
            ('home_field_advantage', results[0].home_field_adj if results else 3.0)
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(summary_data)
        
        return filepath
