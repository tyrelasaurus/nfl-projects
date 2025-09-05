#!/usr/bin/env python3
"""
NFL Spread Model CLI

Command-line interface for generating projected point spreads based on power rankings.
Usage: python cli.py --week 1 [--home-field 2.0]
"""

import argparse
import sys
import os
from typing import Optional

from data_loader import DataLoader
from spread_model import SpreadCalculator
from exporter import CSVExporter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate NFL point spread projections from power rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--week", "-w",
        type=int,
        required=True,
        help="NFL week number to generate spreads for"
    )
    
    parser.add_argument(
        "--home-field", "-hf",
        type=float,
        default=2.0,
        help="Home field advantage in points (default: 2.0)"
    )
    
    parser.add_argument(
        "--power-rankings", "-pr",
        type=str,
        default="../power_ranking/output/power_rankings_week_initial_adjusted.csv",
        help="Path to power rankings CSV file"
    )
    
    parser.add_argument(
        "--schedule", "-s", 
        type=str,
        default="../power_ranking/nfl_schedule_2025_20250831_220432.csv",
        help="Path to NFL schedule CSV file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for CSV files (default: output)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.power_rankings):
        print(f"Error: Power rankings file not found: {args.power_rankings}")
        sys.exit(1)
        
    if not os.path.exists(args.schedule):
        print(f"Error: Schedule file not found: {args.schedule}")
        sys.exit(1)
    
    try:
        # Initialize components
        if args.verbose:
            print(f"Loading data...")
            print(f"  Power rankings: {args.power_rankings}")
            print(f"  Schedule: {args.schedule}")
            
        data_loader = DataLoader(args.power_rankings, args.schedule)
        calculator = SpreadCalculator(home_field_advantage=args.home_field)
        exporter = CSVExporter(args.output_dir)
        
        # Load data
        power_ratings = data_loader.load_power_rankings()
        matchups = data_loader.get_weekly_matchups(args.week)
        
        if args.verbose:
            print(f"  Loaded {len(power_ratings)} team power ratings")
            print(f"  Found {len(matchups)} games for week {args.week}")
        
        if not matchups:
            print(f"No games found for week {args.week}")
            sys.exit(1)
        
        # Calculate spreads
        if args.verbose:
            print(f"\nCalculating spreads (home field advantage: {args.home_field})...")
            
        results = calculator.calculate_week_spreads(matchups, power_ratings, args.week)
        
        # Export results
        spreads_file = exporter.export_week_spreads(results, args.week)
        summary_file = exporter.export_summary_stats(results, args.week)
        
        print(f"\nWeek {args.week} spreads generated successfully!")
        print(f"  Spreads: {spreads_file}")
        print(f"  Summary: {summary_file}")
        
        # Display sample results
        if args.verbose and results:
            print(f"\nSample matchups:")
            for i, result in enumerate(results[:5]):  # Show first 5
                betting_line = calculator.format_spread_as_betting_line(
                    result.projected_spread, result.home_team
                )
                print(f"  {result.away_team} @ {result.home_team}: {betting_line}")
            
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more games")
        
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()