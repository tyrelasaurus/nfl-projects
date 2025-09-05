#!/usr/bin/env python3
"""
Enhanced NFL Spread Model CLI with structured error handling and logging.

Command-line interface for generating projected point spreads based on power rankings.
Features comprehensive error handling, logging, and recovery mechanisms.

Usage: python cli_enhanced.py --week 1 [--home-field 2.0] [--verbose]
"""

import argparse
import sys
import os
import logging
from typing import Optional
from datetime import datetime

# Import enhanced components
from .data_loader_enhanced import EnhancedDataLoader
from .spread_model import SpreadCalculator
from .exporter import CSVExporter

# Import custom exceptions and configuration
from .exceptions import (
    NFLModelError, DataLoadingError, InvalidArgumentError, MissingArgumentError,
    CLIError, log_model_error
)
from .config_manager import get_nfl_config

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging for the CLI.
    
    Args:
        verbose: Enable verbose console output
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('nfl_model_cli')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler.setLevel(console_level)
    
    # Console formatter
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed file formatter
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger

def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate command-line arguments using configuration.
    
    Args:
        args: Parsed arguments
        logger: Logger instance
        
    Raises:
        InvalidArgumentError: For invalid argument values
        MissingArgumentError: For missing required files
    """
    logger.debug("Validating command-line arguments")
    
    # Load configuration for validation ranges
    config = get_nfl_config()
    week_range = config.cli.validation.get('week_range', [1, 22])
    home_field_range = config.cli.validation.get('home_field_range', [-5.0, 10.0])
    
    # Validate week number
    if not isinstance(args.week, int) or args.week < week_range[0] or args.week > week_range[1]:
        raise InvalidArgumentError(
            f"Invalid week number: {args.week}",
            argument="--week",
            valid_options=[f"{week_range[0]}-{week_range[1]}"],
            context={'provided_value': args.week, 'valid_range': week_range}
        )
    
    # Validate home field advantage
    if (not isinstance(args.home_field, (int, float)) or 
        args.home_field < home_field_range[0] or args.home_field > home_field_range[1]):
        raise InvalidArgumentError(
            f"Invalid home field advantage: {args.home_field}",
            argument="--home-field",
            valid_options=[f"{home_field_range[0]} to {home_field_range[1]}"],
            context={'provided_value': args.home_field, 'valid_range': home_field_range}
        )
    
    # Validate power rankings file
    if not os.path.exists(args.power_rankings):
        raise MissingArgumentError(
            f"Power rankings file not found: {args.power_rankings}",
            missing_argument="--power-rankings",
            context={
                'provided_path': args.power_rankings,
                'file_exists': False,
                'current_directory': os.getcwd()
            }
        )
    
    if not os.access(args.power_rankings, os.R_OK):
        raise MissingArgumentError(
            f"Power rankings file not readable: {args.power_rankings}",
            missing_argument="--power-rankings",
            context={
                'provided_path': args.power_rankings,
                'file_readable': False
            }
        )
    
    # Validate schedule file
    if not os.path.exists(args.schedule):
        raise MissingArgumentError(
            f"Schedule file not found: {args.schedule}",
            missing_argument="--schedule",
            context={
                'provided_path': args.schedule,
                'file_exists': False,
                'current_directory': os.getcwd()
            }
        )
    
    # Validate/create output directory
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {args.output_dir}")
        except PermissionError:
            raise InvalidArgumentError(
                f"Cannot create output directory: {args.output_dir}",
                argument="--output-dir",
                context={'permission_error': True}
            )
    
    logger.debug("All arguments validated successfully")

def main():
    """Main CLI entry point with enhanced error handling."""
    # Load configuration for defaults
    try:
        config = get_nfl_config()
        default_hfa = config.model.home_field_advantage
        default_pr_path = config.data.default_paths.get('power_rankings', 
            '../power_ranking/output/power_rankings_week_initial_adjusted.csv')
        default_schedule_path = config.data.default_paths.get('schedule',
            '../power_ranking/nfl_schedule_2025_20250831_220432.csv')
        week_range = config.cli.validation.get('week_range', [1, 22])
        hf_range = config.cli.validation.get('home_field_range', [-5.0, 10.0])
    except Exception:
        # Fallback to hardcoded defaults if config loading fails
        default_hfa = 2.0
        default_pr_path = "../power_ranking/output/power_rankings_week_initial_adjusted.csv"
        default_schedule_path = "../power_ranking/nfl_schedule_2025_20250831_220432.csv"
        week_range = [1, 22]
        hf_range = [-5.0, 10.0]
    
    # Setup argument parser with dynamic help text
    parser = argparse.ArgumentParser(
        description="Generate NFL point spread projections from power rankings with enhanced error handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --week 1                    # Generate spreads for week 1
  %(prog)s --week 5 --home-field 2.5   # Custom home field advantage  
  %(prog)s --week 1 --verbose          # Verbose output
  %(prog)s --week 1 --log-file nfl.log # Log to file
        """
    )
    
    parser.add_argument(
        "--week", "-w",
        type=int,
        required=True,
        help=f"NFL week number to generate spreads for ({week_range[0]}-{week_range[1]})"
    )
    
    parser.add_argument(
        "--home-field", "-hf",
        type=float,
        default=default_hfa,
        help=f"Home field advantage in points (default: {default_hfa}, range: {hf_range[0]} to {hf_range[1]})"
    )
    
    parser.add_argument(
        "--power-rankings", "-pr",
        type=str,
        default=default_pr_path,
        help="Path to power rankings CSV file"
    )
    
    parser.add_argument(
        "--schedule", "-s", 
        type=str,
        default=default_schedule_path,
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
        help="Enable verbose output and debug logging"
    )
    
    parser.add_argument(
        "--log-file", "-l",
        type=str,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback data loading (fail fast on data errors)"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable data validation (faster but less safe)"
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit on error, catch it to provide better error handling
        if e.code != 0:
            logger = logging.getLogger('nfl_model_cli')
            logger.error("Invalid command-line arguments. Use --help for usage information.")
        raise
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("NFL Spread Model CLI - Enhanced Version")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    logger.debug(f"Command-line arguments: {vars(args)}")
    
    try:
        # Validate arguments
        validate_arguments(args, logger)
        
        # Initialize components with enhanced error handling
        logger.info("Initializing components...")
        
        data_loader = EnhancedDataLoader(
            validate_data=not args.no_validation,
            use_fallback=not args.no_fallback
        )
        
        calculator = SpreadCalculator(home_field_advantage=args.home_field)
        exporter = CSVExporter(args.output_dir)
        
        logger.debug("Components initialized successfully")
        
        # Load data with enhanced error handling
        logger.info("Loading data...")
        logger.info(f"  Power rankings: {args.power_rankings}")
        logger.info(f"  Schedule: {args.schedule}")
        
        try:
            power_ratings = data_loader.load_power_rankings(args.power_rankings)
            logger.info(f"  Loaded {len(power_ratings)} team power ratings")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Power ratings preview:")
                for i, (team, rating) in enumerate(list(power_ratings.items())[:5]):
                    logger.debug(f"    {team}: {rating:.2f}")
                if len(power_ratings) > 5:
                    logger.debug(f"    ... and {len(power_ratings) - 5} more teams")
        
        except DataLoadingError as e:
            logger.error(f"Failed to load power rankings: {e.message}")
            if e.recovery_suggestions:
                logger.info("Recovery suggestions:")
                for suggestion in e.recovery_suggestions:
                    logger.info(f"  - {suggestion}")
            sys.exit(1)
        
        try:
            matchups = data_loader.load_schedule(args.schedule, args.week)
            logger.info(f"  Found {len(matchups)} games for week {args.week}")
            
            if logger.isEnabledFor(logging.DEBUG) and matchups:
                logger.debug("Scheduled matchups:")
                for i, (home, away, date) in enumerate(matchups[:3]):
                    logger.debug(f"    {away} @ {home} ({date})")
                if len(matchups) > 3:
                    logger.debug(f"    ... and {len(matchups) - 3} more games")
        
        except DataLoadingError as e:
            logger.error(f"Failed to load schedule: {e.message}")
            if e.recovery_suggestions:
                logger.info("Recovery suggestions:")
                for suggestion in e.recovery_suggestions:
                    logger.info(f"  - {suggestion}")
            sys.exit(1)
        
        # Check if we have games to process
        if not matchups:
            logger.warning(f"No games found for week {args.week}")
            logger.info("This could be normal if the week hasn't been scheduled yet")
            sys.exit(0)
        
        # Calculate spreads
        logger.info("Calculating spreads...")
        logger.info(f"  Home field advantage: {args.home_field} points")
        
        try:
            # Convert matchups to the format expected by calculator
            matchup_tuples = [(home, away) for home, away, _ in matchups]
            results = calculator.calculate_week_spreads(matchup_tuples, power_ratings, args.week)
            
            logger.info(f"  Calculated {len(results)} spread predictions")
        
        except Exception as e:
            logger.error(f"Failed to calculate spreads: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug("Full traceback:")
                logger.debug(traceback.format_exc())
            sys.exit(1)
        
        # Export results
        logger.info("Exporting results...")
        
        try:
            spreads_file = exporter.export_week_spreads(results, args.week)
            summary_file = exporter.export_summary_stats(results, args.week)
            
            logger.info("Export completed successfully!")
            logger.info(f"  Spreads: {spreads_file}")
            logger.info(f"  Summary: {summary_file}")
        
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            logger.warning("Calculation was successful but export failed")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug("Full traceback:")
                logger.debug(traceback.format_exc())
            # Don't exit here - we can still show results
        
        # Display results summary
        if results:
            logger.info("\n" + "=" * 40)
            logger.info(f"Week {args.week} Spread Predictions Summary")
            logger.info("=" * 40)
            
            # Show sample results
            logger.info("Sample matchups:")
            for i, result in enumerate(results[:5]):  # Show first 5
                betting_line = calculator.format_spread_as_betting_line(
                    result.projected_spread, result.home_team
                )
                logger.info(f"  {result.away_team} @ {result.home_team}: {betting_line}")
            
            if len(results) > 5:
                logger.info(f"  ... and {len(results) - 5} more games")
            
            # Statistics
            spreads = [abs(result.projected_spread) for result in results]
            avg_spread = sum(spreads) / len(spreads) if spreads else 0
            max_spread = max(spreads) if spreads else 0
            
            logger.info(f"\nSpread Statistics:")
            logger.info(f"  Average spread: {avg_spread:.1f} points")
            logger.info(f"  Largest spread: {max_spread:.1f} points")
            logger.info(f"  Home favorites: {sum(1 for r in results if r.projected_spread < 0)}")
            logger.info(f"  Road favorites: {sum(1 for r in results if r.projected_spread > 0)}")
    
    except InvalidArgumentError as e:
        logger.error(f"Invalid argument: {e.message}")
        if e.recovery_suggestions:
            logger.info("Suggestions:")
            for suggestion in e.recovery_suggestions:
                logger.info(f"  - {suggestion}")
        sys.exit(1)
    
    except MissingArgumentError as e:
        logger.error(f"Missing required data: {e.message}")
        if e.recovery_suggestions:
            logger.info("Suggestions:")
            for suggestion in e.recovery_suggestions:
                logger.info(f"  - {suggestion}")
        sys.exit(1)
    
    except NFLModelError as e:
        logger.error(f"NFL Model error: {e.message}")
        if e.context:
            logger.debug(f"Error context: {e.context}")
        if e.recovery_suggestions:
            logger.info("Recovery suggestions:")
            for suggestion in e.recovery_suggestions:
                logger.info(f"  - {suggestion}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
        # Log full context for debugging
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug("Full traceback:")
            logger.debug(traceback.format_exc())
        else:
            logger.info("Run with --verbose for detailed error information")
        
        logger.error("This is likely a bug. Please report this issue.")
        sys.exit(1)
    
    finally:
        logger.info("=" * 60)
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()