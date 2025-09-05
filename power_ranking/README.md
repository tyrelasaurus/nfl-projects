# NFL Power Rankings App

A comprehensive Python application that calculates NFL team power rankings using ESPN's API data with complete season coverage and advanced data export capabilities.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (CLI)
```bash
python -m power_ranking.power_ranking.cli
```

When no current season games are available (e.g., during the off-season), the application automatically generates initial power rankings based on the complete previous season data (all 272 regular season games).

### Debug Mode (Detailed Team Analysis)
```bash
python -m power_ranking.power_ranking.cli --debug
```

Shows detailed team metrics including season stats, rolling 5-week performance, and strength of schedule for key teams.

### Specify Week
```bash
python -m power_ranking.power_ranking.cli --week 5
```

### Dry Run (no file output)
```bash
python -m power_ranking.power_ranking.cli --dry-run
```

### Custom Output Directory
```bash
python -m power_ranking.power_ranking.cli --output ./custom_output
```

### Custom Config File
```bash
python -m power_ranking.power_ranking.cli --config my_config.yaml
```

### Last-N Games Per Team (Cross-Season)
- The model now uses each team's most recent N games across seasons (default N=17).
- The CLI merges current-season games with the last completed season to build the pool, then selects each team’s last N by timestamp.

Examples:
```bash
# Default last 17 games per team (recommended)
python -m power_ranking.power_ranking.cli --week 1 --last-n 17 --dry-run

# Use a different window (e.g., last 10 games)
python -m power_ranking.power_ranking.cli --week 3 --last-n 10
```

### Complete Analysis with Full Dataset
```bash
python run_complete_analysis.py
```

Runs analysis using the verified 272-game dataset with comprehensive data export.

## Comprehensive Data Export

The application automatically exports detailed datasets for head-to-head predictive modeling:
- **Game Data**: Complete game-by-game results with computed metrics (JSON & CSV)
- **Team Statistics**: Full season and rolling 5-week performance data (CSV)
- **Performance Matrix**: Team comparison metrics for analysis (CSV)
- **Head-to-Head Matchups**: All 272 regular season matchups with scores and metadata (CSV)
- **Metadata**: Export information and data dictionary (JSON)

## Automatic Fallback to Previous Season

When the current NFL season hasn't started yet or has no completed games, the application will:

1. Detect the absence of current season game data
2. Automatically fetch complete data from the most recently completed season (all 272 regular season games)
3. Use multiple collection methods for comprehensive coverage:
   - Date-based API calls for all 18 weeks
   - Extended date ranges for Thursday/Monday games  
   - ESPN Core API as fallback
   - Year-based bulk collection
4. Generate initial power rankings based on complete historical data
5. Label the output as "Initial (based on 2024 season)"
6. Save the CSV file with "_initial_adjusted" suffix

This ensures the application always provides meaningful rankings based on complete season data from the most recently completed NFL season, even during the off-season.

## Configuration

Edit `config.yaml` to adjust:
- API base URL
- Ranking model weights
- Output directory
- Logging level

## Output

### Power Rankings CSV
The application generates CSV files with the format:
- `power_rankings_week_N.csv`
- Columns: week, rank, team_id, team_name, power_score

### Comprehensive Data Export Structure

The application automatically creates organized exports in structured directories:

#### Game Data (`./output/data/`)
- **JSON**: `game_data_[week]_[timestamp].json` - Complete game results with computed metrics
- **CSV**: `game_data_[week]_[timestamp].csv` - Same data in spreadsheet format
- **Metadata**: `metadata_[week]_[timestamp].json` - Export validation and summary

#### Team Analysis (`./output/analysis/`)
- **Team Statistics**: `team_statistics_[week]_[timestamp].csv` - Full season and rolling 5-week stats
- **Performance Matrix**: `performance_matrix_[week]_[timestamp].csv` - Comparative team metrics

#### Head-to-Head Data (`./output/h2h/`)
- **H2H Matchups**: `h2h_matchups_[week]_[timestamp].csv` - All 272 regular season games with:
  - Team IDs and names
  - Scores and winners
  - Week numbers and game IDs
  - Home/away designation
  - Victory margins

#### Debug and Analysis Tools
- **Complete Analysis**: `run_complete_analysis.py` - Uses verified 272-game dataset
- **Debug Tools**: Various debugging scripts for data collection verification
- **Data Validation**: Scripts to verify complete season coverage

## Architecture

### Core Components
- `api/espn_client.py` - ESPN API integration with multiple collection methods
- `models/power_rankings.py` - Advanced power ranking calculations with rolling stats
- `export/csv_exporter.py` - Power rankings CSV export functionality
- `export/data_exporter.py` - Comprehensive data export for H2H modeling
- `cli.py` - Enhanced command-line interface with debug mode
- `config.yaml` - Configuration settings for weights and API parameters

### Analysis and Debug Tools
- `run_complete_analysis.py` - Complete analysis using verified 272-game dataset
- `debug_full_season.py` - Comprehensive data collection testing across multiple methods
- `analyze_extra_games.py` - Data filtering and duplicate removal
- `find_missing_games.py` - Analysis of missing games and bye weeks
- `search_missing_games.py` - Targeted search for specific missing matchups
- `verify_h2h_matches.py` - Verification of complete H2H dataset

### Data Quality Features
- **Complete Season Coverage**: Collects all 272 regular season games
- **Multiple Collection Methods**: Date-based, extended dates, Core API, and year-based approaches
- **Duplicate Detection**: Filters out duplicate games from different seasons
- **Bye Week Handling**: Properly accounts for NFL bye week schedules
- **Data Validation**: Comprehensive verification of team game counts and week coverage
