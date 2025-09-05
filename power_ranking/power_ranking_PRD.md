# NFL Weekly Power Rankings App — PRD

## 1. Overview

**Objective**  
Build a comprehensive Python application that calculates NFL team power rankings using complete season data from ESPN's API, with advanced data export capabilities for head-to-head predictive modeling. The system provides both weekly rankings and complete seasonal analysis with all 272 regular season games.

**Key Principles**  
- **Complete Data Coverage**: Collect all 272 regular season games using multiple API methods
- **Data Quality Assurance**: Implement comprehensive validation and duplicate detection
- **Advanced Analytics**: Incorporate data-driven modeling with rolling stats, strength of schedule, and recency weighting
- **Robust Architecture**: Use class-based design with defensive programming for ESPN API volatility
- **H2H Export Capability**: Generate comprehensive datasets for head-to-head predictive modeling
- **Extensible Design**: Enable future data source substitution and model enhancements

---

## 2. Scope & Goals

### 2.1 Features  
- **Comprehensive Data Ingestion**  
  - Fetch complete season data using multiple collection methods:
    - Date-based API calls for all 18 weeks
    - Extended date ranges for Thursday/Monday games
    - ESPN Core API as fallback method
    - Year-based bulk collection for comprehensive coverage
  - Retrieve team metadata from `teams` endpoint with full roster info
  - Implement data validation and duplicate detection across seasons
  - Handle bye weeks and irregular scheduling automatically

- **Advanced Modeling Engine**  
  - Compute power rankings with multi-factor analysis:
    - Margin of victory with diminishing returns
    - Strength of schedule calculations
    - Rolling 5-week performance metrics
    - Home/away performance adjustments
    - Week 18 reduced weighting for resting starters
  - Apply systematic weighting with configurable parameters
  - Generate detailed team statistics and performance matrices

- **Comprehensive Export Functionality**  
  - Power Rankings CSV with full team details and scores
  - Head-to-Head Matchups CSV with all 272 regular season games
  - Game Data exports in both JSON and CSV formats
  - Team Statistics with season and rolling performance data
  - Performance Matrix for comparative team analysis
  - Metadata files with export validation and summaries

- **Resilience & Quality Assurance**  
  - Multiple API collection methods with automatic fallbacks
  - Comprehensive data validation and quality checks
  - Duplicate game detection and removal
  - Bye week verification and team game count validation
  - Debug tools for data collection analysis and troubleshooting

### 2.2 Non-Goals  
- Does not include betting lines ingestion or odds sources
- Not a real-time live-updating system—optimized for complete season analysis
- Does not provide playoff predictions or bracket generation  
- Does not include player-level statistics or injury data
- No web interface—command-line focused for data analysis workflows

---

## 3. Requirements

### 3.1 Functional Requirements

1. **Complete Season Analysis**  
   - Accept `--week` parameter for specific week analysis or full season processing
   - Default to complete season analysis when current season unavailable
   - Support debug mode (`--debug`) for detailed team analysis
   - Provide `run_complete_analysis.py` for verified 272-game dataset processing

2. **Robust ESPN API Integration**  
   - Multi-method data collection:
     - Scoreboard endpoint with date-based queries
     - Extended date ranges for complete weekly coverage  
     - ESPN Core API (`sports.core.api.espn.com`) as fallback
     - Year-based bulk collection for comprehensive data
   - Teams endpoint for metadata and team information
   - Intelligent retry logic with exponential backoff
   - Rate limiting and request throttling

3. **Advanced Modeling Logic**  
   - Multi-factor power score calculation:
     - `PowerScore = (Weighted Margin * SOS) + (Rolling Performance * Recency) + Home/Away Adjustment`
   - Rolling 5-week performance metrics
   - Strength of schedule based on opponent quality
   - Week 18 reduced weighting for teams resting starters
   - Configurable weights via YAML configuration

4. **Comprehensive Export System**  
   - **Power Rankings CSV**: `week, rank, team_id, team_name, power_score`
   - **H2H Matchups CSV**: All 272 games with scores, winners, margins
   - **Game Data**: Complete game details in JSON and CSV formats
   - **Team Statistics**: Season and rolling performance metrics
   - **Performance Matrix**: Team comparison data for analysis
   - **Metadata**: Export validation and summary information

5. **Data Quality & Validation**  
   - Verify exactly 272 regular season games collected
   - Ensure all 32 teams have exactly 17 games each  
   - Detect and remove duplicate games across seasons
   - Validate proper bye week handling
   - Comprehensive logging and error reporting

### 3.2 Non-functional Requirements

- **Reliability**: 
  - Graceful degradation on API failures with multiple collection method fallbacks
  - Intelligent retry logic with exponential backoff
  - Data validation at each collection stage
  - Comprehensive error logging and recovery mechanisms

- **Maintainability**: 
  - Modular, class-based architecture with clear separation of concerns
  - Extensive inline documentation and type hints
  - Comprehensive debug tools for troubleshooting data collection issues
  - Organized export structure with timestamped files

- **Performance**: 
  - Complete season analysis (272 games) in under 2 minutes
  - Efficient API usage with request batching and caching
  - Parallel data collection where possible
  - Optimized data processing with minimal memory footprint

- **Data Quality**:
  - 100% accuracy requirement for game count (exactly 272 regular season games)
  - Zero tolerance for duplicate games across seasons  
  - Complete team coverage (all 32 teams with 17 games each)
  - Comprehensive validation reporting and quality metrics

---

## 4. Architecture & Design

### 4.1 Major Components

#### Core System Components

- **`api/espn_client.py`**  
  - `class ESPNClient`: Advanced API integration with multiple collection methods
    - `get_scoreboard(week=None, season=None)`: Single week data collection
    - `get_last_season_final_rankings()`: Complete 272-game season collection
    - `get_teams()`: Team metadata and information
    - Multiple fallback methods with intelligent retry logic

- **`models/power_rankings.py`**  
  - `class PowerRankModel`: Advanced multi-factor ranking calculations
    - Season and rolling statistics computation
    - Strength of schedule analysis
    - Home/away performance adjustments
    - Week 18 weighting modifications

- **`export/csv_exporter.py`**  
  - `class CSVExporter`: Power rankings CSV generation

- **`export/data_exporter.py`**  
  - `class DataExporter`: Comprehensive data export system
    - H2H matchups CSV with all 272 games
    - Game data in JSON and CSV formats
    - Team statistics and performance matrices
    - Metadata and validation reports

#### Analysis and Debug Tools

- **`run_complete_analysis.py`**  
  - Complete season analysis using verified 272-game dataset
  - Comprehensive data export and validation

- **`debug_full_season.py`**  
  - Multi-method data collection testing and validation
  - Performance analysis across different API approaches

- **Debug and Validation Scripts**:
  - `analyze_extra_games.py`: Duplicate detection and filtering
  - `find_missing_games.py`: Missing game analysis and bye week validation  
  - `search_missing_games.py`: Targeted search for specific matchups
  - `verify_h2h_matches.py`: H2H dataset validation and verification

#### Configuration and Interface

- **`cli.py`**  
  - Enhanced command-line interface with debug mode support
  - Automatic comprehensive data export
  - Flexible week specification and configuration options

- **`config.yaml`**  
  - Comprehensive configuration: API URLs, model weights, export settings

### 4.2 Enhanced Workflow

#### Standard Weekly Analysis
1. Run `python cli.py --week N --debug` for detailed analysis
2. `ESPNClient.get_scoreboard(week=N)` retrieves current week data
3. If no current data available, automatically triggers complete season collection
4. `ESPNClient.get_teams()` provides team metadata and mapping
5. `PowerRankModel.compute()` generates rankings with advanced multi-factor analysis
6. `CSVExporter` creates power rankings CSV
7. `DataExporter` generates comprehensive H2H and analysis datasets
8. Validation reports confirm data quality and completeness

#### Complete Season Analysis  
1. Run `python run_complete_analysis.py` for verified 272-game analysis
2. Load filtered dataset with all regular season games
3. Comprehensive multi-factor power ranking computation
4. Generate complete export suite:
   - Power rankings CSV
   - H2H matchups CSV (272 games)
   - Game data (JSON & CSV)
   - Team statistics and performance matrices
   - Metadata and validation reports
5. Quality assurance verification of all outputs

#### Data Collection & Validation Workflow
1. `debug_full_season.py` tests all collection methods in parallel
2. Multiple API approaches ensure complete data coverage:
   - Date-based collection (18 weeks)
   - Extended date ranges (Thu/Sun/Mon games)
   - ESPN Core API fallback
   - Year-based bulk collection
3. `analyze_extra_games.py` filters duplicates and validates game counts
4. Quality checks ensure exactly 272 games with proper team coverage
5. Comprehensive logging and error reporting throughout

---

## 5. API Details — ESPN Hidden API

- **Scoreboard (NFL)**  
  `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard` :contentReference[oaicite:3]{index=3}. Optionally add parameters like `dates=YYYYMMDD` or `week`.

- **Teams Info (All NFL Teams)**  
  `https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams` :contentReference[oaicite:4]{index=4}.

- **Note from Reddit**: ESPN’s endpoints are free but rate‑limited; prefer reducing duplicate requests and caching :contentReference[oaicite:5]{index=5}.

---

## 6. Configuration & User Interface

### CLI Options

| Flag         | Description                          | Default       |
|--------------|--------------------------------------|---------------|
| `--week`     | NFL week number to process           | current/complete season |
| `--config`   | Path to YAML config file             | `config.yaml` |
| `--output`   | Directory to save exports            | `./output/`   |
| `--dry-run`  | Runs the model without writing files | false         |
| `--debug`    | Show detailed team analysis          | false         |

### Complete Analysis Tool

```bash
python run_complete_analysis.py
```
Uses the verified 272-game dataset for complete season analysis with full data export.

### Debug and Validation Tools

```bash
python debug_full_season.py          # Test all collection methods
python analyze_extra_games.py        # Filter and validate dataset  
python verify_h2h_matches.py         # Verify H2H match completeness
```

### Config File Example (`config.yaml`)

```yaml
espn_api_base: "https://site.api.espn.com/apis/site/v2"
weights:
  avg_margin: 0.6
  sos: 0.3
  recency_bonus: 0.1
output_dir: "./output"
