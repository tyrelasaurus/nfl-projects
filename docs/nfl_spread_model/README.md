# NFL Spread Model Documentation

This directory contains documentation specific to the NFL Spread Model system.

## System Overview

The NFL Spread Model implements the Billy Walters methodology for predicting point spreads, transforming power rankings into actionable betting line predictions with statistical confidence intervals.

### Key Features

- **Billy Walters Methodology**: Proven approach used by professional sports bettors
- **Power Rating Integration**: Seamless integration with Power Rankings System
- **Confidence Intervals**: Statistical uncertainty quantification
- **Backtesting Framework**: Historical validation and performance metrics
- **Multiple Export Formats**: CSV, JSON with comprehensive metadata

### Core Philosophy

The model follows Billy Walters' fundamental principle:

> "The spread is simply the power rating differential plus home field advantage"

This approach emphasizes simplicity and statistical robustness over complex algorithms.

## Architecture Components

```
NFL Spread Model
├── Core Engine (nfl_model.spread_model)
│   ├── SpreadCalculator
│   ├── MatchupResult dataclass
│   └── Billy Walters methodology
├── Data Loading (nfl_model.data_loader)
│   ├── Power rankings import
│   ├── Schedule data processing
│   └── Validation integration
├── Configuration (nfl_model.config_manager)
│   ├── Model parameters
│   ├── Home field adjustments
│   └── Export settings
├── Validation Framework (nfl_model.validation)
│   ├── Backtesting engine
│   ├── Performance metrics
│   └── Statistical analysis
└── Export System (nfl_model.exporter)
    ├── Spread predictions CSV
    ├── Weekly summaries
    └── Confidence intervals
```

## Quick Start

### Basic Spread Calculation

```python
from nfl_model.spread_model import SpreadCalculator
from nfl_model.data_loader import DataLoader

# Initialize components
calculator = SpreadCalculator(home_field_advantage=2.5)
loader = DataLoader("power_rankings.csv", "schedule.csv")

# Load power rankings and schedule
power_rankings = loader.load_power_rankings()
week1_df = loader.load_schedule(week=1)

# Normalize to matchup tuples (home, away, date)
matchups = [(r.home_team, r.away_team, getattr(r, 'game_date', '')) for r in week1_df.itertuples(index=False)]

# Calculate spreads for all week 1 matchups
results = calculator.calculate_week_spreads(matchups, power_rankings, week=1)

# Display results
for r in results:
    print(f"{r.away_team} @ {r.home_team}: {calculator.format_spread_as_betting_line(r.projected_spread, r.home_team)}")
```

### Advanced Usage with Confidence Intervals

```python
from nfl_model.spread_model import SpreadCalculator
from nfl_model.validation.performance_metrics import ConfidenceCalculator

calculator = SpreadCalculator()
confidence_calc = ConfidenceCalculator()

# Calculate spread with uncertainty quantification
spread = calculator.calculate_spread("KC", "BUF", power_rankings)
confidence = confidence_calc.calculate_prediction_confidence(
    home_power=power_rankings["KC"],
    away_power=power_rankings["BUF"],
    historical_accuracy=0.54  # 54% ATS accuracy
)

print(f"Spread: KC {spread:+.1f}")
print(f"Confidence: {confidence:.1%}")
print(f"Expected range: {spread-3:.1f} to {spread+3:.1f}")
```

## Configuration

### Model Configuration

```yaml
# nfl_model_config.yaml
spread_model:
  # Billy Walters methodology parameters
  home_field_advantage: 2.5      # Standard NFL home advantage
  power_rating_scale: 11.0       # Target standard deviation
  
  # Confidence calculations
  confidence_threshold: 0.524    # Break-even ATS percentage
  spread_variance: 3.5           # Historical spread prediction error
  
  # Backtesting parameters
  backtest_seasons: [2020, 2021, 2022, 2023]
  validation_weeks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
  
# Data sources
data:
  power_rankings_source: "../power_ranking/output/power_rankings.csv"
  schedule_source: "data/nfl_schedule.csv"
  
# Export settings
export:
  include_confidence: true
  round_spreads: true
  decimal_precision: 1
```

## Statistical Methodology

### Billy Walters Formula

The core calculation follows Walters' proven methodology:

```
Spread = (Home_Power - Away_Power) + Home_Field_Advantage
```

#### Power Rating Normalization

Power ratings are normalized to ensure realistic spread distributions:

```python
def normalize_power_ratings(ratings):
    """Normalize power ratings for spread calculations."""
    mean_rating = statistics.mean(ratings.values())
    std_rating = statistics.stdev(ratings.values())
    
    # Target standard deviation for realistic NFL spreads
    target_std = 11.0
    scale_factor = target_std / std_rating
    
    return {
        team: (rating - mean_rating) * scale_factor 
        for team, rating in ratings.items()
    }
```

#### Home Field Advantage

Default home field advantage of 2.5 points can be adjusted based on:

- **Venue factors**: Altitude (Denver), weather (Green Bay), crowd noise (Seattle)
- **Travel distance**: Cross-country games receive additional adjustment
- **Rest differential**: Teams with more rest receive slight advantage
- **Historical performance**: Team-specific home/away splits

### Confidence Interval Calculation

The model provides statistical confidence based on historical performance:

```python
def calculate_confidence_interval(spread, historical_accuracy):
    """Calculate confidence interval for spread prediction."""
    # Base confidence from historical accuracy
    confidence = min(historical_accuracy * 2, 1.0)
    
    # Standard deviation of spread prediction errors
    spread_std = 3.5
    
    # Calculate bounds (±1 standard deviation)
    lower_bound = spread - (spread_std * confidence)
    upper_bound = spread + (spread_std * confidence)
    
    return lower_bound, upper_bound, confidence
```

## API Reference

### Core Classes

#### SpreadCalculator

Primary spread calculation engine.

```python
class SpreadCalculator:
    def __init__(self, home_field_advantage=2.5, config=None)
    
    def calculate_spread(self, home_team: str, away_team: str, 
                        power_rankings: Dict[str, float]) -> float
    
    def calculate_neutral_spread(self, team_a_power: float, 
                               team_b_power: float) -> float
                               
    def calculate_matchup_results(self, schedule: List, 
                                power_rankings: Dict) -> List[MatchupResult]
```

**Key Methods:**

- `calculate_spread()`: Main spread calculation with home field advantage
- `calculate_neutral_spread()`: Neutral field spread for comparison
- `calculate_matchup_results()`: Batch processing for full week/season

#### DataLoader

Data import and processing utilities.

```python
class DataLoader:
    def load_power_rankings(self) -> Dict[str, float]
    def load_schedule(self, week: int | None = None) -> pd.DataFrame
    def get_weekly_matchups(self, week: int) -> List[Tuple[str, str, str]]
    def validate_data_compatibility(self, rankings, matchups) -> Dict[str, Any]
```

Schedule CSV schema
- Canonical columns: `week`, `home_team`, `away_team`, optional `game_date`
- Alternate supported: `home_team_name`/`away_team_name` are auto-normalized
- Utility: `nfl_model.data_loader.normalize_schedule_dataframe(df)` can validate/normalize a DataFrame
```

### Data Structures

#### MatchupResult

Comprehensive spread calculation result.

```python
@dataclass
class MatchupResult:
    week: int                    # NFL week number
    home_team: str              # Home team abbreviation
    away_team: str              # Away team abbreviation
    home_power: float           # Home team power rating
    away_power: float           # Away team power rating
    neutral_diff: float         # Power rating differential
    home_field_adj: float       # Home field advantage applied
    projected_spread: float     # Final spread prediction
    game_date: str             # Game date/time
```

## Backtesting and Validation

### Performance Metrics

The model tracks several key performance indicators:

#### Against-the-Spread (ATS) Accuracy

```python
def calculate_ats_accuracy(predictions, actual_results):
    """Calculate against-the-spread accuracy percentage."""
    correct_predictions = 0
    total_predictions = len(predictions)
    
    for pred, actual in zip(predictions, actual_results):
        predicted_winner = "home" if pred.spread < 0 else "away"
        actual_winner = determine_ats_winner(actual.home_score, actual.away_score, pred.spread)
        
        if predicted_winner == actual_winner:
            correct_predictions += 1
    
    return (correct_predictions / total_predictions) * 100
```

Target: **>52.4%** (break-even with standard -110 juice)

#### Mean Absolute Error (MAE)

```python
def calculate_mae(predictions, actual_margins):
    """Calculate mean absolute error in spread predictions."""
    errors = [abs(pred.spread - actual) for pred, actual in zip(predictions, actual_margins)]
    return sum(errors) / len(errors)
```

Target: **<3.5 points** per game

#### Profit/Loss Simulation

```python
def simulate_betting_performance(predictions, results, unit_size=100):
    """Simulate betting performance with standard units."""
    total_profit = 0
    wins = 0
    losses = 0
    
    for pred, result in zip(predictions, results):
        # Simulate betting on predicted side
        if is_correct_prediction(pred, result):
            total_profit += unit_size * 0.91  # Win $91 on $100 bet (-110 juice)
            wins += 1
        else:
            total_profit -= unit_size  # Lose $100
            losses += 1
    
    return {
        'total_profit': total_profit,
        'wins': wins,
        'losses': losses,
        'win_percentage': wins / (wins + losses) * 100,
        'roi': (total_profit / (wins + losses) / unit_size) * 100
    }
```

### Historical Validation

```python
from nfl_model.validation.backtesting import BacktestEngine

# Initialize backtesting engine
backtester = BacktestEngine()

# Run comprehensive backtest
results = backtester.run_full_backtest(
    seasons=[2020, 2021, 2022, 2023],
    power_rankings_by_week=historical_rankings,
    actual_results=historical_games
)

# Display performance summary
print(f"ATS Accuracy: {results.ats_accuracy:.1f}%")
print(f"Mean Absolute Error: {results.mae:.2f} points")
print(f"Total Profit (100 unit bets): ${results.total_profit:,.2f}")
print(f"ROI: {results.roi:+.2f}%")
```

## Export Formats

### Spread Predictions CSV

```csv
Week,Date,Home_Team,Away_Team,Home_Power,Away_Power,Neutral_Diff,HFA,Projected_Spread,Confidence
1,2024-09-08,KC,BUF,12.5,11.2,1.3,2.5,-3.8,0.68
1,2024-09-08,DAL,NYG,8.9,3.2,5.7,2.5,-8.2,0.71
```

### Weekly Summary JSON

```json
{
  "week": 1,
  "season": 2024,
  "predictions": [
    {
      "home_team": "KC",
      "away_team": "BUF", 
      "projected_spread": -3.8,
      "confidence": 0.68,
      "power_differential": 1.3,
      "home_field_advantage": 2.5,
      "prediction_bounds": {
        "lower": -7.1,
        "upper": -0.5
      }
    }
  ],
  "summary": {
    "total_games": 16,
    "average_spread": 4.2,
    "high_confidence_games": 12,
    "model_version": "1.0.0"
  },
  "generated_at": "2024-09-01T14:30:00Z"
}
```

## Integration with Power Rankings

### Data Flow

```
Power Rankings System → CSV Export → Spread Model → Predictions
```

### Required Data Format

The spread model expects power rankings in this CSV format:

```csv
Rank,Team,Power_Score,Season_Margin,Rolling_Margin,SOS,Games_Played
1,KC,12.45,8.2,9.1,2.3,17
2,BUF,11.87,7.8,8.9,1.9,17
```

### Automated Integration

```python
# Automated pipeline from power rankings to spreads
from power_ranking.export.csv_exporter import CSVExporter
from nfl_model.data_loader import DataLoader
from nfl_model.spread_model import SpreadCalculator

# Export power rankings
exporter = CSVExporter()
exporter.export_rankings(rankings, "power_rankings.csv")

# Load into spread model
loader = DataLoader()
power_data = loader.load_power_rankings("power_rankings.csv")

# Generate spread predictions
calculator = SpreadCalculator()
predictions = calculator.calculate_matchup_results(schedule, power_data)
```

## Command Line Interface

### Basic Usage

```bash
# Generate spreads for specific week
python -m nfl_model.cli --week 1 --power-rankings power_rankings.csv

# Custom home field advantage
python -m nfl_model.cli --week 1 --home-field 3.0

# Include confidence intervals
python -m nfl_model.cli --week 1 --confidence --export-json

# Backtest historical performance
python -m nfl_model.cli --backtest --seasons 2022,2023 --report
```

### Advanced Options

```bash
# Full season projection
python -m nfl_model.cli --full-season --power-rankings weekly_rankings/

# Export multiple formats
python -m nfl_model.cli --week 1 --export csv,json --output-dir results/

# Validation mode with Vegas line comparison
python -m nfl_model.cli --week 1 --validate --vegas-lines vegas_week1.csv
```

## Testing

### Unit Tests

```bash
# Core spread calculation tests
python -m pytest nfl_model/tests/test_spread_model.py -v

# Data loading tests
python -m pytest nfl_model/tests/test_data_loader.py -v

# Validation framework tests
python -m pytest nfl_model/tests/test_validation.py -v
```

### Integration Tests

```bash
# Full pipeline test
python -m pytest nfl_model/tests/integration/test_full_pipeline.py

# Backtesting validation
python -m pytest nfl_model/tests/integration/test_backtesting.py
```

### Performance Tests

```bash
# Large dataset processing
python -m pytest nfl_model/tests/performance/test_large_datasets.py

# Memory usage validation
python -m pytest nfl_model/tests/performance/test_memory_usage.py
```

## Troubleshooting

### Common Issues

1. **Unrealistic Spreads (>20 points)**
   - Check power rating normalization
   - Verify home field advantage setting
   - Inspect input power rankings for outliers

2. **Poor ATS Performance (<50%)**
   - Review power ranking quality
   - Adjust home field advantage
   - Check for data quality issues

3. **Data Loading Errors**
   - Verify CSV format matches expected schema
   - Check team abbreviation consistency
   - Ensure all required columns present

### Diagnostic Tools

```python
# Spread distribution analysis
from nfl_model.validation.diagnostics import SpreadAnalyzer

analyzer = SpreadAnalyzer()
stats = analyzer.analyze_spread_distribution(predictions)
print(f"Average spread: {stats.mean:.1f}")
print(f"Standard deviation: {stats.std:.1f}")
print(f"Outliers (>14 points): {stats.outliers}")

# Power rating validation
validator = PowerRatingValidator()
issues = validator.validate_power_ratings(power_rankings)
for issue in issues:
    print(f"WARNING: {issue}")
```

---

For detailed API documentation, see the [main API reference](../api_reference.md).
For statistical methodology details, see [Statistical Methods](../statistical_methods.md).
For integration with Power Rankings, see [Power Rankings Documentation](../power_rankings/README.md).
