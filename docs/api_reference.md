# NFL Projects API Reference

This document provides comprehensive API documentation for the NFL Projects suite, covering both the Power Rankings System and NFL Spread Model.

## Table of Contents

- [Power Rankings System API](#power-rankings-system-api)
- [NFL Spread Model API](#nfl-spread-model-api)
- [Common Data Structures](#common-data-structures)
- [Error Handling](#error-handling)

## Power Rankings System API

### Core Models

#### `power_ranking.models.power_rankings.PowerRankingsModel`

The main power rankings calculation engine.

```python
class PowerRankingsModel:
    def __init__(self, config: Optional[Dict] = None)
    def calculate_power_rankings(self, games: List[GameData]) -> List[TeamRanking]
    def update_rankings(self, new_games: List[GameData]) -> List[TeamRanking]
```

**Methods:**

##### `calculate_power_rankings(games: List[GameData]) -> List[TeamRanking]`
Calculates power rankings based on game data.

**Parameters:**
- `games` (List[GameData]): List of completed games with scores and metadata

**Returns:**
- `List[TeamRanking]`: Ordered list of team rankings with power scores

**Example:**
```python
model = PowerRankingsModel()
rankings = model.calculate_power_rankings(games)
for ranking in rankings:
    print(f"{ranking.team}: {ranking.power_score:.2f}")
```

##### `update_rankings(new_games: List[GameData]) -> List[TeamRanking]`
Updates existing rankings with new game results.

**Parameters:**
- `new_games` (List[GameData]): New games to incorporate into rankings

**Returns:**
- `List[TeamRanking]`: Updated team rankings

### ESPN API Client

#### `power_ranking.api.espn_client.ESPNClient`

Client for fetching data from ESPN APIs.

```python
class ESPNClient:
    def __init__(self, rate_limit: float = 1.0, cache_enabled: bool = True)
    def get_games(self, season: int, week: int) -> List[GameData]
    def get_standings(self, season: int) -> List[TeamStanding]
    def get_schedule(self, season: int, team: str = None) -> List[ScheduleEntry]
```

**Methods:**

##### `get_games(season: int, week: int) -> List[GameData]`
Fetches game data for a specific season and week.

**Parameters:**
- `season` (int): NFL season year (e.g., 2024)
- `week` (int): Week number (1-18 for regular season, 19+ for playoffs)

**Returns:**
- `List[GameData]`: List of games with scores, teams, and metadata

**Raises:**
- `APIError`: When ESPN API request fails
- `DataValidationError`: When response data is invalid

**Example:**
```python
client = ESPNClient()
games = client.get_games(2024, 1)
```

### Data Validation

#### `power_ranking.validation.data_quality.DataQualityValidator`

Comprehensive data validation for power rankings inputs.

```python
class DataQualityValidator:
    def validate_game_data(self, games: List[GameData]) -> ValidationResult
    def validate_team_data(self, teams: List[Team]) -> ValidationResult
    def detect_anomalies(self, rankings: List[TeamRanking]) -> List[Anomaly]
```

**Methods:**

##### `validate_game_data(games: List[GameData]) -> ValidationResult`
Validates game data for completeness and consistency.

**Parameters:**
- `games` (List[GameData]): Games to validate

**Returns:**
- `ValidationResult`: Validation status with errors and warnings

**Example:**
```python
validator = DataQualityValidator()
result = validator.validate_game_data(games)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")
```

### Export System

#### `power_ranking.export.csv_exporter.CSVExporter`

Export power rankings to various formats.

```python
class CSVExporter:
    def export_rankings(self, rankings: List[TeamRanking], filepath: str) -> None
    def export_weekly_summary(self, rankings: List[TeamRanking], week: int, filepath: str) -> None
```

**Methods:**

##### `export_rankings(rankings: List[TeamRanking], filepath: str) -> None`
Exports complete rankings to CSV format.

**Parameters:**
- `rankings` (List[TeamRanking]): Rankings to export
- `filepath` (str): Output file path

**Example:**
```python
exporter = CSVExporter()
exporter.export_rankings(rankings, "week_1_rankings.csv")
```

### Memory Optimization (Phase 3 Integration)

#### `power_ranking.memory.memory_monitor.MemoryMonitor`

Real-time memory monitoring and optimization.

```python
class MemoryMonitor:
    def __init__(self, enable_tracemalloc: bool = True)
    def profile_memory(self, operation_name: str) -> ContextManager
    def get_memory_stats(self) -> Dict[str, Any]
```

**Methods:**

##### `profile_memory(operation_name: str) -> ContextManager`
Context manager for memory profiling operations.

**Parameters:**
- `operation_name` (str): Name of the operation being profiled

**Returns:**
- `ContextManager`: Context manager for memory profiling

**Example:**
```python
monitor = MemoryMonitor()
with monitor.profile_memory("ranking_calculation"):
    rankings = model.calculate_power_rankings(games)
stats = monitor.get_memory_stats()
```

#### `power_ranking.memory.data_streaming.DataStreamProcessor`

Memory-efficient data processing for large datasets.

```python
class DataStreamProcessor:
    def stream_csv_file(self, file_path: str) -> Iterator[Dict[str, Any]]
    def process_in_chunks(self, data: Iterator, processor: Callable, chunk_size: int = 1000) -> Iterator
```

## NFL Spread Model API

### Core Spread Model

#### `nfl_model.spread_model.SpreadCalculator`

Billy Walters methodology spread prediction engine.

```python
class SpreadCalculator:
    def __init__(self, home_field_advantage: float = 2.0)
    def calculate_spread(self, home_team: str, away_team: str, power_rankings: Dict[str, float]) -> float
    def predict_game(self, game: ScheduleEntry) -> SpreadPrediction
```

**Methods:**

##### `calculate_spread(home_team: str, away_team: str, power_rankings: Dict[str, float]) -> float`
Calculates point spread for a matchup using power rankings.

**Parameters:**
- `home_team` (str): Home team abbreviation
- `away_team` (str): Away team abbreviation  
- `power_rankings` (Dict[str, float]): Power rankings by team

**Returns:**
- `float`: Predicted point spread (positive = home team favored)

**Example:**
```python
calculator = SpreadCalculator(home_field_advantage=2.5)
spread = calculator.calculate_spread("KC", "BUF", power_rankings)
print(f"Spread: {spread:.1f}")
```

### Data Loading

#### `nfl_model.data_loader.DataLoader`

Loads and processes data for spread calculations.

```python
class DataLoader:
    def load_power_rankings(self) -> Dict[str, float]
    def load_schedule(self, week: int | None = None) -> pd.DataFrame
    def get_weekly_matchups(self, week: int) -> List[Tuple[str, str, str]]
    def validate_data_compatibility(self, rankings, matchups) -> Dict[str, Any]

def normalize_schedule_dataframe(df: pd.DataFrame) -> pd.DataFrame
```

**Methods:**

##### `load_power_rankings() -> Dict[str, float]`
Loads power rankings from CSV file.

**Parameters:**
- Provided at `DataLoader` initialization

**Returns:**
- `Dict[str, float]`: Power rankings by team abbreviation

**Raises:**
- `PowerRankingsLoadError`: When file cannot be loaded or parsed

### Configuration Management

#### `nfl_model.config_manager.ConfigManager`

Manages model configuration and parameters.

```python
class ConfigManager:
    def load_config(self, config_path: str = "config.yaml") -> ModelConfiguration
    def validate_config(self, config: ModelConfiguration) -> bool
```

## Common Data Structures

### GameData

```python
@dataclass
class GameData:
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    week: int
    season: int
    date: datetime
    status: GameStatus
```

### TeamRanking

```python
@dataclass
class TeamRanking:
    team: str
    power_score: float
    rank: int
    season_margin: Optional[float] = None
    strength_of_schedule: Optional[float] = None
```

### SpreadPrediction

```python
@dataclass
class SpreadPrediction:
    home_team: str
    away_team: str
    predicted_spread: float
    home_power: float
    away_power: float
    home_field_advantage: float
    confidence: Optional[float] = None
```

## Error Handling

### Power Rankings Exceptions

- **`APIError`**: ESPN API request failures
- **`DataValidationError`**: Invalid input data
- **`CalculationError`**: Power ranking calculation failures
- **`ExportError`**: File export failures

### NFL Spread Model Exceptions

- **`PowerRankingsLoadError`**: Power rankings file loading failures
- **`ScheduleLoadError`**: Schedule data loading failures
- **`SpreadCalculationError`**: Spread calculation failures
- **`ModelConfigurationError`**: Configuration validation failures

### Usage Examples

```python
from power_ranking.exceptions import APIError, DataValidationError

try:
    client = ESPNClient()
    games = client.get_games(2024, 1)
    
    validator = DataQualityValidator()
    result = validator.validate_game_data(games)
    
    if result.is_valid:
        model = PowerRankingsModel()
        rankings = model.calculate_power_rankings(games)
    else:
        print(f"Validation failed: {result.errors}")
        
except APIError as e:
    print(f"ESPN API error: {e}")
except DataValidationError as e:
    print(f"Data validation error: {e}")
```

## Rate Limiting and Performance

### ESPN API Rate Limiting

The ESPN client implements automatic rate limiting:
- Default: 1 request per second
- Configurable via `rate_limit` parameter
- Automatic retry with exponential backoff

### Memory Optimization

Phase 3 memory optimization features:
- Streaming data processing for large datasets
- Memory profiling and monitoring
- Lazy loading for efficient data access
- Optimized data structures with 30-50% memory savings

### Caching

Both systems implement intelligent caching:
- API response caching (configurable TTL)
- Computed ranking caching
- Memory-aware cache eviction

---

*This API reference is automatically generated and maintained alongside the codebase. For the latest updates, please refer to the inline documentation in the source code.*

##### Schedule CSV Schema
- Canonical columns: `week`, `home_team`, `away_team`, optional `game_date`
- Alternate schema supported: `home_team_name`/`away_team_name` are auto-normalized
- Use `normalize_schedule_dataframe(df)` to validate/normalize DataFrames
