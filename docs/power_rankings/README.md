# Power Rankings System Documentation

This directory contains documentation specific to the NFL Power Rankings System.

## System Overview

The Power Rankings System is an advanced statistical engine that generates comprehensive NFL team rankings based on game performance, strength of schedule, and contextual factors.

### Key Features

- **Multi-factor Analysis**: Combines season performance, recent trends, and strength of schedule
- **ESPN API Integration**: Real-time data collection from ESPN NFL APIs
- **Memory Optimization**: Phase 3 integration for efficient large dataset processing
- **Export Flexibility**: CSV, JSON, and custom format support
- **Validation Framework**: Comprehensive data quality assurance (Phase 1.3)

### Architecture Components

```
Power Rankings System
├── API Layer (power_ranking.api.*)
│   ├── ESPN Client with rate limiting
│   ├── Async processing capabilities
│   └── Performance monitoring
├── Models Layer (power_ranking.models.*)
│   ├── Core ranking algorithms
│   ├── Statistical calculations
│   └── Confidence intervals
├── Validation Layer (power_ranking.validation.*)
│   ├── Data quality assurance
│   ├── Anomaly detection
│   └── Cross-validation
├── Export Layer (power_ranking.export.*)
│   ├── CSV export with metadata
│   ├── JSON structured output
│   └── Custom format support
└── Memory Layer (power_ranking.memory.*)
    ├── Streaming data processing
    ├── Memory monitoring
    └── Performance optimization
```

## Quick Start

### Basic Usage

```python
from power_ranking.models.power_rankings import PowerRankModel
from power_ranking.api.espn_client import ESPNClient

# Initialize components
client = ESPNClient()
model = PowerRankModel()

# Fetch current week data
scoreboard = client.get_scoreboard_data(2024, 1)
teams = client.get_teams_data()

# Calculate rankings
rankings, data = model.compute(scoreboard, teams)

# Display results
for rank, (team_id, team_name, score) in enumerate(rankings, 1):
    print(f"{rank:2d}. {team_name:25} {score:6.2f}")
```

### Memory-Optimized Processing

```python
from power_ranking.memory.memory_monitor import MemoryMonitor
from power_ranking.memory.data_streaming import DataStreamProcessor

# Enable memory monitoring
monitor = MemoryMonitor()
processor = DataStreamProcessor()

# Process large datasets efficiently
with monitor.profile_memory("full_season_analysis"):
    for chunk in processor.stream_csv_file("season_data.csv"):
        # Process in memory-efficient chunks
        results = model.process_chunk(chunk)
```

## Configuration

### Standard Configuration

```yaml
# power_rankings_config.yaml
power_rankings:
  # Weighting factors
  weights:
    season_avg_margin: 0.45    # Primary performance factor
    rolling_avg_margin: 0.30   # Recent trend emphasis
    strength_of_schedule: 0.20  # Opponent quality adjustment
    recency_factor: 0.05       # Temporal weighting

  # Algorithm parameters  
  rolling_window: 8           # Games for rolling averages
  week18_weight: 0.5          # Reduced Week 18 impact
  home_field_advantage: 2.5   # Standard NFL home advantage

  # Convergence settings
  max_iterations: 10          # SOS calculation iterations
  convergence_threshold: 0.01 # Ranking stability threshold

# Memory optimization (Phase 3)
performance:
  memory_monitoring: true
  streaming_chunk_size: 1000
  enable_gc_optimization: true
```

## Statistical Methodology

### Power Score Calculation

The system uses a multi-component weighted approach:

```
Power_Score = α × Season_Margin + β × Rolling_Margin + γ × SOS + δ × Recency
```

Where:
- **α = 0.45**: Season average margin weight
- **β = 0.30**: Rolling average margin weight  
- **γ = 0.20**: Strength of schedule weight
- **δ = 0.05**: Recency factor weight

### Margin Calculation

Margins are calculated with logarithmic dampening to prevent blowout bias:

```python
def calculate_adjusted_margin(home_score, away_score):
    raw_margin = home_score - away_score
    return math.copysign(math.log1p(abs(raw_margin)), raw_margin)
```

### Strength of Schedule

SOS calculation uses iterative refinement to handle circular dependencies:

```python
def calculate_sos(team_games, current_rankings):
    opponent_strengths = []
    for game in team_games:
        opponent = get_opponent(game, team)
        strength = current_rankings.get(opponent, 0.0)
        opponent_strengths.append(strength)
    
    return sum(opponent_strengths) / len(opponent_strengths)
```

## API Reference

### Core Classes

#### PowerRankModel

Primary ranking calculation engine.

```python
class PowerRankModel:
    def __init__(self, weights=None, config=None)
    def compute(self, scoreboard_data, teams_info) -> Tuple[List, Dict]
```

#### ESPNClient  

ESPN API integration with rate limiting.

```python
class ESPNClient:
    def __init__(self, rate_limit=1.0, cache_enabled=True)
    def get_scoreboard_data(self, season, week) -> Dict
    def get_teams_data() -> List[Dict]
```

### Memory Optimization Classes (Phase 3)

#### MemoryMonitor

Real-time memory usage monitoring.

```python
class MemoryMonitor:
    def profile_memory(self, operation_name) -> ContextManager
    def get_memory_stats() -> Dict[str, Any]
```

#### DataStreamProcessor

Memory-efficient data processing.

```python
class DataStreamProcessor:
    def stream_csv_file(self, filepath) -> Iterator[Dict]
    def process_in_chunks(self, data, processor) -> Iterator
```

## Data Validation (Phase 1.3 Integration)

### Quality Assurance Framework

```python
from power_ranking.validation.data_quality import DataQualityValidator

validator = DataQualityValidator()

# Comprehensive validation
result = validator.validate_game_data(games)
if not result.is_valid:
    for error in result.errors:
        logger.error(f"Validation failed: {error}")

# Anomaly detection
anomalies = validator.detect_anomalies(rankings)
for anomaly in anomalies:
    logger.warning(f"Anomaly detected: {anomaly.description}")
```

### Validation Metrics

- **Data Completeness**: All required fields present
- **Statistical Consistency**: Scores within reasonable ranges
- **Temporal Logic**: Game dates and weeks align properly
- **Cross-Validation**: Results compared with historical patterns

## Performance Optimization

### Memory Usage (Phase 3 Features)

The system implements comprehensive memory optimization:

```python
# Streaming for large datasets
processor = DataStreamProcessor()
for chunk in processor.stream_csv_file("large_season_data.csv"):
    results = process_chunk(chunk)

# Memory monitoring
with MemoryMonitor().profile_memory("ranking_calculation"):
    rankings = model.compute(data, teams)
    
# Check memory usage
stats = monitor.get_memory_stats()
print(f"Peak memory: {stats['peak_memory_mb']:.1f}MB")
```

### Caching Strategy

- **API Response Caching**: 1-hour TTL for ESPN data
- **Computation Caching**: Rankings cached until new data available
- **Memory-Aware Eviction**: Automatic cache cleanup under memory pressure

## Export Formats

### CSV Export

Standard comma-separated format with comprehensive metadata:

```csv
Rank,Team,Power_Score,Season_Margin,Rolling_Margin,SOS,Games_Played,Last_Updated
1,Kansas City Chiefs,12.45,8.2,9.1,2.3,17,2024-01-15T10:30:00Z
2,Buffalo Bills,11.87,7.8,8.9,1.9,17,2024-01-15T10:30:00Z
```

### JSON Export

Structured format with detailed breakdowns:

```json
{
  "rankings": [
    {
      "rank": 1,
      "team": "Kansas City Chiefs", 
      "power_score": 12.45,
      "components": {
        "season_margin": 8.2,
        "rolling_margin": 9.1,
        "strength_of_schedule": 2.3,
        "recency_factor": 0.15
      },
      "metadata": {
        "games_played": 17,
        "confidence_interval": [11.8, 13.1]
      }
    }
  ],
  "generated_at": "2024-01-15T10:30:00Z",
  "model_version": "1.0.0"
}
```

## Testing

### Unit Tests

```bash
# Run power rankings tests
python -m pytest power_ranking/tests/ -v

# Test specific components
python -m pytest power_ranking/tests/test_power_rankings.py::test_basic_calculation

# Memory optimization tests
python -m pytest power_ranking/tests/test_memory_optimization.py
```

### Integration Tests

```bash
# Full system integration test
python -m pytest tests/integration/test_full_pipeline.py

# API integration tests
python -m pytest power_ranking/tests/test_espn_client.py
```

## Troubleshooting

### Common Issues

1. **Memory Usage High**
   - Enable streaming for large datasets
   - Reduce chunk sizes in configuration
   - Monitor memory usage with Phase 3 tools

2. **ESPN API Failures**
   - Check rate limiting configuration  
   - Enable caching to reduce API calls
   - Verify network connectivity

3. **Ranking Instability**
   - Increase convergence iterations
   - Adjust weighting factors
   - Check for data quality issues

### Diagnostic Tools

```python
# System health check
from power_ranking.validation.data_monitoring import SystemDiagnostics

diagnostics = SystemDiagnostics()
status = diagnostics.run_system_check()
print(f"System status: {status}")

# Performance analysis
from power_ranking.memory.memory_profiler import AdvancedMemoryProfiler

profiler = AdvancedMemoryProfiler()
recommendations = profiler.get_optimization_recommendations()
```

---

For detailed API documentation, see the [main API reference](../api_reference.md).
For statistical methodology details, see [Statistical Methods](../statistical_methods.md).
For deployment procedures, see the [Deployment Guide](../deployment.md).