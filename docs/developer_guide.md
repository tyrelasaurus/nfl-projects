# NFL Projects Developer Guide

This comprehensive guide provides everything developers need to contribute to the NFL Projects suite, including the Power Rankings System and NFL Spread Model.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Code Standards](#code-standards)
- [Contributing Guidelines](#contributing-guidelines)
- [Troubleshooting](#troubleshooting)

## Quick Start

Get up and running in under 5 minutes:

```bash
# Clone and setup
git clone <repository-url>
cd nfl-projects

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import power_ranking, nfl_model; print('✅ Installation successful')"

# Run basic tests
python -m pytest power_ranking/tests/
python -m pytest nfl_model/tests/
```

## Development Environment Setup

### Prerequisites

- **Python 3.12+** (recommended)
- **Git** for version control
- **pip** for package management
- **ESPN API access** (automatic, no key required)

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

**Core Dependencies:**
```txt
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0
requests>=2.31.0
python-dateutil>=2.8.2
```

**Development Dependencies:**
```txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
```

**Memory Optimization (Phase 3):**
```txt
psutil>=5.9.0
tracemalloc (built-in)
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["power_ranking/tests", "nfl_model/tests"]
}
```

## Project Architecture

### High-Level Overview

```
nfl-projects/
├── power_ranking/           # Power Rankings System
│   ├── api/                # ESPN API integration
│   ├── models/             # Core ranking algorithms
│   ├── validation/         # Data quality & validation
│   ├── export/             # Data export functionality
│   ├── memory/             # Memory optimization (Phase 3)
│   └── caching/            # Performance caching
├── nfl_model/              # NFL Spread Model
│   ├── validation/         # Model validation & backtesting
│   └── tests/              # Unit tests
├── docs/                   # Comprehensive documentation
└── tests/                  # Integration tests
```

### Design Patterns

#### 1. **Modular Architecture**
Each system is designed with clear separation of concerns:
- **Data Layer**: API clients and data loading
- **Business Logic**: Core algorithms and calculations
- **Validation Layer**: Data quality and error handling
- **Export Layer**: Output formatting and file generation

#### 2. **Dependency Injection**
Configuration and dependencies are injected rather than hardcoded:

```python
# Good: Dependency injection
class PowerRankingsModel:
    def __init__(self, config: ConfigManager, validator: DataValidator):
        self.config = config
        self.validator = validator

# Usage
config = ConfigManager()
validator = DataValidator()
model = PowerRankingsModel(config, validator)
```

#### 3. **Error Handling Strategy**
Structured exception hierarchy with specific error types:

```python
try:
    games = espn_client.get_games(2024, 1)
    rankings = model.calculate_power_rankings(games)
except APIError as e:
    logger.error(f"ESPN API failed: {e}")
except DataValidationError as e:
    logger.error(f"Invalid data: {e}")
except CalculationError as e:
    logger.error(f"Ranking calculation failed: {e}")
```

### Data Flow Architecture

```
ESPN API → Data Validation → Power Rankings → Export
    ↓             ↓              ↓           ↓
  Cache       Quality Assurance  Memory    CSV/JSON
            Phase 1.3 Enhanced  Optimization Files
```

## Development Workflow

### 1. **Feature Development**

```bash
# Create feature branch
git checkout -b feature/new-ranking-algorithm

# Make changes with tests
# ... development work ...

# Run tests
python -m pytest power_ranking/tests/ -v

# Check code quality
black power_ranking/
flake8 power_ranking/
mypy power_ranking/

# Commit with descriptive message
git commit -m "Add enhanced ranking algorithm with SOS weighting"
```

### 2. **Code Review Process**

Before submitting changes:

1. **Self-Review Checklist:**
   - [ ] All tests pass
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No debug code left
   - [ ] Performance impact considered

2. **Testing Requirements:**
   - [ ] Unit tests for new functions
   - [ ] Integration tests for system changes
   - [ ] Performance tests for memory-intensive changes

### 3. **Memory Optimization (Phase 3 Integration)**

When working on memory-intensive features:

```python
from power_ranking.memory.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()

with monitor.profile_memory("ranking_calculation"):
    rankings = model.calculate_power_rankings(large_dataset)

# Check memory usage
stats = monitor.get_memory_stats()
print(f"Peak memory: {stats['peak_memory_mb']:.2f}MB")
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                   # Unit tests for individual functions
├── integration/            # Integration tests across modules
├── performance/            # Performance and memory tests
└── fixtures/               # Test data and mocks
```

### Writing Unit Tests

```python
import pytest
from power_ranking.models.power_rankings import PowerRankingsModel

class TestPowerRankingsModel:
    
    @pytest.fixture
    def sample_games(self):
        return [
            GameData(home_team="KC", away_team="BUF", 
                    home_score=24, away_score=20, week=1),
            # ... more test data
        ]
    
    @pytest.fixture
    def model(self):
        return PowerRankingsModel()
    
    def test_calculate_power_rankings_basic(self, model, sample_games):
        rankings = model.calculate_power_rankings(sample_games)
        
        assert len(rankings) == 32  # All NFL teams
        assert all(r.power_score >= -50 and r.power_score <= 50 for r in rankings)
        assert rankings[0].rank == 1  # Top team has rank 1
        
    def test_calculate_power_rankings_empty_input(self, model):
        with pytest.raises(DataValidationError):
            model.calculate_power_rankings([])
```

### Performance Testing

```python
def test_memory_usage_large_dataset():
    monitor = MemoryMonitor()
    
    with monitor.profile_memory("large_dataset_test"):
        # Process large dataset
        results = process_season_data(large_games_list)
    
    stats = monitor.get_memory_stats()
    assert stats['peak_memory_mb'] < 500  # Memory limit
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest power_ranking/tests/test_power_rankings.py

# Run with coverage
python -m pytest --cov=power_ranking --cov-report=html

# Run performance tests
python -m pytest tests/performance/ -v
```

## Code Standards

### Python Style Guide

We follow **PEP 8** with these specific guidelines:

#### 1. **Code Formatting**
```python
# Use Black formatter (line length: 88 characters)
black power_ranking/ nfl_model/

# Configuration in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py312']
```

#### 2. **Type Hints**
Always use type hints for public functions:

```python
from typing import List, Dict, Optional, Union
from datetime import datetime

def calculate_power_rankings(
    games: List[GameData], 
    config: Optional[Dict[str, Any]] = None
) -> List[TeamRanking]:
    """Calculate power rankings from game data."""
    pass
```

#### 3. **Documentation Standards**

**Class Documentation:**
```python
class PowerRankingsModel:
    """
    Power rankings calculation engine using advanced statistical methods.
    
    This model implements a sophisticated ranking algorithm that considers:
    - Margin of victory with diminishing returns
    - Strength of schedule adjustments  
    - Recent performance weighting
    - Home field advantage normalization
    
    Attributes:
        config: Model configuration parameters
        validator: Data validation instance
        
    Example:
        >>> model = PowerRankingsModel()
        >>> rankings = model.calculate_power_rankings(games)
        >>> print(f"Top team: {rankings[0].team}")
    """
```

**Function Documentation:**
```python
def calculate_spread(
    home_team: str, 
    away_team: str, 
    power_rankings: Dict[str, float]
) -> float:
    """
    Calculate point spread using Billy Walters methodology.
    
    Args:
        home_team: Home team abbreviation (e.g., "KC")
        away_team: Away team abbreviation (e.g., "BUF")  
        power_rankings: Power rankings by team abbreviation
        
    Returns:
        Point spread with home team perspective (positive = home favored)
        
    Raises:
        SpreadCalculationError: When teams not found in rankings
        
    Example:
        >>> spread = calculate_spread("KC", "BUF", rankings)
        >>> print(f"KC favored by {spread:.1f}")
    """
```

#### 4. **Error Handling**

Use specific exception types:

```python
# Good: Specific exceptions
try:
    data = espn_client.get_games(2024, 1)
except APIError as e:
    logger.error(f"ESPN API failed: {e}")
    raise DataLoadingError(f"Could not load games: {e}") from e

# Bad: Generic exceptions  
try:
    data = espn_client.get_games(2024, 1)
except Exception as e:
    print(f"Something went wrong: {e}")
```

#### 5. **Configuration Management**

Use centralized configuration:

```python
# Good: Centralized config
from power_ranking.config_manager import ConfigManager

config = ConfigManager()
home_field_advantage = config.get("home_field_advantage", default=2.0)

# Bad: Hardcoded values
home_field_advantage = 2.0  # Magic number
```

### Memory Optimization Guidelines (Phase 3)

#### 1. **Use Memory-Efficient Data Structures**

```python
from power_ranking.memory.optimized_structures import CompactGameRecord

# Good: Memory-optimized structures
games = [CompactGameRecord(game_data) for game_data in raw_games]

# Monitor memory usage
with MemoryMonitor().profile_memory("data_processing"):
    results = process_games(games)
```

#### 2. **Implement Streaming for Large Datasets**

```python
from power_ranking.memory.data_streaming import DataStreamProcessor

processor = DataStreamProcessor()

# Stream large CSV files
for row in processor.stream_csv_file("large_dataset.csv"):
    process_row(row)

# Batch processing
for batch in processor.process_in_chunks(data_iterator, batch_processor):
    handle_batch_results(batch)
```

## Contributing Guidelines

### 1. **Code Contribution Process**

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** changes with tests
4. **Update** documentation
5. **Submit** pull request

### 2. **Pull Request Requirements**

- [ ] **All tests pass** (unit, integration, performance)
- [ ] **Code coverage** maintained or improved
- [ ] **Documentation updated** for new features
- [ ] **Type hints** provided for public functions
- [ ] **Memory impact** assessed for Phase 3 integration

### 3. **Documentation Contributions**

When updating documentation:

- Follow Markdown standards
- Include working code examples
- Update API reference for new functions
- Test all code examples

### 4. **Performance Considerations**

- **Memory Usage**: Monitor memory consumption for large datasets
- **API Rate Limits**: Respect ESPN API rate limiting
- **Caching**: Leverage existing caching mechanisms
- **Streaming**: Use streaming for datasets > 1000 records

## Troubleshooting

### Common Issues

#### 1. **ESPN API Failures**

```python
# Issue: Rate limiting or API downtime
# Solution: Implement retry logic and caching

from power_ranking.api.espn_client import ESPNClient

client = ESPNClient(rate_limit=2.0, cache_enabled=True)
try:
    games = client.get_games(2024, 1)
except APIError as e:
    logger.warning(f"API failed, using cached data: {e}")
    games = client.get_cached_games(2024, 1)
```

#### 2. **Memory Issues with Large Datasets**

```python
# Issue: Memory exhaustion with large datasets
# Solution: Use Phase 3 streaming and optimization

from power_ranking.memory.data_streaming import DataStreamProcessor
from power_ranking.memory.memory_monitor import MemoryMonitor

processor = DataStreamProcessor()
monitor = MemoryMonitor()

# Stream instead of loading all at once
with monitor.profile_memory("large_data_processing"):
    for chunk in processor.stream_csv_file("huge_dataset.csv"):
        process_chunk(chunk)
```

#### 3. **Data Validation Failures**

```python
# Issue: Invalid or inconsistent data
# Solution: Use Phase 1.3 enhanced validation

from power_ranking.validation.data_quality import DataQualityValidator

validator = DataQualityValidator()
result = validator.validate_game_data(games)

if not result.is_valid:
    # Handle validation errors
    for error in result.errors:
        logger.error(f"Validation error: {error}")
    
    # Use data cleaning
    cleaned_games = validator.clean_game_data(games)
```

#### 4. **Performance Optimization**

```python
# Monitor and optimize performance
from power_ranking.memory.memory_profiler import AdvancedMemoryProfiler

profiler = AdvancedMemoryProfiler()

@profiler.profile_function_detailed()
def calculate_rankings(games):
    # Your function implementation
    pass

# Get optimization recommendations  
recommendations = profiler.get_optimization_recommendations()
```

### Development Environment Issues

#### Python Version Compatibility
```bash
# Check Python version
python --version  # Should be 3.12+

# If version mismatch, use pyenv
pyenv install 3.12.0
pyenv local 3.12.0
```

#### Dependency Conflicts
```bash
# Clean install
pip freeze > requirements_backup.txt
pip uninstall -r requirements_backup.txt -y
pip install -r requirements.txt
```

### Getting Help

1. **Check Documentation**: Review this guide and API reference
2. **Search Issues**: Look for similar problems in project issues
3. **Run Diagnostics**: Use built-in diagnostic tools
4. **Create Issue**: Provide detailed error information and steps to reproduce

### Diagnostic Tools

```python
# System diagnostic
from power_ranking.config_manager import ConfigManager
from power_ranking.validation.data_monitoring import SystemDiagnostics

config = ConfigManager()
diagnostics = SystemDiagnostics()

# Run full system check
status = diagnostics.run_system_check()
print(f"System status: {status}")
```

---

This developer guide is maintained alongside the codebase. For the latest updates and advanced topics, please refer to the specific module documentation and inline code comments.

**Happy coding! 🏈📊**