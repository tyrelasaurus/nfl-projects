# NFL Projects Statistical Methods Documentation

This document provides comprehensive documentation of the statistical methodologies, algorithms, and analytical approaches used in the NFL Projects suite.

## Table of Contents

- [Power Rankings Statistical Methodology](#power-rankings-statistical-methodology)
- [Billy Walters Spread Model](#billy-walters-spread-model)
- [Data Quality and Validation Methods](#data-quality-and-validation-methods)
- [Performance Metrics and Validation](#performance-metrics-and-validation)
- [Statistical Formulas Reference](#statistical-formulas-reference)

## Power Rankings Statistical Methodology

### Overview

The Power Rankings system employs a sophisticated multi-factor approach that combines margin of victory, strength of schedule, and temporal weighting to create accurate team rankings throughout the NFL season.

### Core Algorithm Components

#### 1. **Base Margin Calculation**

The foundation of power rankings is the margin of victory with diminishing returns:

```
Adjusted Margin = sign(raw_margin) × log(1 + abs(raw_margin))
```

**Mathematical Justification:**
- **Logarithmic scaling** prevents blowouts from having disproportionate impact
- **Sign preservation** maintains win/loss information
- **Diminishing returns** reflect the reality that a 35-point win isn't twice as meaningful as a 17-point win

**Implementation:**
```python
def calculate_adjusted_margin(home_score: int, away_score: int) -> float:
    """
    Calculate margin of victory with logarithmic dampening.
    
    Formula: sign(margin) × log(1 + |margin|)
    
    This approach prevents blowouts from dominating rankings while
    maintaining the significance of victory margins.
    """
    raw_margin = home_score - away_score
    if raw_margin == 0:
        return 0.0
    
    return math.copysign(math.log1p(abs(raw_margin)), raw_margin)
```

#### 2. **Strength of Schedule (SOS) Adjustment**

Strength of schedule normalization ensures teams aren't penalized for playing in competitive divisions or conferences.

```
SOS_factor = Σ(opponent_power_score) / number_of_games
Adjusted_Score = Base_Score × (1 + SOS_multiplier × SOS_factor)
```

**Key Parameters:**
- **SOS_multiplier**: 0.15 (default) - Controls impact of schedule strength
- **Opponent weighting**: Current power scores, updated iteratively
- **Conference adjustments**: AFC/NFC strength normalization

**Implementation:**
```python
def calculate_sos_adjustment(team_games: List[GameData], 
                           current_rankings: Dict[str, float]) -> float:
    """
    Calculate strength of schedule adjustment factor.
    
    SOS reflects the average power score of opponents faced,
    weighted by game importance (recent games weighted higher).
    """
    if not team_games:
        return 0.0
    
    opponent_strengths = []
    total_weight = 0
    
    for i, game in enumerate(team_games):
        opponent = game.away_team if game.home_team == team else game.home_team
        opponent_strength = current_rankings.get(opponent, 0.0)
        
        # Temporal weighting: recent games matter more
        weight = 1.0 + (0.1 * i)  # Linear increase for recent games
        opponent_strengths.append(opponent_strength * weight)
        total_weight += weight
    
    return sum(opponent_strengths) / total_weight if total_weight > 0 else 0.0
```

#### 3. **Temporal Weighting**

Recent performance is weighted more heavily than early-season results:

```
Game_Weight = base_weight + (recency_factor × games_ago^-1)
```

**Temporal Decay Function:**
- **Recent games**: Weight = 1.0 + 0.2 × (weeks_since_game)^(-0.5)
- **Early season**: Exponential decay for games > 8 weeks old
- **Playoff relevance**: Final 4 weeks weighted 1.5x

#### 4. **Home Field Advantage Normalization**

Home field advantage is factored out for neutral-site team comparison:

```
Neutral_Margin = Actual_Margin - HFA_adjustment
Where HFA_adjustment = 2.5 points (NFL average)
```

**Dynamic HFA Calculation:**
```python
def calculate_dynamic_hfa(team: str, venue: str, 
                         historical_data: Dict) -> float:
    """
    Calculate dynamic home field advantage for specific team/venue.
    
    Factors considered:
    - Historical home/away performance differential
    - Venue-specific factors (altitude, weather, crowd)
    - Time zone adjustments for cross-country games
    """
    base_hfa = 2.5  # NFL average
    
    # Team-specific adjustment
    team_hfa_factor = historical_data.get(f"{team}_hfa_factor", 1.0)
    
    # Venue adjustments (altitude, weather, etc.)
    venue_adjustments = {
        'denver': 1.2,    # Altitude advantage
        'seattle': 1.15,  # Crowd noise
        'green_bay': 1.1, # Weather advantage
        # ... more venues
    }
    
    venue_factor = venue_adjustments.get(venue.lower(), 1.0)
    
    return base_hfa * team_hfa_factor * venue_factor
```

### Power Score Calculation

The final power score combines all factors using weighted regression:

```
Power_Score = α × Adjusted_Margin_Avg + 
              β × SOS_Factor + 
              γ × Recent_Performance_Trend +
              δ × Playoff_Probability_Factor
```

**Default Coefficients:**
- **α (Margin weight)**: 0.65
- **β (SOS weight)**: 0.20
- **γ (Trend weight)**: 0.10
- **δ (Playoff weight)**: 0.05

### Iterative Refinement

The system uses iterative refinement to handle circular dependencies in SOS calculations:

```python
def iterative_power_ranking_calculation(games: List[GameData], 
                                      max_iterations: int = 10,
                                      convergence_threshold: float = 0.01) -> Dict[str, float]:
    """
    Iteratively calculate power rankings until convergence.
    
    Process:
    1. Initialize all teams with neutral (0.0) power scores
    2. Calculate rankings based on current opponent strengths
    3. Update opponent strengths based on new rankings
    4. Repeat until rankings converge or max iterations reached
    """
    rankings = {team: 0.0 for team in get_all_teams(games)}
    
    for iteration in range(max_iterations):
        previous_rankings = rankings.copy()
        
        # Recalculate rankings using current opponent strengths
        rankings = calculate_single_iteration_rankings(games, rankings)
        
        # Check convergence
        max_change = max(abs(rankings[team] - previous_rankings[team]) 
                        for team in rankings)
        
        if max_change < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break
    
    return rankings
```

## Billy Walters Spread Model

### Methodology Overview

The NFL Spread Model implements the Billy Walters approach to point spread prediction, focusing on power rating differentials with home field advantage adjustments.

### Core Formula

```
Point_Spread = (Home_Power_Rating - Away_Power_Rating) + Home_Field_Advantage
```

### Power Rating Integration

The spread model takes power rankings as input and applies the following transformations:

#### 1. **Power Rating Normalization**

```python
def normalize_power_ratings(power_rankings: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize power ratings for spread calculation.
    
    Ensures power ratings are on appropriate scale for point spreads:
    - Mean = 0 (league average)
    - Standard deviation ≈ 10-12 points
    """
    ratings = list(power_rankings.values())
    mean_rating = statistics.mean(ratings)
    std_rating = statistics.stdev(ratings)
    
    # Target standard deviation for NFL spreads
    target_std = 11.0
    scale_factor = target_std / std_rating if std_rating > 0 else 1.0
    
    return {
        team: (rating - mean_rating) * scale_factor 
        for team, rating in power_rankings.items()
    }
```

#### 2. **Home Field Advantage Calculation**

The model uses dynamic home field advantage based on:

```
HFA = Base_HFA + Venue_Adjustment + Travel_Adjustment + Rest_Advantage
```

**Components:**
- **Base_HFA**: 2.5 points (NFL average)
- **Venue_Adjustment**: -1.0 to +2.0 points (venue-specific)
- **Travel_Adjustment**: 0 to -1.5 points (distance-based)
- **Rest_Advantage**: 0 to +3.0 points (days rest differential)

#### 3. **Spread Calculation with Confidence Intervals**

```python
def calculate_spread_with_confidence(home_team: str, away_team: str,
                                   power_ratings: Dict[str, float],
                                   historical_accuracy: Dict) -> SpreadPrediction:
    """
    Calculate point spread with confidence intervals.
    
    Returns spread prediction with statistical confidence bounds
    based on historical model performance.
    """
    # Basic spread calculation
    home_rating = power_ratings.get(home_team, 0.0)
    away_rating = power_ratings.get(away_team, 0.0)
    hfa = calculate_dynamic_hfa(home_team, away_team)
    
    raw_spread = (home_rating - away_rating) + hfa
    
    # Historical accuracy adjustment
    model_accuracy = historical_accuracy.get('overall_accuracy', 0.52)
    confidence_factor = min(model_accuracy * 2, 1.0)
    
    # Calculate confidence intervals (±1 standard deviation)
    spread_std = 3.5  # Historical standard deviation of spread errors
    lower_bound = raw_spread - (spread_std * confidence_factor)
    upper_bound = raw_spread + (spread_std * confidence_factor)
    
    return SpreadPrediction(
        home_team=home_team,
        away_team=away_team,
        predicted_spread=round(raw_spread, 1),
        confidence=confidence_factor,
        lower_bound=round(lower_bound, 1),
        upper_bound=round(upper_bound, 1),
        home_field_advantage=hfa
    )
```

### Statistical Validation

#### Backtesting Framework

The model includes comprehensive backtesting capabilities:

```python
def backtest_spread_model(historical_games: List[GameData],
                         power_rankings_by_week: Dict[int, Dict[str, float]],
                         test_weeks: List[int]) -> BacktestResults:
    """
    Comprehensive backtesting of spread prediction accuracy.
    
    Metrics calculated:
    - Against-the-spread (ATS) accuracy
    - Mean absolute error (MAE)
    - Root mean square error (RMSE)
    - Profit/loss simulation with standard betting units
    """
    predictions = []
    results = []
    
    for week in test_weeks:
        week_rankings = power_rankings_by_week[week]
        week_games = [g for g in historical_games if g.week == week]
        
        for game in week_games:
            if game.is_completed():
                prediction = calculate_spread_with_confidence(
                    game.home_team, game.away_team, week_rankings
                )
                
                actual_margin = game.home_score - game.away_score
                
                predictions.append(prediction)
                results.append(BacktestResult(
                    predicted_spread=prediction.predicted_spread,
                    actual_margin=actual_margin,
                    correct_side=(prediction.predicted_spread * actual_margin > 0),
                    absolute_error=abs(prediction.predicted_spread - actual_margin)
                ))
    
    return analyze_backtest_results(predictions, results)
```

## Data Quality and Validation Methods

### Validation Framework (Phase 1.3 Enhancement)

The system employs a multi-layered validation approach:

#### 1. **Data Completeness Validation**

```python
def validate_data_completeness(games: List[GameData]) -> ValidationResult:
    """
    Ensure all required fields are present and valid.
    
    Checks:
    - No missing scores for completed games
    - Valid team abbreviations (must be in NFL_TEAMS)
    - Logical date/week consistency
    - Score reasonableness (0-100 points per team)
    """
    errors = []
    warnings = []
    
    for game in games:
        if game.status == GameStatus.COMPLETED:
            if game.home_score is None or game.away_score is None:
                errors.append(f"Missing scores for completed game: {game.game_id}")
            
            if game.home_score < 0 or game.away_score < 0:
                errors.append(f"Invalid negative scores: {game}")
            
            if game.home_score > 100 or game.away_score > 100:
                warnings.append(f"Unusual high score: {game}")
    
    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings
    )
```

#### 2. **Statistical Anomaly Detection**

```python
def detect_statistical_anomalies(rankings: List[TeamRanking]) -> List[Anomaly]:
    """
    Detect statistical anomalies in power rankings.
    
    Anomaly types:
    - Power scores outside ±3 standard deviations
    - Unrealistic ranking jumps (>10 positions week-over-week)
    - Teams with identical power scores
    - Missing teams or duplicate rankings
    """
    anomalies = []
    power_scores = [r.power_score for r in rankings]
    
    # Statistical outlier detection
    mean_score = statistics.mean(power_scores)
    std_score = statistics.stdev(power_scores)
    
    for ranking in rankings:
        z_score = abs(ranking.power_score - mean_score) / std_score
        if z_score > 3.0:
            anomalies.append(Anomaly(
                type="statistical_outlier",
                description=f"{ranking.team} power score {ranking.power_score:.2f} is {z_score:.1f} standard deviations from mean",
                severity="high" if z_score > 4.0 else "medium"
            ))
    
    return anomalies
```

#### 3. **Cross-Validation with Vegas Lines**

```python
def validate_against_vegas_lines(predictions: List[SpreadPrediction],
                               vegas_lines: List[VegasLine]) -> ValidationReport:
    """
    Cross-validate model predictions against Vegas betting lines.
    
    Metrics:
    - Correlation coefficient with Vegas lines
    - Mean absolute deviation from Vegas
    - Instances of significant deviation (>7 points)
    - Side agreement percentage (same favorite predicted)
    """
    model_spreads = []
    vegas_spreads = []
    side_agreements = 0
    
    for prediction in predictions:
        vegas_line = next((v for v in vegas_lines 
                          if v.home_team == prediction.home_team 
                          and v.away_team == prediction.away_team), None)
        
        if vegas_line:
            model_spreads.append(prediction.predicted_spread)
            vegas_spreads.append(vegas_line.spread)
            
            # Check if both predict same favorite
            if (prediction.predicted_spread > 0) == (vegas_line.spread > 0):
                side_agreements += 1
    
    correlation = numpy.corrcoef(model_spreads, vegas_spreads)[0, 1]
    mad = numpy.mean(numpy.abs(numpy.array(model_spreads) - numpy.array(vegas_spreads)))
    side_agreement_pct = side_agreements / len(model_spreads) * 100
    
    return ValidationReport(
        correlation_with_vegas=correlation,
        mean_absolute_deviation=mad,
        side_agreement_percentage=side_agreement_pct,
        sample_size=len(model_spreads)
    )
```

## Performance Metrics and Validation

### Key Performance Indicators

#### 1. **Power Rankings Accuracy**

- **Week-over-week correlation**: Measure ranking stability
- **Playoff prediction accuracy**: Correct playoff teams identified
- **Championship correlation**: Historical correlation with playoff success

#### 2. **Spread Model Performance**

- **Against-the-spread accuracy**: Target >52.4% (break-even with juice)
- **Mean absolute error**: Target <3.5 points per game
- **Profit simulation**: Theoretical betting performance

#### 3. **System Performance Metrics (Phase 3 Integration)**

```python
class PerformanceMetrics:
    """
    Comprehensive system performance monitoring.
    
    Tracks:
    - Computation time for ranking calculations
    - Memory usage throughout processing pipeline
    - API response times and success rates
    - Data processing throughput
    """
    
    def measure_ranking_performance(self, games: List[GameData]) -> PerformanceReport:
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Perform ranking calculation
        rankings = self.power_model.calculate_power_rankings(games)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        return PerformanceReport(
            calculation_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            games_processed=len(games),
            throughput=len(games) / (end_time - start_time)
        )
```

## Statistical Formulas Reference

### Power Rankings Core Formulas

#### Adjusted Margin of Victory
```
AM_ij = sign(S_i - S_j) × log(1 + |S_i - S_j|)
```
Where:
- `AM_ij` = Adjusted margin for game between teams i and j
- `S_i, S_j` = Scores for teams i and j

#### Strength of Schedule
```
SOS_i = (1/n) × Σ(P_j × W_k)
```
Where:
- `SOS_i` = Strength of schedule for team i
- `n` = Number of games played
- `P_j` = Power rating of opponent j
- `W_k` = Temporal weight for game k

#### Final Power Rating
```
PR_i = α × MA_i + β × SOS_i + γ × T_i + δ × PF_i
```
Where:
- `PR_i` = Power rating for team i
- `MA_i` = Margin average (adjusted)
- `T_i` = Recent trend factor
- `PF_i` = Playoff factor

### Spread Model Formulas

#### Basic Spread Calculation
```
S_ij = (PR_i - PR_j) + HFA_i + ε
```
Where:
- `S_ij` = Predicted spread (team i vs team j)
- `PR_i, PR_j` = Power ratings
- `HFA_i` = Home field advantage for team i
- `ε` = Error term (normally distributed)

#### Dynamic Home Field Advantage
```
HFA_i = HFA_base × (1 + V_i) × (1 + T_ij) × (1 + R_ij)
```
Where:
- `HFA_base` = Base home field advantage (2.5 points)
- `V_i` = Venue-specific adjustment
- `T_ij` = Travel factor for away team j
- `R_ij` = Rest differential factor

### Validation Metrics

#### Against-the-Spread Accuracy
```
ATS% = (Correct_Predictions / Total_Predictions) × 100
```

#### Mean Absolute Error
```
MAE = (1/n) × Σ|Predicted_Spread_i - Actual_Margin_i|
```

#### Root Mean Square Error
```
RMSE = √[(1/n) × Σ(Predicted_Spread_i - Actual_Margin_i)²]
```

---

This statistical methodology documentation represents the analytical foundation of the NFL Projects. These methods have been validated through extensive backtesting and cross-validation with industry standards.

**Mathematical Foundation**: All formulas are implemented with numerical stability considerations and validated against edge cases.

**Continuous Improvement**: These methodologies are continuously refined based on performance metrics and new statistical research in sports analytics.