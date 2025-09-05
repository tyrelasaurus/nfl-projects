# NFL Matchup Point Spread Model — PRD

## 1. Overview

**Objective**  
Develop a Python-based application that takes two teams’ preseason (or current) power ratings and computes an implied point spread for their head-to-head matchup. The system will use the relative difference in ratings as the baseline spread, with adjustments for home field advantage and other configurable factors.

**Philosophy (Billy Walters’ System)**  
- Ratings are expressed in point-spread equivalent units.  
- Relative difference in power ratings = projected line on a neutral field.  
- Adjustments are then layered for game-specific contexts (home field, injuries, etc.).  
- Ratings can be negative (below-average teams), positive (above-average), or neutral (0 = league average).  

---

## 2. Scope & Goals

### 2.1 Features
- **Input**:  
  - CSV of power rankings (e.g., `power_rankings_week_initial_adjusted.csv`).  
  - Upcoming NFL schedule (pre-gathered as CSV or API feed).  

- **Computation**:  
  - Calculate projected spread as:  
    `Spread = (Team_A_Power – Team_B_Power) + Home_Field_Adj`  
  - Home Field Adjustment: Default +2.0 to home team (configurable).  
  - Ensure negative ratings are handled correctly (e.g., –4 vs –7 = +3 for the less-bad team).  

- **Output**:  
  - Generate CSV of projected spreads for each matchup, with columns:  
    `week, home_team, away_team, home_power, away_power, neutral_diff, home_field_adj, projected_spread`.  
  - Example: Bears (+5) vs Vikings (+2). Neutral spread = 3. Home field = +2 → Bears –5 favorite at home.

- **Configurable Parameters**:  
  - Home field value (default = 2.0 points).  
  - Option to adjust based on future modules (injuries, travel, weather).  

### 2.2 Non-Goals
- No machine learning or advanced regression—this is a simple arithmetic model.  
- Does not integrate real-time injury/weather adjustments in this first iteration.  
- Not intended to predict totals (O/U), only spreads.

---

## 3. Requirements

### 3.1 Functional Requirements
1. **Schedule Ingestion**  
   - Read NFL schedule (CSV or JSON).  
   - Map each team to its power score from `power_rankings_week_initial_adjusted.csv`.

2. **Spread Calculation**  
   - Neutral field spread = power rating difference.  
   - Apply home field adjustment.  
   - Allow negative ratings and compute spreads accordingly.  

3. **CSV Export**  
   - Save weekly matchup spreads to CSV:  
     - Filename: `projected_spreads_week_<N>.csv`.  

### 3.2 Non-Functional Requirements
- **Simplicity**: Lightweight, runs under 5 seconds per week.  
- **Accuracy**: Correct handling of negative values.  
- **Extensibility**: Easy to add injury/weather adjustments later.  

---

## 4. Architecture & Design

### 4.1 Components
- **`data_loader.py`**  
  - Loads power ranking CSV and schedule CSV.  

- **`spread_model.py`**  
  - `class SpreadCalculator`: Implements point spread formula.  

- **`exporter.py`**  
  - `class CSVExporter`: Writes computed spreads to CSV.  

- **`cli.py`**  
  - Entry-point CLI tool: `python cli.py --week 1`.  

### 4.2 Workflow
1. Load team power ratings from CSV.  
2. Load weekly schedule.  
3. For each matchup, compute:  
   - `neutral_diff = team_A_power – team_B_power`.  
   - `projected_spread = neutral_diff + home_field_adj`.  
4. Write results to CSV.  

---

## 5. Example Calculation

- **Input Power Ratings**:  
  - Chiefs = +8  
  - Raiders = –4  
- **Neutral Spread**: Chiefs – Raiders = +12  
- **At Kansas City (home field = +2)**: Chiefs –14 favorite.  
- **At Las Vegas (home field = +2)**: Chiefs –10 favorite.  

---

## 6. Testing Strategy

- **Unit Tests**:  
  - Verify neutral spread calculation with positive, negative, and mixed ratings.  
  - Validate home field adjustment.  

- **Integration Tests**:  
  - Run full schedule and confirm CSV outputs correctly formatted.  

---

## 7. Future Extensions

- Dynamic weekly updates (adjusted power ratings per performance).  
- Injury adjustments (subtract values for missing key players).  
- Travel, weather, and motivational factors (per Billy Walters’ methodology).  
- Integration with betting line data for edge comparison.  

---

**Conclusion**  
This PRD establishes a simple, arithmetic-driven NFL matchup spread model. It converts preseason power ratings into projected spreads, handling negative numbers and home field advantage. This is a foundational layer that can later expand into a more sophisticated handicapping system.
