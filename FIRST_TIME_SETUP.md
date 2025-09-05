# First-Time Setup and Execution Guide

**Welcome to the NFL Projects Suite!** This guide will walk you through your first execution of both Power Rankings and NFL Spread predictions.

## 📋 Prerequisites

### System Requirements
- **Python 3.12+** (check with `python --version`)
- **pip** package manager
- **Internet connection** (for ESPN API data)

### Quick Installation
```bash
# 1. Clone the repository
git clone https://github.com/tyrelasaurus/nfl-projects.git
cd nfl-projects

# 2. Install dependencies
pip install -r power_ranking/requirements.txt
pip install -r requirements-test.txt

# 3. Verify installation
python -c "import power_ranking, nfl_model; print('✅ Installation successful')"
```

---

## 🏈 Part 1: Generate NFL Power Rankings

### Step 1: Basic Power Rankings Execution

Create a simple script to generate current week power rankings:

```python
# save as: run_power_rankings.py
from power_ranking.models.power_rankings import PowerRankModel
from power_ranking.api.espn_client import ESPNClient

def generate_power_rankings():
    """Generate current NFL power rankings."""
    print("🏈 NFL Power Rankings Generator")
    print("=" * 40)
    
    try:
        # Initialize components
        print("📡 Connecting to ESPN API...")
        client = ESPNClient()
        model = PowerRankModel()
        
        # Get current season data
        print("📊 Fetching current season data...")
        current_year = 2024  # Update as needed
        
        # Get teams data
        teams = client.get_teams_data()
        print(f"✅ Retrieved {len(teams)} teams")
        
        # Try to get current scoreboard data
        try:
            current_week = client.get_current_week()
            print(f"📅 Current week: {current_week}")
            scoreboard = client.get_scoreboard_data(current_year, current_week)
        except:
            # Fallback to a recent completed week
            print("⚠️  Using historical data for demonstration")
            scoreboard = client.get_scoreboard_data(current_year, 10)  # Week 10 as example
        
        print(f"🎯 Retrieved {len(scoreboard)} games")
        
        # Calculate power rankings
        print("🧮 Calculating power rankings...")
        rankings, processed_data = model.compute(scoreboard, teams)
        
        # Display results
        print("\n🏆 NFL POWER RANKINGS")
        print("=" * 50)
        print(f"{'Rank':<4} {'Team':<25} {'Score':<8} {'Record'}")
        print("-" * 50)
        
        for i, (team_id, team_name, power_score) in enumerate(rankings[:10], 1):
            # Get team record from processed data if available
            record = processed_data.get(team_id, {}).get('record', 'N/A')
            print(f"{i:<4} {team_name:<25} {power_score:<8.2f} {record}")
        
        print(f"\n✅ Generated rankings for {len(rankings)} teams")
        
        # Save results
        import json
        output_data = {
            'rankings': [(team_id, team_name, power_score) for team_id, team_name, power_score in rankings],
            'metadata': {
                'total_teams': len(rankings),
                'year': current_year,
                'games_processed': len(scoreboard)
            }
        }
        
        with open('power_rankings_output.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("💾 Results saved to: power_rankings_output.json")
        return rankings
        
    except Exception as e:
        print(f"❌ Error generating power rankings: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Check internet connection")
        print("- Verify ESPN API is accessible")
        print("- Try again in a few minutes")
        return None

if __name__ == "__main__":
    generate_power_rankings()
```

### Step 2: Run Power Rankings

```bash
python run_power_rankings.py
```

Expected output:
```
🏈 NFL Power Rankings Generator
========================================
📡 Connecting to ESPN API...
📊 Fetching current season data...
✅ Retrieved 32 teams
📅 Current week: 12
🎯 Retrieved 16 games
🧮 Calculating power rankings...

🏆 NFL POWER RANKINGS
==================================================
Rank Team                     Score    Record
--------------------------------------------------
1    Kansas City Chiefs       85.34    10-2
2    Buffalo Bills            82.67    9-3
3    Philadelphia Eagles      79.45    8-4
...
✅ Generated rankings for 32 teams
💾 Results saved to: power_rankings_output.json
```

---

## 📊 Part 2: Generate NFL Spread Predictions

### Step 1: Create Sample Power Rankings Data

First, let's create a CSV file with power rankings data:

```python
# save as: create_sample_data.py
import pandas as pd

def create_sample_power_rankings():
    """Create sample power rankings CSV for spread calculations."""
    # Sample data based on typical NFL power rankings
    power_rankings = {
        'team': ['KC', 'BUF', 'PHI', 'SF', 'DAL', 'BAL', 'MIA', 'CIN', 
                'JAX', 'NYJ', 'LV', 'DEN', 'LAC', 'IND', 'CLE', 'PIT',
                'HOU', 'TEN', 'ATL', 'TB', 'CAR', 'NO', 'MIN', 'GB',
                'DET', 'CHI', 'LAR', 'SEA', 'ARI', 'NYG', 'WAS', 'NE'],
        'power_score': [85.3, 82.7, 79.4, 78.9, 76.2, 75.8, 74.5, 73.2,
                       71.8, 70.5, 69.3, 68.1, 67.4, 66.8, 65.9, 65.2,
                       64.7, 63.5, 62.8, 62.1, 60.9, 60.3, 59.7, 58.9,
                       58.2, 57.4, 56.8, 56.1, 55.3, 54.6, 53.8, 52.1]
    }
    
    df = pd.DataFrame(power_rankings)
    df.to_csv('power_rankings.csv', index=False)
    print("✅ Created power_rankings.csv")
    return df

def create_sample_schedule():
    """Create sample schedule for Week 1 games."""
    # Sample Week 1 matchups
    schedule = {
        'week': [1, 1, 1, 1, 1, 1, 1, 1],
        'home_team': ['KC', 'BUF', 'PHI', 'SF', 'DAL', 'BAL', 'MIA', 'CIN'],
        'away_team': ['LV', 'NYJ', 'NE', 'ARI', 'NYG', 'PIT', 'LAC', 'CLE'],
        'game_date': ['2024-09-08'] * 8
    }
    
    df = pd.DataFrame(schedule)
    df.to_csv('schedule.csv', index=False)
    print("✅ Created schedule.csv")
    return df

if __name__ == "__main__":
    create_sample_power_rankings()
    create_sample_schedule()
```

### Step 2: Generate Spread Predictions

```python
# save as: run_spread_predictions.py
from nfl_model.spread_model import SpreadCalculator
from nfl_model.data_loader import DataLoader

def generate_spread_predictions():
    """Generate NFL spread predictions for upcoming games."""
    print("📊 NFL Spread Predictions Generator")
    print("=" * 40)
    
    try:
        # Initialize components
        print("🔧 Initializing spread calculator...")
        calculator = SpreadCalculator(home_field_advantage=2.5)
        loader = DataLoader()
        
        # Load power rankings
        print("📈 Loading power rankings...")
        power_rankings = loader.load_power_rankings("power_rankings.csv")
        print(f"✅ Loaded power rankings for {len(power_rankings)} teams")
        
        # Load schedule
        print("📅 Loading game schedule...")
        schedule = loader.load_schedule("schedule.csv", week=1)
        print(f"🎯 Found {len(schedule)} games for Week 1")
        
        # Calculate spreads for the week
        print("🧮 Calculating spread predictions...")
        week_spreads = calculator.calculate_week_spreads(schedule, power_rankings)
        
        # Display results
        print("\n🏈 WEEK 1 SPREAD PREDICTIONS")
        print("=" * 60)
        print(f"{'Matchup':<30} {'Spread':<15} {'Betting Line'}")
        print("-" * 60)
        
        for result in week_spreads:
            matchup = f"{result.away_team} @ {result.home_team}"
            spread_text = f"{result.spread:+.1f}"
            betting_line = calculator.format_spread_as_betting_line(
                result.spread, result.home_team, result.away_team
            )
            print(f"{matchup:<30} {spread_text:<15} {betting_line}")
        
        print(f"\n✅ Generated {len(week_spreads)} spread predictions")
        
        # Save results
        import json
        output_data = {
            'predictions': [
                {
                    'matchup': f"{r.away_team} @ {r.home_team}",
                    'home_team': r.home_team,
                    'away_team': r.away_team,
                    'spread': r.spread,
                    'betting_line': calculator.format_spread_as_betting_line(
                        r.spread, r.home_team, r.away_team
                    )
                }
                for r in week_spreads
            ],
            'metadata': {
                'week': 1,
                'home_field_advantage': 2.5,
                'total_games': len(week_spreads)
            }
        }
        
        with open('spread_predictions_output.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("💾 Results saved to: spread_predictions_output.json")
        return week_spreads
        
    except Exception as e:
        print(f"❌ Error generating spreads: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Ensure power_rankings.csv and schedule.csv exist")
        print("- Check file formats match expected structure")
        print("- Verify team abbreviations are consistent")
        return None

if __name__ == "__main__":
    generate_spread_predictions()
```

### Step 3: Run Complete Workflow

```bash
# 1. Create sample data
python create_sample_data.py

# 2. Generate spread predictions
python run_spread_predictions.py
```

Expected output:
```
📊 NFL Spread Predictions Generator
========================================
🔧 Initializing spread calculator...
📈 Loading power rankings...
✅ Loaded power rankings for 32 teams
📅 Loading game schedule...
🎯 Found 8 games for Week 1
🧮 Calculating spread predictions...

🏈 WEEK 1 SPREAD PREDICTIONS
============================================================
Matchup                        Spread          Betting Line
------------------------------------------------------------
LV @ KC                        -13.5           KC -13.5
NYJ @ BUF                      -9.7            BUF -9.5
NE @ PHI                       -23.9           PHI -24.0
ARI @ SF                       -21.1           SF -21.0
NYG @ DAL                      -19.0           DAL -19.0
PIT @ BAL                      -8.1            BAL -8.0
LAC @ MIA                      -4.5            MIA -4.5
CLE @ CIN                      -4.8            CIN -5.0

✅ Generated 8 spread predictions
💾 Results saved to: spread_predictions_output.json
```

---

## 🚀 Part 3: Advanced Execution

### Combining Power Rankings and Spreads

```python
# save as: full_nfl_analysis.py
from power_ranking.models.power_rankings import PowerRankModel
from power_ranking.api.espn_client import ESPNClient
from nfl_model.spread_model import SpreadCalculator
import pandas as pd

def run_complete_nfl_analysis():
    """Complete NFL analysis: Power Rankings → Spread Predictions."""
    print("🏈 Complete NFL Analysis Pipeline")
    print("=" * 50)
    
    # Step 1: Generate Power Rankings
    print("\n📈 STEP 1: Generating Power Rankings")
    print("-" * 30)
    
    client = ESPNClient()
    model = PowerRankModel()
    
    teams = client.get_teams_data()
    scoreboard = client.get_scoreboard_data(2024, 10)  # Use reliable week
    rankings, _ = model.compute(scoreboard, teams)
    
    print(f"✅ Generated power rankings for {len(rankings)} teams")
    
    # Step 2: Convert to DataFrame and save
    power_df = pd.DataFrame([
        {'team': team_abbr_from_name(team_name), 'power_score': score}
        for team_id, team_name, score in rankings
    ])
    power_df.to_csv('live_power_rankings.csv', index=False)
    
    # Step 3: Create sample upcoming games
    upcoming_games = [
        ('KC', 'BUF'), ('PHI', 'SF'), ('DAL', 'BAL'), 
        ('MIA', 'CIN'), ('LV', 'DEN')
    ]
    
    schedule_df = pd.DataFrame([
        {'week': 1, 'home_team': home, 'away_team': away, 'game_date': '2024-09-15'}
        for home, away in upcoming_games
    ])
    schedule_df.to_csv('upcoming_schedule.csv', index=False)
    
    # Step 4: Generate Spreads
    print("\n📊 STEP 2: Generating Spread Predictions")
    print("-" * 30)
    
    calculator = SpreadCalculator(home_field_advantage=2.5)
    loader = DataLoader()
    
    power_rankings = loader.load_power_rankings('live_power_rankings.csv')
    schedule = loader.load_schedule('upcoming_schedule.csv', week=1)
    spreads = calculator.calculate_week_spreads(schedule, power_rankings)
    
    print(f"✅ Generated {len(spreads)} spread predictions")
    
    # Step 5: Display Combined Results
    print("\n🏆 COMBINED ANALYSIS RESULTS")
    print("=" * 60)
    
    for result in spreads:
        home_rating = power_rankings.get(result.home_team, 0)
        away_rating = power_rankings.get(result.away_team, 0)
        
        print(f"\n{result.away_team} @ {result.home_team}")
        print(f"  Power Ratings: {result.away_team}({away_rating:.1f}) vs {result.home_team}({home_rating:.1f})")
        print(f"  Predicted Spread: {result.spread:+.1f}")
        print(f"  Betting Line: {calculator.format_spread_as_betting_line(result.spread, result.home_team, result.away_team)}")

def team_abbr_from_name(team_name):
    """Convert team names to abbreviations."""
    # Simple mapping - expand as needed
    mapping = {
        'Kansas City Chiefs': 'KC', 'Buffalo Bills': 'BUF',
        'Philadelphia Eagles': 'PHI', 'San Francisco 49ers': 'SF',
        'Dallas Cowboys': 'DAL', 'Baltimore Ravens': 'BAL',
        'Miami Dolphins': 'MIA', 'Cincinnati Bengals': 'CIN',
        # Add more as needed...
    }
    return mapping.get(team_name, team_name[:3].upper())

if __name__ == "__main__":
    run_complete_nfl_analysis()
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Ensure you're in the project directory
   cd nfl-projects
   
   # Install missing dependencies
   pip install pandas requests pyyaml numpy
   ```

2. **ESPN API Errors**
   - Check internet connection
   - ESPN API may have rate limits - wait a few minutes
   - Try different week numbers if current week fails

3. **Data Format Issues**
   - Ensure CSV files have correct column names
   - Check team abbreviations are consistent
   - Verify no missing data in power rankings

4. **Configuration Issues**
   ```bash
   # Test configuration loading
   python -c "from power_ranking.config_manager import get_config; print(get_config())"
   ```

### System Health Check

Run the monitoring system to verify everything is working:

```python
# Quick system check
from monitoring.health_checks import HealthChecker

checker = HealthChecker()
health_status = checker.check_system_health()

print(f"System Status: {health_status.overall_status}")
for component, status in health_status.components.items():
    print(f"  {component}: {status.status}")
```

---

## 📚 Next Steps

Once you've successfully run the basic examples:

1. **Explore Configuration**: Modify `power_ranking/config.yaml` to adjust model weights
2. **Historical Analysis**: Use different seasons and weeks for backtesting
3. **Custom Data**: Replace sample data with your own power rankings or schedules
4. **Advanced Features**: Explore the monitoring dashboard and validation systems

For detailed API documentation, see `docs/api_reference.md`.

---

## 🎯 Expected Results Summary

After running the complete workflow, you should have:

- ✅ **Power Rankings**: JSON file with team rankings and scores
- ✅ **Spread Predictions**: JSON file with game predictions and betting lines  
- ✅ **CSV Data Files**: Raw data for further analysis
- ✅ **System Verification**: Confirmed all components are working

**Congratulations!** You've successfully set up and executed the NFL Projects Suite. The system is now ready for advanced usage, customization, and production deployment.