#!/usr/bin/env python3
"""Simple test of core NFL Projects functionality."""

from power_ranking.models.power_rankings import PowerRankModel
from nfl_model.spread_model import SpreadCalculator
import pandas as pd

def test_basic_functionality():
    """Test basic functionality without external APIs."""
    print("🏈 Simple NFL Projects Test")
    print("=" * 40)
    
    # Test 1: PowerRankModel initialization
    print("1️⃣ Testing PowerRankModel...")
    try:
        model = PowerRankModel()
        print("✅ PowerRankModel initialized successfully")
    except Exception as e:
        print(f"❌ PowerRankModel failed: {e}")
        return False
    
    # Test 2: SpreadCalculator
    print("2️⃣ Testing SpreadCalculator...")
    try:
        calculator = SpreadCalculator(home_field_advantage=2.5)
        
        # Test basic spread calculation
        kc_power = 85.0
        buf_power = 82.0
        spread = calculator.calculate_neutral_spread(kc_power, buf_power)
        print(f"✅ SpreadCalculator working: KC vs BUF spread = {spread:+.1f}")
        
    except Exception as e:
        print(f"❌ SpreadCalculator failed: {e}")
        return False
    
    # Test 3: Create sample data and full spread calculation
    print("3️⃣ Testing full spread workflow...")
    try:
        # Create sample power rankings
        power_data = pd.DataFrame({
            'team_name': ['KC', 'BUF', 'PHI', 'SF'],
            'power_score': [85.0, 82.0, 79.0, 78.0]
        })
        power_data.to_csv('test_power.csv', index=False)
        
        # Create sample schedule
        schedule_data = pd.DataFrame({
            'week': [1, 1],
            'home_team': ['KC', 'PHI'],
            'away_team': ['BUF', 'SF'],
            'game_date': ['2024-09-08', '2024-09-08']
        })
        schedule_data.to_csv('test_schedule.csv', index=False)
        
        # Load and calculate
        from nfl_model.data_loader import DataLoader
        loader = DataLoader('test_power.csv', 'test_schedule.csv')
        
        power_rankings = loader.load_power_rankings()
        schedule = loader.load_schedule(week=1)
        spreads = calculator.calculate_week_spreads(schedule, power_rankings, 1)
        
        print(f"✅ Generated {len(spreads)} spread predictions")
        for result in spreads:
            betting_line = calculator.format_spread_as_betting_line(
                result.spread, result.home_team, result.away_team
            )
            print(f"   {result.away_team} @ {result.home_team}: {betting_line}")
        
    except Exception as e:
        print(f"❌ Full workflow failed: {e}")
        return False
    
    # Test 4: Configuration system
    print("4️⃣ Testing configuration system...")
    try:
        from power_ranking.config_manager import get_config
        config = get_config()
        print(f"✅ Configuration loaded: {type(config).__name__}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    print("✅ NFL Projects Suite is ready for use!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)