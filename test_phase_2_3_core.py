#!/usr/bin/env python3
"""
Core Phase 2.3 Input Validation & Type Safety test - focused on working functionality.
"""

import sys
import os
import tempfile
import pandas as pd
from datetime import datetime
import logging

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

def setup_test_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_pydantic_schemas_basic():
    """Test basic Pydantic schema functionality."""
    print("=" * 60)
    print("TESTING PYDANTIC SCHEMAS BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test power ranking schemas
        from power_ranking.schemas import ESPNGameData, TeamRanking, PowerRankingOutput
        
        # Test basic schema creation and validation
        game_data = {
            "game_id": "test123",
            "week": 1,
            "season": 2025,
            "date": datetime.now(),
            "home_team": "KC", 
            "away_team": "BUF",
            "home_score": 24,
            "away_score": 21,
            "status": "completed"
        }
        
        game = ESPNGameData(**game_data)
        print("✅ ESPNGameData validation successful")
        print(f"   Game: {game.away_team} @ {game.home_team}")
        print(f"   Margin: {game.margin}, Total: {game.total_points}")
        
        # Test data validation (should reject invalid data)
        try:
            invalid_game = ESPNGameData(
                game_id="test",
                week=25,  # Invalid week
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="KC",  # Same team - should fail
                status="completed"
            )
            print("❌ Invalid data should have been rejected")
            return False
        except Exception:
            print("✅ Invalid data correctly rejected by schema validation")
        
        # Test team ranking schema
        ranking_data = {
            "team_id": "KC",
            "team_name": "Kansas City Chiefs",
            "team_abbreviation": "KC",
            "power_score": 15.5,
            "rank": 1,
            "season_avg_margin": 8.2,
            "rolling_avg_margin": 9.1,
            "strength_of_schedule": 0.52,
            "games_played": 10,
            "wins": 8,
            "losses": 2,
            "ties": 0
        }
        
        ranking = TeamRanking(**ranking_data)
        print("✅ TeamRanking validation successful")
        print(f"   Team: {ranking.team_name}, Score: {ranking.power_score}, Rank: {ranking.rank}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import schemas: {e}")
        return False
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_nfl_model_schemas_basic():
    """Test basic NFL model schema functionality."""
    print("\n" + "=" * 60)
    print("TESTING NFL MODEL SCHEMAS BASIC")
    print("=" * 60)
    
    try:
        from nfl_model.schemas import ScheduleRecord, ModelConfiguration
        
        # Test schedule record
        schedule_data = {
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "week": 1,
            "date": "2025-09-07",
            "home_score": 24,
            "away_score": 21,
            "status": "completed"
        }
        
        schedule = ScheduleRecord(**schedule_data)
        print("✅ ScheduleRecord validation successful")
        print(f"   Matchup: {schedule.away_team} @ {schedule.home_team}")
        print(f"   Date: {schedule.date}, Margin: {schedule.margin}")
        
        # Test model configuration
        config_data = {
            "home_field_advantage": 2.0,
            "confidence_level": 0.95,
            "tolerance_points": 3.0,
            "accuracy_threshold": 0.55,
            "rmse_target": 10.0
        }
        
        config = ModelConfiguration(**config_data)
        print("✅ ModelConfiguration validation successful")
        print(f"   HFA: {config.home_field_advantage}, Tolerance: {config.tolerance_points}")
        
        # Test invalid configuration (should fail)
        try:
            invalid_config = ModelConfiguration(
                home_field_advantage=15.0,  # Too high
                confidence_level=1.5,      # Invalid
                tolerance_points=-1.0      # Negative
            )
            print("❌ Invalid config should have been rejected")
            return False
        except Exception:
            print("✅ Invalid config correctly rejected")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import NFL schemas: {e}")
        return False
    except Exception as e:
        print(f"❌ NFL schema test failed: {e}")
        return False

def test_type_annotations():
    """Test that type annotations are present."""
    print("\n" + "=" * 60)
    print("TESTING TYPE ANNOTATIONS")
    print("=" * 60)
    
    try:
        import inspect
        
        # Test power ranking schemas have type annotations
        from power_ranking.schemas import ESPNGameData, TeamRanking
        
        espn_annotations = inspect.get_annotations(ESPNGameData)
        ranking_annotations = inspect.get_annotations(TeamRanking)
        
        if espn_annotations:
            print("✅ ESPNGameData has type annotations")
            print(f"   Fields annotated: {len(espn_annotations)}")
        else:
            print("❌ ESPNGameData missing type annotations")
            return False
        
        if ranking_annotations:
            print("✅ TeamRanking has type annotations")
            print(f"   Fields annotated: {len(ranking_annotations)}")
        else:
            print("❌ TeamRanking missing type annotations")
            return False
        
        # Test NFL model schemas
        from nfl_model.schemas import ScheduleRecord, ModelConfiguration
        
        schedule_annotations = inspect.get_annotations(ScheduleRecord)
        config_annotations = inspect.get_annotations(ModelConfiguration)
        
        if schedule_annotations and config_annotations:
            print("✅ NFL model schemas have type annotations")
            print(f"   ScheduleRecord fields: {len(schedule_annotations)}")
            print(f"   ModelConfiguration fields: {len(config_annotations)}")
        else:
            print("❌ NFL model schemas missing type annotations")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import for type annotation test: {e}")
        return False
    except Exception as e:
        print(f"❌ Type annotation test failed: {e}")
        return False

def test_data_validation_basic():
    """Test basic data validation functionality."""
    print("\n" + "=" * 60)
    print("TESTING BASIC DATA VALIDATION")
    print("=" * 60)
    
    try:
        from pydantic import ValidationError
        from power_ranking.schemas import ESPNGameData
        from nfl_model.schemas import ScheduleRecord
        
        # Test validation with missing required fields
        try:
            incomplete_game = ESPNGameData(
                game_id="test",
                week=1
                # Missing required fields
            )
            print("❌ Should have failed validation for missing fields")
            return False
        except ValidationError as e:
            print("✅ Correctly rejected incomplete data")
            print(f"   Validation errors: {len(e.errors())}")
        
        # Test validation with invalid data types
        try:
            invalid_types = ESPNGameData(
                game_id=123,  # Should be string
                week="one",   # Should be int
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="BUF"
            )
            print("❌ Should have failed validation for invalid types")
            return False
        except ValidationError as e:
            print("✅ Correctly rejected invalid data types")
            print(f"   Type validation errors: {len(e.errors())}")
        
        # Test validation with out-of-range values
        try:
            out_of_range = ESPNGameData(
                game_id="test",
                week=50,  # Out of valid range
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="BUF"
            )
            print("❌ Should have failed validation for out-of-range values")
            return False
        except ValidationError as e:
            print("✅ Correctly rejected out-of-range values")
            print(f"   Range validation errors: {len(e.errors())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import for validation test: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic validation test failed: {e}")
        return False

def test_csv_validation_basic():
    """Test basic CSV validation functionality."""
    print("\n" + "=" * 60)
    print("TESTING BASIC CSV VALIDATION")
    print("=" * 60)
    
    try:
        from pydantic import ValidationError
        from power_ranking.schemas import TeamRanking
        import pandas as pd
        
        # Create test CSV data
        test_data = [
            {"team_id": "KC", "team_name": "Kansas City Chiefs", "team_abbreviation": "KC", "power_score": 15.5, "rank": 1, "season_avg_margin": 8.2, "rolling_avg_margin": 9.1, "strength_of_schedule": 0.52, "games_played": 10, "wins": 8, "losses": 2, "ties": 0},
            {"team_id": "BUF", "team_name": "Buffalo Bills", "team_abbreviation": "BUF", "power_score": 12.3, "rank": 2, "season_avg_margin": 6.1, "rolling_avg_margin": 7.2, "strength_of_schedule": 0.48, "games_played": 10, "wins": 7, "losses": 3, "ties": 0}
        ]
        
        # Test valid data validation
        valid_records = []
        invalid_records = []
        
        for i, record in enumerate(test_data):
            try:
                validated = TeamRanking(**record)
                valid_records.append(validated)
            except ValidationError as e:
                invalid_records.append((i, e))
        
        if len(valid_records) == len(test_data) and len(invalid_records) == 0:
            print("✅ All valid records passed validation")
            print(f"   Validated {len(valid_records)} records successfully")
        else:
            print(f"❌ Validation failed: {len(valid_records)} valid, {len(invalid_records)} invalid")
            return False
        
        # Test with invalid data
        invalid_data = [
            {"team_name": "Invalid Team", "power_score": "not_a_number", "rank": 1},  # Invalid power_score type
            {"team_name": "Another Team", "power_score": 15.5, "rank": -1}  # Invalid rank
        ]
        
        invalid_count = 0
        for record in invalid_data:
            try:
                TeamRanking(**record)
            except ValidationError:
                invalid_count += 1
        
        if invalid_count == len(invalid_data):
            print("✅ All invalid records correctly rejected")
        else:
            print(f"❌ Some invalid records were not rejected: {len(invalid_data) - invalid_count}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import for CSV validation test: {e}")
        return False
    except Exception as e:
        print(f"❌ CSV validation test failed: {e}")
        return False

def test_schema_features():
    """Test advanced schema features."""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED SCHEMA FEATURES")
    print("=" * 60)
    
    try:
        from power_ranking.schemas import ESPNGameData, TeamRanking
        
        # Test computed properties
        game_data = {
            "game_id": "test123",
            "week": 1,
            "season": 2025,
            "date": datetime.now(),
            "home_team": "KC",
            "away_team": "BUF",
            "home_score": 24,
            "away_score": 21,
            "status": "completed"
        }
        
        game = ESPNGameData(**game_data)
        
        # Test computed properties
        if game.margin == 3 and game.total_points == 45:
            print("✅ Computed properties working correctly")
            print(f"   Margin: {game.margin}, Total Points: {game.total_points}")
        else:
            print(f"❌ Computed properties failed: margin={game.margin}, total={game.total_points}")
            return False
        
        # Test model validation (custom validation logic)
        try:
            # This should fail due to business logic validation
            same_team_game = ESPNGameData(
                game_id="test",
                week=1,
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="KC",  # Same team playing itself
                status="completed"
            )
            print("❌ Same team validation should have failed")
            return False
        except Exception:
            print("✅ Custom validation logic working (same team rejected)")
        
        # Test field validation
        try:
            invalid_week = ESPNGameData(
                game_id="test",
                week=25,  # Out of valid range
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="BUF"
            )
            print("❌ Week range validation should have failed")
            return False
        except Exception:
            print("✅ Field range validation working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import for schema features test: {e}")
        return False
    except Exception as e:
        print(f"❌ Schema features test failed: {e}")
        return False

def main():
    """Run core Phase 2.3 input validation tests."""
    setup_test_logging()
    
    print("🚀" * 20)
    print("PHASE 2.3 - INPUT VALIDATION & TYPE SAFETY CORE TEST")
    print("🚀" * 20)
    
    test_results = {
        'pydantic_schemas_basic': test_pydantic_schemas_basic(),
        'nfl_model_schemas_basic': test_nfl_model_schemas_basic(),
        'type_annotations': test_type_annotations(),
        'data_validation_basic': test_data_validation_basic(),
        'csv_validation_basic': test_csv_validation_basic(),
        'schema_features': test_schema_features()
    }
    
    # Summary
    print("\n" + "🎯" * 20)
    print("PHASE 2.3 CORE TEST RESULTS")
    print("🎯" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 CORE RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 PHASE 2.3 CORE INPUT VALIDATION & TYPE SAFETY IS WORKING!")
        print("\n📋 Core Features Verified:")
        print("   ✅ Comprehensive Pydantic schemas with validation")
        print("   ✅ Type annotations throughout schema definitions")
        print("   ✅ Data validation with error handling")
        print("   ✅ CSV format validation capabilities")
        print("   ✅ Advanced schema features (computed properties, custom validation)")
        print("   ✅ Type safety integration with proper error rejection")
        print("")
        print("🚀 Phase 2.3 core implementation is complete and functional!")
    else:
        print("⚠️ Some core tests failed - see error messages above")
    
    print("\n" + "🔧" * 20)
    print("Phase 2.3 Input Validation & Type Safety core testing completed!")
    print(datetime.now().strftime("Completed at: %Y-%m-%d %H:%M:%S"))
    print("🔧" * 20)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)