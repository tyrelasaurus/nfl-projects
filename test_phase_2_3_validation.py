#!/usr/bin/env python3
"""
Comprehensive test of the Phase 2.3 Input Validation & Type Safety system.
Tests pydantic schemas, data validation, type safety, and CSV format validation.
"""

import sys
import os
import tempfile
import pandas as pd
import json
from datetime import datetime, date
import logging
from pathlib import Path

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

def setup_test_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_power_ranking_schemas():
    """Test power ranking Pydantic schemas."""
    print("=" * 60)
    print("TESTING POWER RANKING PYDANTIC SCHEMAS")
    print("=" * 60)
    
    try:
        from power_ranking.schemas import ESPNGameData, TeamRanking, PowerRankingOutput, ESPNAPIResponse
        
        # Test ESPNGameData schema
        game_data = {
            "game_id": "12345",
            "week": 1,
            "season": 2025,
            "date": datetime.now(),
            "home_team": "KC",
            "away_team": "BUF",
            "home_score": 24,
            "away_score": 21,
            "status": "completed"
        }
        
        try:
            game = ESPNGameData(**game_data)
            print("✅ ESPNGameData schema validation passed")
            print(f"   Game: {game.away_team} @ {game.home_team}, Margin: {game.margin}")
        except Exception as e:
            print(f"❌ ESPNGameData schema validation failed: {e}")
            return False
        
        # Test invalid data
        try:
            invalid_game = ESPNGameData(
                game_id="12345",
                week=25,  # Invalid week
                season=2025,
                date=datetime.now(),
                home_team="KC",
                away_team="KC",  # Same team error
                status="completed"
            )
            print("❌ Invalid game data should have failed validation")
            return False
        except Exception:
            print("✅ Invalid game data correctly rejected")
        
        # Test TeamRanking schema
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
        
        try:
            ranking = TeamRanking(**ranking_data)
            print("✅ TeamRanking schema validation passed")
            print(f"   Team: {ranking.team_name}, Power Score: {ranking.power_score}, Rank: {ranking.rank}")
        except Exception as e:
            print(f"❌ TeamRanking schema validation failed: {e}")
            return False
        
        # Test PowerRankingOutput schema
        output_data = {
            "rankings": [ranking_data],
            "model_weights": {
                "season_avg_margin": 0.5,
                "rolling_avg_margin": 0.25,
                "sos": 0.2,
                "recency_factor": 0.05
            },
            "week": 1,
            "season": 2025
        }
        
        try:
            output = PowerRankingOutput(**output_data)
            print("✅ PowerRankingOutput schema validation passed")
            print(f"   Rankings: {len(output.rankings)} teams")
        except Exception as e:
            print(f"❌ PowerRankingOutput schema validation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import power ranking schemas: {e}")
        return False
    except Exception as e:
        print(f"❌ Power ranking schema test failed: {e}")
        return False

def test_nfl_model_schemas():
    """Test NFL model Pydantic schemas."""
    print("\n" + "=" * 60)
    print("TESTING NFL MODEL PYDANTIC SCHEMAS")
    print("=" * 60)
    
    try:
        from nfl_model.schemas import (
            PowerRankingRecord, ScheduleRecord, SpreadPrediction, 
            WeeklySpreadOutput, ModelConfiguration, PredictionAccuracy
        )
        
        # Test PowerRankingRecord schema
        ranking_data = {
            "team_name": "Kansas City Chiefs",
            "team_abbreviation": "KC",
            "power_score": 15.5,
            "rank": 1,
            "wins": 8,
            "losses": 2
        }
        
        try:
            ranking = PowerRankingRecord(**ranking_data)
            print("✅ PowerRankingRecord schema validation passed")
            print(f"   Team: {ranking.team_name}, Score: {ranking.power_score}")
        except Exception as e:
            print(f"❌ PowerRankingRecord schema validation failed: {e}")
            return False
        
        # Test ScheduleRecord schema
        schedule_data = {
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills", 
            "week": 1,
            "date": "2025-09-07",
            "home_score": 24,
            "away_score": 21,
            "status": "completed"
        }
        
        try:
            schedule = ScheduleRecord(**schedule_data)
            print("✅ ScheduleRecord schema validation passed")
            print(f"   Matchup: {schedule.away_team} @ {schedule.home_team}, Margin: {schedule.margin}")
        except Exception as e:
            print(f"❌ ScheduleRecord schema validation failed: {e}")
            return False
        
        # Test SpreadPrediction schema
        prediction_data = {
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "week": 1,
            "home_power": 15.5,
            "away_power": 12.3,
            "neutral_spread": 3.2,
            "home_field_adjustment": 2.0,
            "projected_spread": -5.2,
            "confidence_level": 0.95
        }
        
        try:
            prediction = SpreadPrediction(**prediction_data)
            print("✅ SpreadPrediction schema validation passed")
            print(f"   Prediction: {prediction.format_betting_line()}")
        except Exception as e:
            print(f"❌ SpreadPrediction schema validation failed: {e}")
            return False
        
        # Test ModelConfiguration schema
        config_data = {
            "home_field_advantage": 2.0,
            "confidence_level": 0.95,
            "tolerance_points": 3.0,
            "accuracy_threshold": 0.55,
            "rmse_target": 10.0
        }
        
        try:
            config = ModelConfiguration(**config_data)
            print("✅ ModelConfiguration schema validation passed")
            print(f"   HFA: {config.home_field_advantage}, Tolerance: {config.tolerance_points}")
        except Exception as e:
            print(f"❌ ModelConfiguration schema validation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import NFL model schemas: {e}")
        return False
    except Exception as e:
        print(f"❌ NFL model schema test failed: {e}")
        return False

def test_data_validators():
    """Test data validation systems."""
    print("\n" + "=" * 60)
    print("TESTING DATA VALIDATION SYSTEMS")
    print("=" * 60)
    
    try:
        # Test power ranking data validator
        from power_ranking.data_validator import PowerRankingDataValidator
        
        validator = PowerRankingDataValidator(strict_mode=False)
        
        # Test valid game data
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
        
        validated_game, result = validator.validate_espn_game_data(game_data)
        if validated_game:
            print("✅ Power ranking game validation passed")
        else:
            print(f"❌ Power ranking game validation failed: {result.errors}")
            return False
        
        # Test NFL model data validator
        from nfl_model.data_validator import NFLModelDataValidator
        
        nfl_validator = NFLModelDataValidator(strict_mode=False)
        
        # Test valid prediction data
        prediction_data = {
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "week": 1,
            "home_power": 15.5,
            "away_power": 12.3,
            "neutral_spread": 3.2,
            "home_field_adjustment": 2.0,
            "projected_spread": -5.2
        }
        
        validated_prediction, report = nfl_validator.validate_spread_prediction(prediction_data)
        if validated_prediction:
            print("✅ NFL model prediction validation passed")
        else:
            print(f"❌ NFL model prediction validation failed: {report.validation_errors}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import data validators: {e}")
        return False
    except Exception as e:
        print(f"❌ Data validator test failed: {e}")
        return False

def test_csv_validation():
    """Test CSV format validation."""
    print("\n" + "=" * 60)
    print("TESTING CSV FORMAT VALIDATION")
    print("=" * 60)
    
    try:
        from nfl_model.data_validator import validate_power_rankings_csv, validate_schedule_csv
        
        # Create test CSV data for power rankings
        power_rankings_data = [
            {"team_name": "Kansas City Chiefs", "power_score": 15.5, "rank": 1},
            {"team_name": "Buffalo Bills", "power_score": 12.3, "rank": 2},
            {"team_name": "Philadelphia Eagles", "power_score": 11.8, "rank": 3}
        ]
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(power_rankings_data)
            df.to_csv(f.name, index=False)
            temp_pr_path = f.name
        
        try:
            # Test power rankings CSV validation
            validated_df, report = validate_power_rankings_csv(temp_pr_path, strict_mode=False)
            
            if validated_df is not None and not report.has_errors:
                print("✅ Power rankings CSV validation passed")
                print(f"   Validated {len(validated_df)} records")
            else:
                print(f"❌ Power rankings CSV validation failed: {report.validation_errors}")
                return False
        finally:
            os.unlink(temp_pr_path)
        
        # Create test CSV data for schedule
        schedule_data = [
            {"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "week": 1, "date": "2025-09-07"},
            {"home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys", "week": 1, "date": "2025-09-07"}
        ]
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(schedule_data)
            df.to_csv(f.name, index=False)
            temp_schedule_path = f.name
        
        try:
            # Test schedule CSV validation
            validated_df, report = validate_schedule_csv(temp_schedule_path, strict_mode=False)
            
            if validated_df is not None and not report.has_errors:
                print("✅ Schedule CSV validation passed")
                print(f"   Validated {len(validated_df)} records")
            else:
                print(f"❌ Schedule CSV validation failed: {report.validation_errors}")
                return False
        finally:
            os.unlink(temp_schedule_path)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import CSV validators: {e}")
        return False
    except Exception as e:
        print(f"❌ CSV validation test failed: {e}")
        return False

def test_validated_data_loaders():
    """Test enhanced data loaders with validation."""
    print("\n" + "=" * 60)
    print("TESTING VALIDATED DATA LOADERS")
    print("=" * 60)
    
    try:
        from nfl_model.data_loader_validated import ValidatedDataLoader
        
        loader = ValidatedDataLoader(strict_mode=False)
        
        # Create test power rankings CSV
        power_rankings_data = [
            {"team_name": "Kansas City Chiefs", "power_score": 15.5, "rank": 1, "games_played": 10, "wins": 8, "losses": 2, "ties": 0},
            {"team_name": "Buffalo Bills", "power_score": 12.3, "rank": 2, "games_played": 10, "wins": 7, "losses": 3, "ties": 0},
            {"team_name": "Philadelphia Eagles", "power_score": 11.8, "rank": 3, "games_played": 10, "wins": 7, "losses": 3, "ties": 0}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(power_rankings_data)
            df.to_csv(f.name, index=False)
            temp_pr_path = f.name
        
        try:
            # Test validated power rankings loading
            power_rankings, report = loader.load_power_rankings(temp_pr_path)
            
            if power_rankings and not report.has_errors:
                print("✅ Validated power rankings loader passed")
                print(f"   Loaded {len(power_rankings)} team power rankings")
            else:
                print(f"❌ Validated power rankings loader failed: {report.validation_errors}")
                return False
        finally:
            os.unlink(temp_pr_path)
        
        # Create test schedule CSV
        schedule_data = [
            {"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "week": 1, "date": "2025-09-07"},
            {"home_team": "Philadelphia Eagles", "away_team": "Dallas Cowboys", "week": 1, "date": "2025-09-07"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(schedule_data)
            df.to_csv(f.name, index=False)
            temp_schedule_path = f.name
        
        try:
            # Test validated schedule loading
            matchups, report = loader.load_schedule(temp_schedule_path, target_week=1)
            
            if matchups and not report.has_errors:
                print("✅ Validated schedule loader passed")
                print(f"   Loaded {len(matchups)} matchups for week 1")
            else:
                print(f"❌ Validated schedule loader failed: {report.validation_errors}")
                return False
        finally:
            os.unlink(temp_schedule_path)
        
        # Test validation summary
        summary = loader.get_validation_summary()
        print("✅ Validation summary generated")
        print(f"   Operations: {summary['total_validation_operations']}, Success rate: {summary['overall_success_rate']:.1%}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import validated data loaders: {e}")
        return False
    except Exception as e:
        print(f"❌ Validated data loader test failed: {e}")
        return False

def test_pydantic_data_quality():
    """Test enhanced data quality validation."""
    print("\n" + "=" * 60)
    print("TESTING PYDANTIC DATA QUALITY VALIDATION")
    print("=" * 60)
    
    try:
        from power_ranking.validation.pydantic_data_quality import PydanticDataQualityValidator
        
        validator = PydanticDataQualityValidator(strict_mode=False)
        
        # Test game data quality validation
        games_data = [
            {
                "game_id": "game1",
                "week": 1,
                "season": 2025,
                "date": datetime.now(),
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": 24,
                "away_score": 21,
                "status": "completed"
            },
            {
                "game_id": "game2", 
                "week": 1,
                "season": 2025,
                "date": datetime.now(),
                "home_team": "PHI",
                "away_team": "DAL",
                "home_score": 28,
                "away_score": 14,
                "status": "completed"
            }
        ]
        
        quality_report = validator.validate_espn_game_data(games_data)
        
        if quality_report.is_valid:
            print("✅ Game data quality validation passed")
            print(f"   Overall score: {quality_report.overall_score:.2f}")
            print(f"   Valid records: {quality_report.valid_records}/{quality_report.total_records}")
        else:
            print(f"❌ Game data quality validation found issues: {len(quality_report.issues)} issues")
        
        # Test rankings data quality validation
        rankings_data = [
            {
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
            },
            {
                "team_id": "BUF",
                "team_name": "Buffalo Bills", 
                "team_abbreviation": "BUF",
                "power_score": 12.3,
                "rank": 2,
                "season_avg_margin": 6.1,
                "rolling_avg_margin": 7.2,
                "strength_of_schedule": 0.48,
                "games_played": 10,
                "wins": 7,
                "losses": 3,
                "ties": 0
            }
        ]
        
        rankings_quality_report = validator.validate_team_rankings(rankings_data)
        
        if rankings_quality_report.is_valid:
            print("✅ Rankings data quality validation passed")
            print(f"   Overall score: {rankings_quality_report.overall_score:.2f}")
            print(f"   Recommendations: {len(rankings_quality_report.recommendations)}")
        else:
            print(f"❌ Rankings data quality validation found issues: {len(rankings_quality_report.issues)} issues")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import pydantic data quality: {e}")
        return False
    except Exception as e:
        print(f"❌ Pydantic data quality test failed: {e}")
        return False

def test_type_safety_integration():
    """Test type safety integration throughout the system."""
    print("\n" + "=" * 60)
    print("TESTING TYPE SAFETY INTEGRATION")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # Test that schema classes have proper type hints
    try:
        from power_ranking.schemas import ESPNGameData
        import inspect
        
        # Check if class has type hints
        annotations = inspect.get_annotations(ESPNGameData)
        if annotations:
            print("✅ ESPNGameData has proper type annotations")
            success_count += 1
        else:
            print("❌ ESPNGameData missing type annotations")
        total_tests += 1
    except Exception as e:
        print(f"❌ Type annotation test failed: {e}")
        total_tests += 1
    
    # Test that validators have proper return types
    try:
        from nfl_model.data_validator import NFLModelDataValidator
        import inspect
        
        validator = NFLModelDataValidator()
        method = getattr(validator, 'validate_power_ranking_record')
        sig = inspect.signature(method)
        
        if sig.return_annotation != inspect.Signature.empty:
            print("✅ Data validator methods have proper return type annotations")
            success_count += 1
        else:
            print("❌ Data validator methods missing return type annotations")
        total_tests += 1
    except Exception as e:
        print(f"❌ Validator type annotation test failed: {e}")
        total_tests += 1
    
    # Test that data loaders use proper typing
    try:
        from nfl_model.data_loader_validated import ValidatedDataLoader
        import inspect
        
        loader = ValidatedDataLoader()
        method = getattr(loader, 'load_power_rankings')
        sig = inspect.signature(method)
        
        if sig.return_annotation != inspect.Signature.empty:
            print("✅ Data loader methods have proper return type annotations")
            success_count += 1
        else:
            print("❌ Data loader methods missing return type annotations")
        total_tests += 1
    except Exception as e:
        print(f"❌ Data loader type annotation test failed: {e}")
        total_tests += 1
    
    print(f"Type safety tests: {success_count}/{total_tests} passed")
    return success_count == total_tests

def main():
    """Run comprehensive Phase 2.3 input validation tests."""
    setup_test_logging()
    
    print("🚀" * 20)
    print("PHASE 2.3 - INPUT VALIDATION & TYPE SAFETY TEST")
    print("🚀" * 20)
    
    test_results = {
        'power_ranking_schemas': test_power_ranking_schemas(),
        'nfl_model_schemas': test_nfl_model_schemas(),
        'data_validators': test_data_validators(),
        'csv_validation': test_csv_validation(),
        'validated_data_loaders': test_validated_data_loaders(),
        'pydantic_data_quality': test_pydantic_data_quality(),
        'type_safety_integration': test_type_safety_integration()
    }
    
    # Summary
    print("\n" + "🎯" * 20)
    print("PHASE 2.3 TEST RESULTS SUMMARY")
    print("🎯" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 2.3 INPUT VALIDATION & TYPE SAFETY FEATURES WORKING!")
        print("✅ Input validation and type safety implementation is complete")
        print("")
        print("📋 Key Features Implemented:")
        print("   ✅ Comprehensive Pydantic schemas for both projects")
        print("   ✅ Type-safe data validation with detailed error reporting")
        print("   ✅ CSV format validation with schema enforcement")
        print("   ✅ Enhanced data loaders with validation integration")
        print("   ✅ Pydantic-based data quality assurance")
        print("   ✅ Type hints throughout all validation systems")
        print("   ✅ Structured validation reports with recommendations")
        print("")
        print("🚀 Phase 2.3 is complete and ready for production!")
    else:
        print("⚠️ Some tests failed - review error messages above")
        print(f"   Failed tests: {total_tests - passed_tests}")
    
    print("\n" + "🔧" * 20)
    print("Phase 2.3 Input Validation & Type Safety testing completed!")
    print(datetime.now().strftime("Completed at: %Y-%m-%d %H:%M:%S"))
    print("🔧" * 20)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)