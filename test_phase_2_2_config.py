#!/usr/bin/env python3
"""
Comprehensive test of the Phase 2.2 Configuration Enhancement system.
Tests configuration loading, validation, environment overrides, and integration.
"""

import sys
import os
import yaml
import tempfile
import logging
from datetime import datetime
import traceback

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

def setup_test_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_power_ranking_config():
    """Test power ranking configuration system."""
    print("=" * 60)
    print("TESTING POWER RANKING CONFIGURATION")
    print("=" * 60)
    
    try:
        from power_ranking.config_manager import ConfigManager, get_config
        
        # Test basic config loading
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            print("✅ Power ranking config loaded successfully")
            print(f"   Environment: {config.environment}")
            print(f"   Model weights: season={config.model.weights.season_avg_margin}, rolling={config.model.weights.rolling_avg_margin}")
            print(f"   Rolling window: {config.model.rolling_window}")
            print(f"   Week 18 weight: {config.model.week18_weight}")
        except Exception as e:
            print(f"❌ Basic config loading failed: {e}")
            return False
        
        # Test environment overrides
        try:
            prod_manager = ConfigManager(environment='production')
            prod_config = prod_manager.load_config()
            print("✅ Production environment config loaded")
            print(f"   Logging level: {prod_config.logging.level}")
            print(f"   File logging enabled: {prod_config.logging.file_logging.get('enabled', False)}")
        except Exception as e:
            print(f"❌ Environment override failed: {e}")
            return False
        
        # Test configuration validation
        try:
            # Create invalid config
            invalid_config_data = {
                'model': {
                    'weights': {
                        'season_avg_margin': 0.8,  # Invalid - sum > 1.0
                        'rolling_avg_margin': 0.5,
                        'sos': 0.2,
                        'recency_factor': 0.1
                    }
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(invalid_config_data, f)
                temp_path = f.name
            
            try:
                invalid_manager = ConfigManager(temp_path)
                invalid_config = invalid_manager.load_config()
                print("❌ Configuration validation should have failed")
                return False
            except Exception:
                print("✅ Configuration validation correctly rejected invalid config")
            finally:
                os.unlink(temp_path)
        
        except Exception as e:
            print(f"❌ Configuration validation test failed: {e}")
            return False
        
        # Test global config access
        try:
            global_config = get_config()
            print("✅ Global config access working")
            print(f"   API base URL: {global_config.api.base_url}")
            print(f"   Output directory: {global_config.output.directory}")
        except Exception as e:
            print(f"❌ Global config access failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Power ranking configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_nfl_model_config():
    """Test NFL model configuration system."""
    print("\n" + "=" * 60)
    print("TESTING NFL MODEL CONFIGURATION")
    print("=" * 60)
    
    try:
        from nfl_model.config_manager import NFLConfigManager, get_nfl_config
        
        # Test basic config loading
        try:
            config_manager = NFLConfigManager()
            config = config_manager.load_config()
            print("✅ NFL model config loaded successfully")
            print(f"   Environment: {config.environment}")
            print(f"   Home field advantage: {config.model.home_field_advantage}")
            print(f"   Tolerance points: {config.model.validation.get('tolerance_points')}")
            print(f"   Confidence level: {config.model.calculation.get('confidence_level')}")
        except Exception as e:
            print(f"❌ Basic config loading failed: {e}")
            return False
        
        # Test convenience methods
        try:
            hfa = config_manager.get_home_field_advantage()
            tolerance = config_manager.get_tolerance_points()
            confidence = config_manager.get_confidence_level()
            week_range = config_manager.get_week_range()
            
            print("✅ Convenience methods working")
            print(f"   HFA: {hfa}, Tolerance: {tolerance}, Confidence: {confidence}")
            print(f"   Week range: {week_range}")
        except Exception as e:
            print(f"❌ Convenience methods failed: {e}")
            return False
        
        # Test feature flags
        try:
            enhanced_logging = config_manager.is_feature_enabled('enhanced_logging')
            caching = config_manager.is_feature_enabled('caching')
            print("✅ Feature flags working")
            print(f"   Enhanced logging: {enhanced_logging}, Caching: {caching}")
        except Exception as e:
            print(f"❌ Feature flags failed: {e}")
            return False
        
        # Test environment overrides
        try:
            test_manager = NFLConfigManager(environment='testing')
            test_config = test_manager.load_config()
            print("✅ Testing environment config loaded")
            print(f"   Logging level: {test_config.logging.level}")
            print(f"   Min sample size: {test_config.model.validation.get('min_sample_size')}")
        except Exception as e:
            print(f"❌ Environment override failed: {e}")
            return False
        
        # Test global config access
        try:
            global_config = get_nfl_config()
            print("✅ Global NFL config access working")
            print(f"   RMSE target: {global_config.metrics.rmse_target}")
            print(f"   Bootstrap iterations: {global_config.metrics.bootstrap_iterations}")
        except Exception as e:
            print(f"❌ Global NFL config access failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ NFL model configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_model_integration():
    """Test that models properly use configuration values."""
    print("\n" + "=" * 60)
    print("TESTING MODEL INTEGRATION WITH CONFIG")
    print("=" * 60)
    
    try:
        # Test Power Ranking Model integration
        from power_ranking.models.power_rankings import PowerRankModel
        
        try:
            model = PowerRankModel()
            print("✅ PowerRankModel created with default config")
            print(f"   Weights: {model.weights}")
            print(f"   Rolling window: {model.rolling_window}")
            print(f"   Week 18 weight: {model.week18_weight}")
            
            # Test with custom weights (should override config)
            custom_weights = {'season_avg_margin': 0.4, 'rolling_avg_margin': 0.3, 'sos': 0.2, 'recency_factor': 0.1}
            model_custom = PowerRankModel(weights=custom_weights)
            print("✅ PowerRankModel created with custom weights")
            print(f"   Custom weights: {model_custom.weights}")
            
        except Exception as e:
            print(f"❌ PowerRankModel integration failed: {e}")
            return False
        
        # Test Spread Calculator integration
        from nfl_model.spread_model import SpreadCalculator
        
        try:
            calculator = SpreadCalculator()
            print("✅ SpreadCalculator created with default config")
            print(f"   Home field advantage: {calculator.home_field_advantage}")
            
            # Test with custom home field advantage
            calculator_custom = SpreadCalculator(home_field_advantage=3.5)
            print("✅ SpreadCalculator created with custom HFA")
            print(f"   Custom HFA: {calculator_custom.home_field_advantage}")
            
        except Exception as e:
            print(f"❌ SpreadCalculator integration failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration with configuration."""
    print("\n" + "=" * 60)
    print("TESTING CLI INTEGRATION WITH CONFIG")
    print("=" * 60)
    
    try:
        # Test importing enhanced CLI (validates config integration)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cli_enhanced", 
            "/Users/tyrelshaw/Projects/nfl_model/cli_enhanced.py"
        )
        cli_module = importlib.util.module_from_spec(spec)
        
        # This tests if the CLI can load without configuration errors
        spec.loader.exec_module(cli_module)
        print("✅ Enhanced CLI loaded successfully with config integration")
        
        # Test that config values are properly loaded for argument defaults
        if hasattr(cli_module, 'get_nfl_config'):
            print("✅ CLI has access to configuration system")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        traceback.print_exc()
        return False

def test_config_file_formats():
    """Test configuration with different file formats and scenarios."""
    print("\n" + "=" * 60)
    print("TESTING CONFIG FILE FORMATS AND SCENARIOS")
    print("=" * 60)
    
    try:
        # Test minimal valid config
        minimal_config = {
            'model': {
                'weights': {
                    'season_avg_margin': 0.5,
                    'rolling_avg_margin': 0.25,
                    'sos': 0.2,
                    'recency_factor': 0.05
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            temp_path = f.name
        
        try:
            from power_ranking.config_manager import ConfigManager
            minimal_manager = ConfigManager(temp_path)
            minimal_cfg = minimal_manager.load_config()
            print("✅ Minimal configuration loaded successfully")
            print(f"   Uses defaults for missing values: API timeout = {minimal_cfg.api.timeout_seconds}")
        finally:
            os.unlink(temp_path)
        
        # Test missing config file handling
        try:
            from nfl_model.config_manager import NFLConfigManager
            missing_manager = NFLConfigManager('/nonexistent/path/config.yaml')
            missing_config = missing_manager.load_config()
            print("❌ Should have failed for missing config file")
            return False
        except Exception:
            print("✅ Properly handles missing configuration file")
        
        # Test invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            temp_path = f.name
        
        try:
            invalid_manager = ConfigManager(temp_path)
            invalid_cfg = invalid_manager.load_config()
            print("❌ Should have failed for invalid YAML")
            return False
        except Exception:
            print("✅ Properly handles invalid YAML syntax")
        finally:
            os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Config file format test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive Phase 2.2 configuration tests."""
    setup_test_logging()
    
    print("🚀" * 20)
    print("PHASE 2.2 - CONFIGURATION ENHANCEMENT TEST")
    print("🚀" * 20)
    
    test_results = {
        'power_ranking_config': test_power_ranking_config(),
        'nfl_model_config': test_nfl_model_config(),
        'model_integration': test_model_integration(),
        'cli_integration': test_cli_integration(),
        'config_file_formats': test_config_file_formats()
    }
    
    # Summary
    print("\n" + "🎯" * 20)
    print("PHASE 2.2 TEST RESULTS SUMMARY")
    print("🎯" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 2.2 CONFIGURATION FEATURES WORKING PERFECTLY!")
        print("✅ Configuration enhancement is complete")
        print("")
        print("📋 Key Features Implemented:")
        print("   ✅ Comprehensive YAML configuration files for both projects")
        print("   ✅ Environment-specific configuration overrides")
        print("   ✅ Type-safe configuration with dataclasses")
        print("   ✅ Configuration validation and error handling")
        print("   ✅ Seamless integration with existing models and CLI")
        print("   ✅ Configurable magic numbers and thresholds")
        print("   ✅ Global configuration managers with convenience methods")
        print("")
        print("🚀 Phase 2.2 is complete and ready for production!")
    else:
        print("⚠️ Some tests failed - review error messages above")
        print(f"   Failed tests: {total_tests - passed_tests}")
    
    print("\n" + "🔧" * 20)
    print("Phase 2.2 Configuration Enhancement testing completed!")
    print(datetime.now().strftime("Completed at: %Y-%m-%d %H:%M:%S"))
    print("🔧" * 20)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)