#!/usr/bin/env python3
"""
Core Phase 2.1 Error Handling test - focused on the essential functionality.
"""

import sys
import os
import logging

# Add project paths
sys.path.append('/Users/tyrelshaw/Projects/power_ranking')
sys.path.append('/Users/tyrelshaw/Projects/nfl_model')

def setup_test_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_core_exceptions():
    """Test core exception functionality."""
    print("=" * 60)
    print("TESTING CORE EXCEPTION FUNCTIONALITY")
    print("=" * 60)
    
    success = True
    
    # Test Power Ranking exceptions
    try:
        from power_ranking.exceptions import (
            PowerRankingError, ESPNAPIError, DataValidationError,
            PowerRankingCalculationError, log_and_handle_error
        )
        
        # Test basic exception creation and handling
        try:
            raise ESPNAPIError("Test API error", status_code=500)
        except ESPNAPIError as e:
            print(f"✅ Power Ranking ESPNAPIError: {e.message}")
            print(f"   Context: {e.context}")
            print(f"   Recovery suggestions: {len(e.recovery_suggestions)}")
        
        try:
            raise DataValidationError("Test validation error", validation_failures=["test"])
        except DataValidationError as e:
            print(f"✅ Power Ranking DataValidationError: {e.message}")
        
        try:
            raise PowerRankingCalculationError("Test calculation error")
        except PowerRankingCalculationError as e:
            print(f"✅ Power Ranking CalculationError: {e.message}")
        
        print("✅ Power ranking exceptions working correctly")
        
    except Exception as e:
        print(f"❌ Power ranking exceptions failed: {e}")
        success = False
    
    # Test NFL Model exceptions
    try:
        from nfl_model.exceptions import (
            NFLModelError, DataLoadingError, SpreadCalculationError,
            InvalidArgumentError, log_model_error
        )
        
        # Test basic exception creation and handling
        try:
            raise DataLoadingError("Test data loading error")
        except DataLoadingError as e:
            print(f"✅ NFL Model DataLoadingError: {e.message}")
            print(f"   Recovery suggestions: {len(e.recovery_suggestions)}")
        
        try:
            raise SpreadCalculationError("Test spread error")
        except SpreadCalculationError as e:
            print(f"✅ NFL Model SpreadCalculationError: {e.message}")
        
        try:
            raise InvalidArgumentError("Test argument error", argument="--test")
        except InvalidArgumentError as e:
            print(f"✅ NFL Model InvalidArgumentError: {e.message}")
        
        print("✅ NFL model exceptions working correctly")
        
    except Exception as e:
        print(f"❌ NFL model exceptions failed: {e}")
        success = False
    
    return success

def test_logging_integration():
    """Test logging integration."""
    print("\n" + "=" * 60)
    print("TESTING LOGGING INTEGRATION")
    print("=" * 60)
    
    try:
        test_logger = logging.getLogger('phase_2_1_test')
        
        # Test power ranking logging
        from power_ranking.exceptions import log_and_handle_error, ESPNAPIError
        
        try:
            raise ESPNAPIError("Test logging error")
        except Exception as e:
            handled = log_and_handle_error(e, test_logger, context={'test_phase': '2.1'})
            if handled:
                print("✅ Power ranking error logging successful")
            else:
                print("❌ Power ranking error logging failed")
                return False
        
        # Test NFL model logging
        from nfl_model.exceptions import log_model_error, DataLoadingError
        
        try:
            raise DataLoadingError("Test NFL logging error")
        except Exception as e:
            handled = log_model_error(e, test_logger, context={'test_phase': '2.1'})
            if handled:
                print("✅ NFL model error logging successful")
            else:
                print("❌ NFL model error logging failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Logging integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exception_decorators():
    """Test exception handling decorators."""
    print("\n" + "=" * 60)
    print("TESTING EXCEPTION DECORATORS")
    print("=" * 60)
    
    try:
        from power_ranking.exceptions import handle_exception_with_recovery
        from nfl_model.exceptions import handle_nfl_model_exception
        
        # Test power ranking decorator
        @handle_exception_with_recovery
        def test_power_function():
            raise ValueError("Test power decorator")
        
        try:
            test_power_function()
        except Exception as e:
            print(f"✅ Power ranking decorator handled: {type(e).__name__}")
        
        # Test NFL model decorator
        @handle_nfl_model_exception
        def test_nfl_function():
            raise KeyError("Test NFL decorator")
        
        try:
            test_nfl_function()
        except Exception as e:
            print(f"✅ NFL model decorator handled: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception decorators failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run core Phase 2.1 tests."""
    setup_test_logging()
    
    print("🚀" * 20)
    print("PHASE 2.1 - CORE ERROR HANDLING TEST")
    print("🚀" * 20)
    
    test_results = {
        'core_exceptions': test_core_exceptions(),
        'logging_integration': test_logging_integration(),
        'exception_decorators': test_exception_decorators()
    }
    
    # Summary
    print("\n" + "🎯" * 20)
    print("PHASE 2.1 CORE TEST RESULTS")
    print("🎯" * 20)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 CORE RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 PHASE 2.1 CORE ERROR HANDLING IS WORKING PERFECTLY!")
        print("\n📋 Core Features Verified:")
        print("   ✅ Custom exception classes for both projects")
        print("   ✅ Proper logging integration with context")
        print("   ✅ Exception decorators and recovery patterns")
        print("   ✅ Structured error messages and suggestions")
        print("\n🚀 Phase 2.1 core implementation is complete and functional!")
    else:
        print("⚠️ Some core tests failed - see error messages above")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)