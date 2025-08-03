#!/usr/bin/env python3
"""
Test script to verify the modular structure works correctly

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported without errors"""
    print("Testing modular structure imports...")
    
    try:
        from analysis_engine import AnalysisOptions, DataLoader, SteadyStateDetector, RampParameters
        print("PASS: analysis_engine imports successful")
    except ImportError as e:
        print(f"FAIL: analysis_engine import failed: {e}")
        return False
    
    try:
        from plot_generator import PlotGenerator
        print("PASS: plot_generator imports successful")
    except ImportError as e:
        print(f"FAIL: plot_generator import failed: {e}")
        return False
    
    try:
        from results_manager import ResultsComparison, AnalysisReporter, ConfigManager, DataExporter
        print("PASS: results_manager imports successful")
    except ImportError as e:
        print(f"FAIL: results_manager import failed: {e}")
        return False
    
    # Test the main GUI file can be imported
    try:
        import main_gui
        print("PASS: Main GUI file imports successful")
    except ImportError as e:
        print(f"FAIL: Main GUI import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key classes"""
    print("\nTesting basic functionality...")
    
    try:
        from analysis_engine import AnalysisOptions, RampParameters
        
        # Test AnalysisOptions
        options = AnalysisOptions()
        print(f"PASS: AnalysisOptions created: {options}")
        
        # Test RampParameters
        ramp = RampParameters(duration=60, direction="up", curve_shape="r")
        print(f"PASS: RampParameters created: {ramp}")
        
        return True
    except Exception as e:
        print(f"FAIL: Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MODULAR STRUCTURE TEST")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: ALL TESTS PASSED! The modular structure is working correctly.")
        print("You can now run the main GUI with: python main_gui.py")
    else:
        print("FAILED: Some tests failed. Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
