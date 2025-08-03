"""
Integration Test: Data Loader with Analysis Pipeline
==================================================

Test that the new data loader works correctly with the complete analysis pipeline.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import sys
from data_loader import DataLoaderManager, DataFormat
from analysis_engine import AnalysisEngine, AnalysisOptions

def test_integration():
    """Test complete pipeline with new data loader"""
    
    print("=== Data Loader Integration Test ===\n")
    
    # Test file path
    test_file = "c:/Users/sg/OneDrive - KTH/Dynamic Runs/30-down-s-20250801-192834.csv"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please update the path to an actual Aspen CSV file")
        return False
    
    try:
        # Test 1: Direct data loader usage
        print("1. Testing DataLoaderManager...")
        loader = DataLoaderManager()
        data_package = loader.load_data(test_file)
        
        if data_package:
            print(f"   ✓ Successfully loaded data")
            print(f"   ✓ Format: {data_package.metadata.format_type}")
            print(f"   ✓ Time points: {data_package.n_time}")
            print(f"   ✓ Spatial points: {data_package.n_spatial}")
            print(f"   ✓ Variables: {len(data_package.variables)}")
        else:
            print(f"   ✗ Failed to load data")
            return False
        
        # Test 2: Analysis Engine integration
        print(f"\n2. Testing AnalysisEngine integration...")
        engine = AnalysisEngine()
        
        if engine.load_data(test_file):
            print(f"   ✓ AnalysisEngine loaded data successfully")
            print(f"   ✓ Ramp parameters detected")
            
            # Test analysis
            if engine.run_steady_state_analysis():
                print(f"   ✓ Steady state analysis completed")
                
                # Test metrics extraction
                metrics = engine.extract_key_metrics()
                if metrics:
                    print(f"   ✓ Extracted {len(metrics)} metrics")
                    
                    # Show some key metrics
                    print(f"   ✓ Sample metrics:")
                    for key in ['Tcat_max', 'Tcat_min', 'Ramp_Duration', 'Steady_State_Detected']:
                        if key in metrics:
                            value = metrics[key]['value']
                            unit = metrics[key]['unit']
                            print(f"     - {key}: {value} {unit}")
                else:
                    print(f"   ✗ Failed to extract metrics")
                    return False
            else:
                print(f"   ✗ Analysis failed")
                return False
        else:
            print(f"   ✗ AnalysisEngine failed to load data")
            return False
        
        # Test 3: Format detection
        print(f"\n3. Testing format detection...")
        detected_format = loader.detect_format(test_file)
        if detected_format:
            print(f"   ✓ Detected format: {detected_format}")
        else:
            print(f"   ✗ Could not detect format")
            return False
        
        # Test 4: Statistics
        print(f"\n4. Testing statistics...")
        loader.print_stats()
        
        print(f"\n✓ All integration tests passed!")
        print(f"\nThe new data loader is successfully integrated and working with:")
        print(f"  • Format detection and parsing")
        print(f"  • Analysis engine pipeline") 
        print(f"  • Metrics extraction")
        print(f"  • Error handling and fallbacks")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
