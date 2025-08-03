"""
Example: Creating a Custom Data Parser for New CSV Formats
=========================================================

This script demonstrates how to extend the data loader with custom parsers
for different CSV formats.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

from data_loader import BaseDataParser, StandardDataPackage, DataMetadata, DataFormat, DataLoaderManager
import pandas as pd
import numpy as np
import os

class CustomExperimentParser(BaseDataParser):
    """Example parser for custom experimental data format"""
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file matches custom format"""
        try:
            # Check if file has specific header pattern
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Example: files starting with "# Experiment Data"
            return first_line.startswith("# Experiment Data")
            
        except Exception:
            return False
    
    def parse(self, file_path: str) -> StandardDataPackage:
        """Parse custom experimental data"""
        print(f"Parsing custom experiment format: {os.path.basename(file_path)}")
        
        try:
            # Skip comment lines and read data
            df = pd.read_csv(file_path, comment='#')
            
            # Assuming format: Time, Temp1, Temp2, Flow, Pressure
            time_vector = df['Time'].values
            
            # Create variables dictionary
            variables = {}
            for col in df.columns:
                if col != 'Time':
                    # Make it 2D for consistency (non-spatial data)
                    variables[col] = df[col].values.reshape(-1, 1)
            
            # Create metadata
            metadata = DataMetadata(
                format_type=DataFormat.CUSTOM_FORMAT,
                source_file=file_path,
                dimensions={'n_time': len(time_vector), 'm_length': 1},
                time_range=(time_vector.min(), time_vector.max()),
                variables=list(variables.keys()),
                units={var: "custom_unit" for var in variables.keys()}
            )
            
            return StandardDataPackage(
                time_vector=time_vector,
                length_vector=None,  # Non-spatial
                variables=variables,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error parsing custom format: {e}")
            return None
    
    def get_format_description(self) -> str:
        return "Custom experimental data with comment headers"

# Example usage
def demonstrate_extensibility():
    """Demonstrate how to add custom parsers"""
    
    print("=== Data Loader Extensibility Demo ===\n")
    
    # Create data loader manager
    loader = DataLoaderManager()
    
    print("Default supported formats:")
    for i, fmt in enumerate(loader.get_supported_formats(), 1):
        print(f"  {i}. {fmt}")
    
    # Add custom parser
    custom_parser = CustomExperimentParser()
    loader.add_parser(custom_parser)
    
    print(f"\nAfter adding custom parser:")
    for i, fmt in enumerate(loader.get_supported_formats(), 1):
        print(f"  {i}. {fmt}")
    
    # Create example custom file
    example_file = "example_custom_data.csv"
    with open(example_file, 'w') as f:
        f.write("# Experiment Data - Custom Format\n")
        f.write("# Date: 2025-08-03\n")
        f.write("Time,Temperature_A,Temperature_B,Flow_Rate,Pressure\n")
        f.write("0.0,25.0,23.5,100.0,1.01\n")
        f.write("1.0,26.2,24.1,102.0,1.02\n")
        f.write("2.0,27.5,25.0,105.0,1.03\n")
        f.write("3.0,28.1,25.8,108.0,1.04\n")
    
    print(f"\nTesting with custom file: {example_file}")
    
    # Try to load the custom file
    data = loader.load_data(example_file)
    
    if data:
        print(f"✓ Successfully loaded custom format!")
        print(f"  Format: {data.metadata.format_type}")
        print(f"  Time points: {data.n_time}")
        print(f"  Variables: {list(data.variables.keys())}")
        print(f"  Time range: {data.metadata.time_range}")
    else:
        print("✗ Failed to load custom format")
    
    # Clean up
    if os.path.exists(example_file):
        os.remove(example_file)
    
    print(f"\nThis demonstrates how you can easily add support for:")
    print(f"  • Different CSV header formats")
    print(f"  • Custom comment styles")
    print(f"  • Specialized data structures")
    print(f"  • Industry-specific formats")

if __name__ == "__main__":
    demonstrate_extensibility()
