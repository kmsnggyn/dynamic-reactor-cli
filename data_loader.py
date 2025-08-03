"""
Data Loader Module for Dynamic Reactor Analysis
==============================================

Handles loading and parsing of various CSV data formats into standardized data structures.
Supports multiple formats and provides extensible framework for new formats.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

class DataFormat(Enum):
    """Supported data formats"""
    ASPEN_PLUS_DYNAMICS = "aspen_dynamics"
    ASPEN_PLUS_STEADY = "aspen_steady"
    GENERIC_TIME_SERIES = "generic_timeseries"
    CUSTOM_FORMAT = "custom"

@dataclass
class DataMetadata:
    """Metadata about the loaded data"""
    format_type: DataFormat
    source_file: str
    dimensions: Dict[str, int]
    time_range: Tuple[float, float]
    spatial_range: Optional[Tuple[float, float]] = None
    variables: List[str] = None
    units: Dict[str, str] = None
    parsing_notes: List[str] = None

@dataclass
class StandardDataPackage:
    """Standardized data structure for all formats"""
    time_vector: np.ndarray
    length_vector: Optional[np.ndarray]  # None for non-spatial data
    variables: Dict[str, np.ndarray]  # Variable name -> data matrix
    metadata: DataMetadata
    
    @property
    def is_spatial(self) -> bool:
        """Check if data has spatial dimension"""
        return self.length_vector is not None
    
    @property
    def n_time(self) -> int:
        """Number of time points"""
        return len(self.time_vector)
    
    @property
    def n_spatial(self) -> int:
        """Number of spatial points (0 if non-spatial)"""
        return len(self.length_vector) if self.length_vector is not None else 0

class BaseDataParser(ABC):
    """Abstract base class for data parsers"""
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file"""
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> Optional[StandardDataPackage]:
        """Parse the file into standard data package"""
        pass
    
    @abstractmethod
    def get_format_description(self) -> str:
        """Get human-readable description of format"""
        pass

class AspenDynamicsParser(BaseDataParser):
    """Parser for Aspen Plus Dynamics CSV files"""
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file matches Aspen Dynamics format"""
        try:
            # Quick format check
            df = pd.read_csv(file_path, header=None, nrows=10)
            
            # Check structure indicators
            has_time_header = df.iloc[0, 0] and str(df.iloc[0, 0]).strip().lower() == "time"
            has_minutes_header = df.iloc[1, 0] and str(df.iloc[1, 0]).strip().lower() == "minutes"
            has_position_row = pd.to_numeric(df.iloc[2, 1:], errors='coerce').notna().any()
            
            return has_time_header and has_minutes_header and has_position_row
            
        except Exception:
            return False
    
    def parse(self, file_path: str) -> Optional[StandardDataPackage]:
        """Parse Aspen Dynamics CSV file"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        print(f"Parsing Aspen Dynamics format: {os.path.basename(file_path)}")
        
        try:
            # Load raw CSV
            df = pd.read_csv(file_path, header=None)
            print(f"  Raw data shape: {df.shape}")
            
            # Extract length vector from row 3 (index 2)
            position_row = df.iloc[2]
            length_vector = pd.to_numeric(position_row[1:], errors='coerce')
            length_vector = length_vector[~np.isnan(length_vector)]
            
            print(f"  Length vector: {len(length_vector)} positions from {length_vector.min():.3f} to {length_vector.max():.3f} m")
            
            # Find all time rows (every 6th row starting from index 2)
            time_values = []
            time_row_indices = []
            
            for i in range(2, len(df), 6):
                if i < len(df):
                    time_val = pd.to_numeric(df.iloc[i, 0], errors='coerce')
                    if not np.isnan(time_val):
                        time_values.append(time_val)
                        time_row_indices.append(i)
            
            time_vector = np.array(time_values)
            print(f"  Time vector: {len(time_vector)} points from {time_vector.min():.3f} to {time_vector.max():.3f} min")
            
            # Expected variables in order
            expected_variables = [
                "T_cat (°C)",
                "T (°C)", 
                "Reaction Rate (kmol/m3/hr)",
                "Heat Transfer to Catalyst (GJ/m3/hr)",
                "Heat Transfer with coolant (kW/m2)"
            ]
            
            # Initialize matrices
            n_time = len(time_vector)
            m_length = len(length_vector)
            variables = {}
            
            for var_name in expected_variables:
                variables[var_name] = np.full((n_time, m_length), np.nan)
            
            # Parse the data
            parsing_issues = []
            for t_idx, time_row_idx in enumerate(time_row_indices):
                if t_idx >= n_time:
                    break
                
                for var_idx in range(len(expected_variables)):
                    data_row_idx = time_row_idx + 1 + var_idx
                    
                    if data_row_idx < len(df):
                        row_data = df.iloc[data_row_idx, 1:1+m_length].values
                        row_data = pd.to_numeric(row_data, errors='coerce')
                        variables[expected_variables[var_idx]][t_idx, :] = row_data
                    else:
                        parsing_issues.append(f"Missing data row at time index {t_idx}, variable {expected_variables[var_idx]}")
            
            # Create metadata
            units = {
                "T_cat (°C)": "°C",
                "T (°C)": "°C",
                "Reaction Rate (kmol/m3/hr)": "kmol/m³/hr",
                "Heat Transfer to Catalyst (GJ/m3/hr)": "GJ/m³/hr",
                "Heat Transfer with coolant (kW/m2)": "kW/m²"
            }
            
            metadata = DataMetadata(
                format_type=DataFormat.ASPEN_PLUS_DYNAMICS,
                source_file=file_path,
                dimensions={'n_time': n_time, 'm_length': m_length},
                time_range=(time_vector.min(), time_vector.max()),
                spatial_range=(length_vector.min(), length_vector.max()),
                variables=list(expected_variables),
                units=units,
                parsing_notes=parsing_issues
            )
            
            # Create standard data package
            data_package = StandardDataPackage(
                time_vector=time_vector,
                length_vector=length_vector,
                variables=variables,
                metadata=metadata
            )
            
            print(f"  ✓ Successfully parsed {n_time} time points × {m_length} spatial points")
            if parsing_issues:
                print(f"  ⚠ {len(parsing_issues)} parsing issues noted")
            
            return data_package
            
        except Exception as e:
            print(f"  ✗ Error parsing Aspen Dynamics format: {e}")
            return None
    
    def get_format_description(self) -> str:
        return "Aspen Plus Dynamics CSV export with spatial temperature and reaction data"

class GenericTimeSeriesParser(BaseDataParser):
    """Parser for generic time-series CSV files"""
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file is a generic time-series CSV"""
        try:
            # Try to read as standard CSV with headers
            df = pd.read_csv(file_path, nrows=5)
            
            # Check if it has time-like first column and numeric data
            first_col = df.iloc[:, 0]
            has_time_col = (
                'time' in df.columns[0].lower() or
                pd.to_numeric(first_col, errors='coerce').notna().all()
            )
            
            # Check if other columns are mostly numeric
            numeric_cols = 0
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.8:
                    numeric_cols += 1
            
            return has_time_col and numeric_cols >= 1
            
        except Exception:
            return False
    
    def parse(self, file_path: str) -> Optional[StandardDataPackage]:
        """Parse generic time-series CSV"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        print(f"Parsing generic time-series format: {os.path.basename(file_path)}")
        
        try:
            # Load CSV with headers
            df = pd.read_csv(file_path)
            print(f"  Raw data shape: {df.shape}")
            
            # Extract time vector (first column)
            time_vector = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            time_vector = time_vector[~np.isnan(time_vector)]
            
            # Extract variables (remaining columns)
            variables = {}
            for col in df.columns[1:]:
                data = pd.to_numeric(df[col], errors='coerce').values[:len(time_vector)]
                if not np.isnan(data).all():  # Skip completely empty columns
                    variables[col] = data.reshape(-1, 1)  # Make it 2D for consistency
            
            print(f"  Time vector: {len(time_vector)} points from {time_vector.min():.3f} to {time_vector.max():.3f}")
            print(f"  Variables: {list(variables.keys())}")
            
            # Create metadata
            metadata = DataMetadata(
                format_type=DataFormat.GENERIC_TIME_SERIES,
                source_file=file_path,
                dimensions={'n_time': len(time_vector), 'm_length': 1},
                time_range=(time_vector.min(), time_vector.max()),
                spatial_range=None,
                variables=list(variables.keys()),
                units={var: "unknown" for var in variables.keys()}
            )
            
            # Create standard data package (no spatial dimension)
            data_package = StandardDataPackage(
                time_vector=time_vector,
                length_vector=None,  # No spatial data
                variables=variables,
                metadata=metadata
            )
            
            print(f"  ✓ Successfully parsed {len(time_vector)} time points, {len(variables)} variables")
            return data_package
            
        except Exception as e:
            print(f"  ✗ Error parsing time-series format: {e}")
            return None
    
    def get_format_description(self) -> str:
        return "Generic time-series CSV with time column and data columns"

class DataLoaderManager:
    """Main data loader that manages multiple parsers"""
    
    def __init__(self):
        self.parsers = [
            AspenDynamicsParser(),
            GenericTimeSeriesParser(),
        ]
        self._format_stats = {}
    
    def add_parser(self, parser: BaseDataParser):
        """Add a custom parser to the manager"""
        self.parsers.append(parser)
    
    def load_data(self, file_path: str, preferred_format: Optional[DataFormat] = None) -> Optional[StandardDataPackage]:
        """Load data using the best available parser"""
        if not file_path or not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
        
        print(f"\n=== Data Loading: {os.path.basename(file_path)} ===")
        
        # Try preferred format first
        if preferred_format:
            for parser in self.parsers:
                if (hasattr(parser, 'format_type') and 
                    parser.format_type == preferred_format and 
                    parser.can_parse(file_path)):
                    
                    result = parser.parse(file_path)
                    if result:
                        self._update_stats(preferred_format, True)
                        return result
        
        # Try all parsers in order
        for i, parser in enumerate(self.parsers):
            print(f"Trying parser {i+1}/{len(self.parsers)}: {parser.get_format_description()}")
            
            if parser.can_parse(file_path):
                print(f"  ✓ Format detected")
                result = parser.parse(file_path)
                if result:
                    self._update_stats(result.metadata.format_type, True)
                    return result
                else:
                    print(f"  ✗ Parsing failed")
            else:
                print(f"  - Format not matched")
        
        print(f"✗ No suitable parser found for file")
        return None
    
    def detect_format(self, file_path: str) -> Optional[DataFormat]:
        """Detect the format of a file without parsing it"""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                if hasattr(parser, 'format_type'):
                    return parser.format_type
                elif isinstance(parser, AspenDynamicsParser):
                    return DataFormat.ASPEN_PLUS_DYNAMICS
                elif isinstance(parser, GenericTimeSeriesParser):
                    return DataFormat.GENERIC_TIME_SERIES
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported format descriptions"""
        return [parser.get_format_description() for parser in self.parsers]
    
    def _update_stats(self, format_type: DataFormat, success: bool):
        """Update parsing statistics"""
        if format_type not in self._format_stats:
            self._format_stats[format_type] = {'success': 0, 'failure': 0}
        
        if success:
            self._format_stats[format_type]['success'] += 1
        else:
            self._format_stats[format_type]['failure'] += 1
    
    def print_stats(self):
        """Print parsing statistics"""
        if not self._format_stats:
            print("No parsing statistics available")
            return
        
        print("\n=== Data Loader Statistics ===")
        for format_type, stats in self._format_stats.items():
            total = stats['success'] + stats['failure']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            print(f"{format_type.value}: {stats['success']}/{total} successful ({success_rate:.1f}%)")

# Legacy compatibility functions
def load_and_parse_aspen_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Legacy compatibility function for existing code"""
    loader = DataLoaderManager()
    data_package = loader.load_data(file_path, DataFormat.ASPEN_PLUS_DYNAMICS)
    
    if data_package is None:
        return None
    
    # Convert to legacy format
    return {
        'time_vector': data_package.time_vector,
        'length_vector': data_package.length_vector,
        'variables': data_package.variables,
        'file_path': file_path,
        'dimensions': data_package.metadata.dimensions,
        'format_type': 'aspen_csv'  # Legacy format identifier
    }

def parse_ramp_parameters_from_filename(file_path: str):
    """Legacy compatibility function - imports from analysis_engine"""
    from analysis_engine import RampParameters
    
    filename = os.path.basename(file_path).lower()
    filename_parts = filename.replace('.csv', '').split('-')
    
    # Remove timestamp parts
    clean_parts = []
    for part in filename_parts:
        if not (part.isdigit() and len(part) >= 4):
            clean_parts.append(part)
    
    ramp_params = RampParameters()
    
    if len(clean_parts) >= 3:
        try:
            ramp_params.duration = int(clean_parts[0])
            ramp_params.direction = clean_parts[1]
            ramp_params.curve_shape = clean_parts[2]
        except (ValueError, IndexError):
            print(f"Warning: Could not parse filename format: {filename}")
    else:
        # Fallback detection
        if "up" in filename:
            ramp_params.direction = "up"
        elif "down" in filename:
            ramp_params.direction = "down"
    
    return ramp_params

# Create global instance for easy access
data_loader = DataLoaderManager()
