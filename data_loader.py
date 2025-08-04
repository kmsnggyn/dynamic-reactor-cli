"""
Data Loader Module for Dynamic Reactor Analysis
==============================================

Handles loading and parsing of various CSV data formats into standardized data structures.
Supports multiple formats and provides extensible framework for new formats.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Data Loading Configuration Constants
MIN_NUMERIC_COLUMN_RATIO = 0.8  # Minimum ratio of numeric values to consider a column valid
ASPEN_ROW_BLOCK_SIZE = 6  # Number of rows per time block in Aspen Dynamics format
DEFAULT_RAMP_START_TIME = 10.0  # Default ramp start time in minutes
MIN_DIGIT_COUNT_TIMESTAMP = 8  # Minimum digits for timestamp detection
TIMESTAMP_PATTERN = r'(\d{8}-\d{6})'  # Pattern for YYYYMMDD-HHMMSS format

def extract_timestamp_from_filename(file_path: str) -> Optional[str]:
    """
    Extract timestamp from filename in format YYYYMMDD-HHMMSS.
    
    Args:
        file_path: Path to the file with timestamp in filename
        
    Returns:
        Extracted timestamp string or None if not found
        
    Example:
        >>> extract_timestamp_from_filename('data_20250804-143022.csv')
        '20250804-143022'
    """
    filename = os.path.basename(file_path)
    # Look for pattern: 8 digits, hyphen, 6 digits (YYYYMMDD-HHMMSS)
    match = re.search(TIMESTAMP_PATTERN, filename)
    return match.group(1) if match else None

def extract_ramp_parameters_from_filename(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract ramp parameters from filename in format: duration-direction-curve_shape.
    
    Args:
        file_path: Path to the file with ramp parameters in filename
        
    Returns:
        Dictionary with extracted parameters or None if parsing fails
        
    Example:
        >>> extract_ramp_parameters_from_filename('36-down-s-20250804-143022.csv')
        {'duration': 36, 'direction': 'down', 'curve_shape': 's', 'start_time': 10.0, 'end_time': 46.0}
    """
    filename = os.path.basename(file_path).lower()
    filename_parts = filename.replace('.csv', '').split('-')
    
    # Remove timestamp parts (8+ digits)
    clean_parts = []
    for part in filename_parts:
        if not (part.isdigit() and len(part) >= MIN_DIGIT_COUNT_TIMESTAMP):
            clean_parts.append(part)
    
    if len(clean_parts) >= 3:
        try:
            duration = int(clean_parts[0])
            direction = clean_parts[1]
            curve_shape = clean_parts[2]
            
            return {
                'duration': duration,
                'direction': direction,
                'curve_shape': curve_shape,
                'start_time': DEFAULT_RAMP_START_TIME,  # Default start time
                'end_time': DEFAULT_RAMP_START_TIME + duration
            }
        except (ValueError, IndexError):
            pass
    
    # Fallback detection
    ramp_info = {}
    if "up" in filename:
        ramp_info['direction'] = "up"
    elif "down" in filename:
        ramp_info['direction'] = "down"
    
    return ramp_info if ramp_info else None

class DataFormat(Enum):
    """
    Enumeration of supported data formats.
    
    Defines the different CSV formats that can be parsed by the data loader.
    Each format has its own specialized parser that can handle the specific
    structure and conventions of that format.
    """
    ASPEN_PLUS_DYNAMICS = "aspen_dynamics"
    GENERIC_TIME_SERIES = "generic_timeseries"

@dataclass
class DataMetadata:
    """
    Metadata container for loaded data with comprehensive information.
    
    Stores all relevant metadata about the loaded dataset including format details,
    dimensions, ranges, and extracted parameters. This metadata is essential for
    proper analysis and visualization of the reactor data.
    
    Attributes:
        format_type: The detected/used data format
        source_file: Full path to the original data file
        dimensions: Dictionary containing data dimensions (n_time, m_length, etc.)
        time_range: Tuple of (min_time, max_time) in minutes
        spatial_range: Tuple of (min_position, max_position) in meters, or None
        variables: List of variable names found in the data
        units: Dictionary mapping variable names to their units
        parsing_notes: List of warnings or issues encountered during parsing
        file_timestamp: Timestamp extracted from filename (YYYYMMDD-HHMMSS format)
        ramp_parameters: Dictionary with extracted ramp experiment parameters
        
    Example:
        >>> metadata = DataMetadata(
        ...     format_type=DataFormat.ASPEN_PLUS_DYNAMICS,
        ...     source_file="reactor_data.csv",
        ...     dimensions={'n_time': 1000, 'm_length': 50},
        ...     time_range=(0.0, 100.0)
        ... )
    """
    format_type: DataFormat
    source_file: str
    dimensions: Dict[str, int]
    time_range: Tuple[float, float]
    spatial_range: Optional[Tuple[float, float]] = None
    variables: Optional[List[str]] = None
    units: Optional[Dict[str, str]] = None
    parsing_notes: Optional[List[str]] = None
    file_timestamp: Optional[str] = None  # Extracted from filename (YYYYMMDD-HHMMSS)
    ramp_parameters: Optional[Dict[str, Any]] = None  # Extracted ramp experiment parameters

@dataclass
class StandardDataPackage:
    """
    Standardized data structure for all supported formats.
    
    This class provides a unified interface for reactor data regardless of the
    original file format. All parsers convert their specific formats into this
    standard structure, enabling consistent analysis across different data sources.
    
    The structure supports both spatial (1D reactor) and non-spatial (lumped) data.
    For spatial data, variables are 2D matrices [time, position]. For non-spatial
    data, variables are 2D matrices [time, 1] for consistency.
    
    Attributes:
        time_vector: 1D array of time points in minutes
        length_vector: 1D array of spatial positions in meters (None for non-spatial)
        variables: Dictionary mapping variable names to their data matrices
        metadata: Complete metadata about the data source and parsing
        
    Properties:
        is_spatial: True if data has spatial dimension
        n_time: Number of time points
        n_spatial: Number of spatial points (0 for non-spatial)
        
    Example:
        >>> package = StandardDataPackage(
        ...     time_vector=np.linspace(0, 100, 1000),
        ...     length_vector=np.linspace(0, 1, 50),
        ...     variables={'T_cat (°C)': np.random.normal(500, 10, (1000, 50))},
        ...     metadata=metadata
        ... )
        >>> print(f"Data: {package.n_time} times × {package.n_spatial} positions")
    """
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

class AspenDynamicsParser:
    """
    Parser for Aspen Plus Dynamics CSV export files.
    
    This parser handles the specific format generated by Aspen Plus Dynamics
    when exporting spatial reactor data. The format consists of:
    - Header row with "Time" in first column
    - Units row (may be "Minutes" or empty)
    - Position values in third row
    - Data blocks of 6 rows: time + 5 variables at each time point
    
    Supported variables in order:
    1. Catalyst Temperature (°C)
    2. Gas Temperature (°C)
    3. Reaction Rate (kmol/m³/hr)
    4. Heat Transfer to Catalyst (GJ/m³/hr)
    5. Heat Transfer with Coolant (kW/m²)
    
    The parser automatically detects the format and extracts both temporal
    and spatial data into standardized matrices.
    """
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file matches Aspen Dynamics format"""
        try:
            # Quick format check
            df = pd.read_csv(file_path, header=None, nrows=10)
            
            # Check structure indicators
            has_time_header = bool(df.iloc[0, 0] and str(df.iloc[0, 0]).strip().lower() == "time")
            
            # Row 2 can either be "Minutes" or empty/nan (newer Aspen exports)
            if pd.isna(df.iloc[1, 0]):
                row2_val = ""  # Treat NaN as empty
            else:
                row2_val = str(df.iloc[1, 0]).strip().lower()
            has_minutes_header = bool(row2_val in ["minutes", ""])
            
            # Check if row 3 has spatial positions starting from column 2
            has_position_row = bool(pd.to_numeric(df.iloc[2, 1:], errors='coerce').notna().any())
            
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
            length_series = pd.to_numeric(position_row[1:], errors='coerce')
            length_vector = np.array(length_series[~np.isnan(length_series)])  # Convert to numpy array
            
            print(f"  Length vector: {len(length_vector)} positions from {length_vector.min():.3f} to {length_vector.max():.3f} m")
            
            # Find all time rows (every 6th row starting from index 2)
            time_values = []
            time_row_indices = []
            
            for i in range(2, len(df), ASPEN_ROW_BLOCK_SIZE):
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
                        row_data = pd.to_numeric(pd.Series(row_data), errors='coerce').values
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
            
            # Extract timestamp and ramp parameters from filename
            file_timestamp = extract_timestamp_from_filename(file_path)
            ramp_parameters = extract_ramp_parameters_from_filename(file_path)
            
            metadata = DataMetadata(
                format_type=DataFormat.ASPEN_PLUS_DYNAMICS,
                source_file=file_path,
                dimensions={'n_time': n_time, 'm_length': m_length},
                time_range=(time_vector.min(), time_vector.max()),
                spatial_range=(length_vector.min(), length_vector.max()),
                variables=list(expected_variables),
                units=units,
                parsing_notes=parsing_issues,
                file_timestamp=file_timestamp,
                ramp_parameters=ramp_parameters
            )
            
            # Create standard data package
            data_package = StandardDataPackage(
                time_vector=time_vector,
                length_vector=length_vector,
                variables=variables,
                metadata=metadata
            )
            
            print(f"  ✓ Successfully parsed {n_time} time points × {m_length} spatial points")
            if file_timestamp:
                print(f"  ✓ Extracted timestamp: {file_timestamp}")
            if ramp_parameters:
                ramp_desc = f"{ramp_parameters.get('duration', '?')}min {ramp_parameters.get('direction', '?')}-{ramp_parameters.get('curve_shape', '?')}"
                print(f"  ✓ Extracted ramp parameters: {ramp_desc}")
            if parsing_issues:
                print(f"  ⚠ {len(parsing_issues)} parsing issues noted")
            
            return data_package
            
        except Exception as e:
            print(f"  ✗ Error parsing Aspen Dynamics format: {e}")
            return None
    
    def get_format_description(self) -> str:
        return "Aspen Plus Dynamics CSV export with spatial temperature and reaction data"

class GenericTimeSeriesParser:
    """Parser for generic time-series CSV files"""
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file is a generic time-series CSV"""
        try:
            # Try to read as standard CSV with headers
            df = pd.read_csv(file_path, nrows=5)
            
            # Check if it has time-like first column and numeric data
            first_col = df.iloc[:, 0]
            has_time_col = bool(
                'time' in df.columns[0].lower() or
                pd.to_numeric(first_col, errors='coerce').notna().all()
            )
            
            # Check if other columns are mostly numeric
            numeric_cols = 0
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().mean() > MIN_NUMERIC_COLUMN_RATIO:
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
            time_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            time_vector = np.array(time_series[~np.isnan(time_series)])
            
            # Extract variables (remaining columns)
            variables = {}
            for col in df.columns[1:]:
                data_series = pd.to_numeric(df[col], errors='coerce')
                data = np.array(data_series.values[:len(time_vector)])
                if not np.isnan(data).all():  # Skip completely empty columns
                    variables[col] = data.reshape(-1, 1)  # Make it 2D for consistency
            
            print(f"  Time vector: {len(time_vector)} points from {time_vector.min():.3f} to {time_vector.max():.3f}")
            print(f"  Variables: {list(variables.keys())}")
            
            # Create metadata
            file_timestamp = extract_timestamp_from_filename(file_path)
            ramp_parameters = extract_ramp_parameters_from_filename(file_path)
            
            metadata = DataMetadata(
                format_type=DataFormat.GENERIC_TIME_SERIES,
                source_file=file_path,
                dimensions={'n_time': len(time_vector), 'm_length': 1},
                time_range=(time_vector.min(), time_vector.max()),
                spatial_range=None,
                variables=list(variables.keys()),
                units={var: "unknown" for var in variables.keys()},
                file_timestamp=file_timestamp,
                ramp_parameters=ramp_parameters
            )
            
            # Create standard data package (no spatial dimension)
            data_package = StandardDataPackage(
                time_vector=time_vector,
                length_vector=None,  # No spatial data
                variables=variables,
                metadata=metadata
            )
            
            print(f"  ✓ Successfully parsed {len(time_vector)} time points, {len(variables)} variables")
            if file_timestamp:
                print(f"  ✓ Extracted timestamp: {file_timestamp}")
            if ramp_parameters:
                ramp_desc = f"{ramp_parameters.get('duration', '?')}min {ramp_parameters.get('direction', '?')}-{ramp_parameters.get('curve_shape', '?')}"
                print(f"  ✓ Extracted ramp parameters: {ramp_desc}")
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
    
    def load_data(self, file_path: str, preferred_format: Optional[DataFormat] = None) -> Optional[StandardDataPackage]:
        """Load data using the best available parser"""
        if not file_path or not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
        
        print(f"\n=== Data Loading: {os.path.basename(file_path)} ===")
        
        # Try preferred format first
        if preferred_format:
            for parser in self.parsers:
                if ((isinstance(parser, AspenDynamicsParser) and preferred_format == DataFormat.ASPEN_PLUS_DYNAMICS) or
                    (isinstance(parser, GenericTimeSeriesParser) and preferred_format == DataFormat.GENERIC_TIME_SERIES)) and parser.can_parse(file_path):
                    
                    result = parser.parse(file_path)
                    if result:
                        return result
        
        # Try all parsers in order
        for i, parser in enumerate(self.parsers):
            print(f"Trying parser {i+1}/{len(self.parsers)}: {parser.get_format_description()}")
            
            if parser.can_parse(file_path):
                print(f"  ✓ Format detected")
                result = parser.parse(file_path)
                if result:
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
                if isinstance(parser, AspenDynamicsParser):
                    return DataFormat.ASPEN_PLUS_DYNAMICS
                elif isinstance(parser, GenericTimeSeriesParser):
                    return DataFormat.GENERIC_TIME_SERIES
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported format descriptions"""
        return [parser.get_format_description() for parser in self.parsers]


