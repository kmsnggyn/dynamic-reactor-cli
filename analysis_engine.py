"""
Analysis Engine for Dynamic Reactor Ramp Analysis
================================================

Core analysis functionality separated from GUI for modularity and extensibility.
This module handles data loading, processing, analysis, and result generation.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.signal import find_peaks
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import contextlib
import io

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

@dataclass
class RampParameters:
    """Data class for ramp experiment parameters"""
    duration: Optional[int] = None
    direction: Optional[str] = None  # "up" or "down"
    curve_shape: Optional[str] = None  # "r" (linear) or "s" (sinusoidal)
    start_time: float = 10.0
    
    @property
    def end_time(self) -> Optional[float]:
        return self.start_time + self.duration if self.duration else None
    
    @property
    def is_ramp_up(self) -> bool:
        return self.direction == "up"
    
    @property
    def is_ramp_down(self) -> bool:
        return self.direction == "down"
    
    @property
    def is_linear(self) -> bool:
        return self.curve_shape == "r"
    
    @property
    def is_sinusoidal(self) -> bool:
        return self.curve_shape == "s"
    
    @property
    def curve_type(self) -> str:
        if self.is_linear:
            return "Linear"
        elif self.is_sinusoidal:
            return "Sinusoidal"
        else:
            return "Unknown"
    
    @property
    def analysis_title(self) -> str:
        if self.direction and self.curve_shape:
            return f"Flow Ramp-{self.direction.title()} ({self.curve_type} Curve)"
        elif self.is_ramp_up:
            return "Flow Ramp-Up"
        elif self.is_ramp_down:
            return "Flow Ramp-Down"
        else:
            return "Unknown Ramp Type"

@dataclass
class AnalysisOptions:
    """Data class for analysis options selected by user"""
    temperature_response: bool = True
    stability_analysis: bool = True
    spatial_gradients: bool = True
    heat_transfer_3d: bool = True
    temperature_difference: bool = True
    time_limit: Optional[float] = None

# Import data loading functionality from dedicated module
try:
    from data_loader import DataLoaderManager, load_and_parse_aspen_data, parse_ramp_parameters_from_filename
    print("Using new modular data loader")
    
    # Create wrapper class for compatibility
    class DataLoader:
        """Wrapper for new data loader functionality"""
        
        @staticmethod
        def load_and_parse_aspen_data(file_path: str) -> Optional[Dict[str, Any]]:
            """Load Aspen data using new data loader"""
            return load_and_parse_aspen_data(file_path)
        
        @staticmethod
        def parse_ramp_parameters_from_filename(file_path: str) -> RampParameters:
            """Parse ramp parameters using new data loader"""
            return parse_ramp_parameters_from_filename(file_path)
        
        @staticmethod
        def load_data_auto_detect(file_path: str) -> Optional[Dict[str, Any]]:
            """Auto-detect file format using new data loader"""
            loader = DataLoaderManager()
            data_package = loader.load_data(file_path)
            
            if data_package is None:
                return None
            
            # Convert to legacy format for compatibility
            return {
                'time_vector': data_package.time_vector,
                'length_vector': data_package.length_vector,
                'variables': data_package.variables,
                'file_path': file_path,
                'dimensions': data_package.metadata.dimensions,
                'format_type': data_package.metadata.format_type.value
            }

except ImportError:
    print("Warning: Could not import new data loader, using legacy implementation")
    
    class DataLoader:
        """Legacy data loader implementation"""
        
        @staticmethod
        def load_and_parse_aspen_data(file_path: str) -> Optional[Dict[str, Any]]:
            """Load raw Aspen CSV file and parse into vectors and matrices structure"""
            if not file_path:
                return None
            
            print(f"Loading: {os.path.basename(file_path)}")
            
            try:
                # Load raw CSV
                df = pd.read_csv(file_path, header=None)
                print(f"Raw data shape: {df.shape}")
                
                # Extract Length vector from row 2 (positions)
                position_row = df.iloc[2]
                length_vector = pd.to_numeric(position_row[1:], errors='coerce')
                length_vector = length_vector[~np.isnan(length_vector)]
                
                print(f"Length vector: {len(length_vector)} positions from {length_vector.min():.3f} to {length_vector.max():.3f} m")
                
                # Find all time rows
                time_values = []
                time_row_indices = []
                
                for i in range(2, len(df), 6):
                    if i < len(df):
                        time_val = pd.to_numeric(df.iloc[i, 0], errors='coerce')
                        if not np.isnan(time_val):
                            time_values.append(time_val)
                            time_row_indices.append(i)
                
                time_vector = np.array(time_values)
                print(f"Time vector: {len(time_vector)} points from {time_vector.min():.3f} to {time_vector.max():.3f} min")
                
                # Expected variables
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
                for t_idx, time_row_idx in enumerate(time_row_indices):
                    if t_idx >= n_time:
                        break
                    
                    for var_idx in range(len(expected_variables)):
                        data_row_idx = time_row_idx + 1 + var_idx
                        
                        if data_row_idx < len(df):
                            row_data = df.iloc[data_row_idx, 1:1+m_length].values
                            row_data = pd.to_numeric(row_data, errors='coerce')
                            variables[expected_variables[var_idx]][t_idx, :] = row_data
                
                # Data structure summary
                print(f"\n=== Data Structure Created ===")
                print(f"Time vector: shape ({len(time_vector)},)")
                print(f"Length vector: shape ({len(length_vector)},)")
                print(f"Variable matrices: shape ({n_time}, {m_length})")
                
                data_package = {
                    'time_vector': time_vector,
                    'length_vector': length_vector,
                    'variables': variables,
                    'file_path': file_path,
                    'dimensions': {'n_time': n_time, 'm_length': m_length},
                    'format_type': 'aspen_csv'
                }
                
                return data_package
                
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        
        @staticmethod
        def parse_ramp_parameters_from_filename(file_path: str) -> RampParameters:
            """Parse ramp parameters from filename convention: {duration}-{direction}-{curve_shape}"""
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
        
        @staticmethod
        def load_data_auto_detect(file_path: str) -> Optional[Dict[str, Any]]:
            """Auto-detect file format and load accordingly"""
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                # Try Aspen format first
                data_package = DataLoader.load_and_parse_aspen_data(file_path)
                if data_package is not None:
                    return data_package
                
                # Add other CSV format detection here in the future
                print(f"Warning: Could not parse CSV file in any known format")
                return None
            
            # Add other file format support here (Excel, JSON, etc.)
            print(f"Warning: Unsupported file format: {file_ext}")
            return None

class SteadyStateDetector:
    """Handles steady state detection logic"""
    
    @staticmethod
    def detect_steady_state(time_vector: np.ndarray, 
                          catalyst_temp_matrix: np.ndarray,
                          threshold: float = 0.05,
                          min_duration: float = 10,
                          search_start_time: Optional[float] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Detect steady state conditions"""
        
        print(f"\n=== Steady State Detection ===")
        print(f"Parameters: threshold={threshold:.3f} °C min⁻¹, min_duration={min_duration} min")
        
        # Calculate derivatives
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 0.05
        temp_derivatives = np.gradient(catalyst_temp_matrix, dt, axis=0)
        
        # Calculate stability metrics
        rms_change_rates = np.sqrt(np.nanmean(temp_derivatives**2, axis=1))
        max_abs_change_rates = np.nanmax(np.abs(temp_derivatives), axis=1)
        
        # Find stable periods
        stable_mask = (rms_change_rates < threshold) & (max_abs_change_rates < threshold * 2)
        
        # Remove isolated stable points
        stable_mask = SteadyStateDetector._remove_isolated_stable_points(stable_mask)
        
        # Add initial steady state for ramp experiments
        if search_start_time is not None and search_start_time >= 10.0:
            initial_steady_mask = time_vector < 10.0
            stable_mask = stable_mask | initial_steady_mask
        
        # Find steady state time
        steady_state_time = None
        steady_state_idx = None
        
        if search_start_time is not None:
            search_mask = time_vector >= search_start_time
            search_stable_mask = stable_mask & search_mask
            
            # Find first qualifying stable period
            stable_periods = SteadyStateDetector._find_stable_periods(
                time_vector, search_stable_mask, min_duration
            )
            
            if stable_periods:
                steady_state_idx = stable_periods[0][0]
                steady_state_time = time_vector[steady_state_idx]
                print(f"Steady state detected at t = {steady_state_time:.1f} min")
        
        stability_metrics = {
            'rms_change_rates': rms_change_rates,
            'max_abs_change_rates': max_abs_change_rates,
            'stable_mask': stable_mask,
            'threshold': threshold,
            'steady_state_idx': steady_state_idx,
            'min_rms_rate': np.nanmin(rms_change_rates),
            'min_max_rate': np.nanmin(max_abs_change_rates),
            'search_start_time': search_start_time
        }
        
        return steady_state_time, stability_metrics
    
    @staticmethod
    def _remove_isolated_stable_points(mask: np.ndarray, min_contiguous_length: int = 5) -> np.ndarray:
        """Remove isolated stable points, keep only contiguous stable regions"""
        if not np.any(mask):
            return mask
        
        # Find contiguous stable regions
        stable_regions = []
        in_stable_region = False
        region_start = None
        
        for i, is_stable in enumerate(mask):
            if is_stable and not in_stable_region:
                in_stable_region = True
                region_start = i
            elif not is_stable and in_stable_region:
                region_length = i - region_start
                stable_regions.append((region_start, i, region_length))
                in_stable_region = False
        
        if in_stable_region:
            region_length = len(mask) - region_start
            stable_regions.append((region_start, len(mask), region_length))
        
        # Create filtered mask
        filtered_mask = np.zeros_like(mask, dtype=bool)
        removed_count = 0
        
        for start_idx, end_idx, length in stable_regions:
            if length >= min_contiguous_length:
                filtered_mask[start_idx:end_idx] = True
            else:
                removed_count += length
        
        if removed_count > 0:
            print(f"  -> Removed {removed_count} isolated stable points")
        
        return filtered_mask
    
    @staticmethod
    def _find_stable_periods(time_vector: np.ndarray, 
                           stable_mask: np.ndarray, 
                           min_duration: float) -> List[Tuple[int, int, float]]:
        """Find continuous stable periods meeting minimum duration"""
        stable_periods = []
        in_stable_period = False
        period_start = None
        
        for i, is_stable in enumerate(stable_mask):
            if is_stable and not in_stable_period:
                period_start = i
                in_stable_period = True
            elif not is_stable and in_stable_period:
                period_duration = time_vector[i-1] - time_vector[period_start]
                if period_duration >= min_duration:
                    stable_periods.append((period_start, i-1, period_duration))
                in_stable_period = False
        
        if in_stable_period:
            period_duration = time_vector[-1] - time_vector[period_start]
            if period_duration >= min_duration:
                stable_periods.append((period_start, len(time_vector)-1, period_duration))
        
        return stable_periods

class AnalysisEngine:
    """Main analysis engine that coordinates all analysis operations"""
    
    def __init__(self):
        self.data_package = None
        self.ramp_params = None
        self.steady_state_time = None
        self.stability_metrics = None
        self.analysis_results = {}
    
    def load_data(self, file_path: str) -> bool:
        """Load and parse data from file"""
        try:
            self.data_package = DataLoader.load_data_auto_detect(file_path)
            if self.data_package is None:
                return False
            
            self.ramp_params = DataLoader.parse_ramp_parameters_from_filename(file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def run_steady_state_analysis(self, threshold: float = 0.05, min_duration: float = 10) -> bool:
        """Run steady state detection analysis"""
        if self.data_package is None:
            print("Error: No data loaded")
            return False
        
        try:
            catalyst_temp = self.data_package['variables']['T_cat (°C)']
            time_vector = self.data_package['time_vector']
            
            search_start_time = self.ramp_params.end_time if self.ramp_params.end_time else None
            
            self.steady_state_time, self.stability_metrics = SteadyStateDetector.detect_steady_state(
                time_vector, catalyst_temp, threshold, min_duration, search_start_time
            )
            
            return True
        except Exception as e:
            print(f"Error in steady state analysis: {e}")
            return False
    
    def _calculate_ramp_rate(self) -> float:
        """Calculate ramp rate as percentage change per minute"""
        try:
            if (self.ramp_params.duration is None or 
                self.ramp_params.direction is None or 
                self.ramp_params.duration == 'N/A' or 
                self.ramp_params.direction == 'N/A'):
                return 'N/A'
            
            # Convert duration to float if it's a string (e.g., "20" from "20min")
            duration = self.ramp_params.duration
            if isinstance(duration, str):
                # Remove "min" suffix if present
                duration = duration.replace('min', '').strip()
                try:
                    duration = float(duration)
                except (ValueError, TypeError):
                    return 'N/A'
            
            if duration <= 0:
                return 'N/A'
            
            # Calculate ramp rate based on direction
            direction = str(self.ramp_params.direction).lower()
            if direction in ['down', 'd']:
                # For down ramps: -90% / ramp_time
                ramp_rate = -90.0 / duration
            elif direction in ['up', 'u']:
                # For up ramps: 90% / ramp_time  
                ramp_rate = 90.0 / duration
            else:
                return 'N/A'
            
            return ramp_rate
            
        except Exception as e:
            print(f"Error calculating ramp rate: {e}")
            return 'N/A'
    
    def extract_key_metrics(self) -> Dict[str, Any]:
        """Extract key metrics for comparison"""
        if self.data_package is None:
            return {}
        
        catalyst_temp = self.data_package['variables']['T_cat (°C)']
        time_vector = self.data_package['time_vector']
        
        # Calculate temperature change rates
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 0.05
        temp_derivatives = np.gradient(catalyst_temp, dt, axis=0)
        
        # Get heat transfer data if available
        heat_transfer = self.data_package['variables'].get('Heat Transfer with coolant (kW/m2)')
        
        # Calculate key metrics with units
        metrics = {
            # File and experiment info
            'Source_File': {'value': os.path.basename(self.data_package['file_path']), 'unit': '-'},
            'Analysis_Timestamp': {'value': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), 'unit': '-'},
            
            # Ramp parameters
            'Ramp_Duration': {'value': self.ramp_params.duration if self.ramp_params.duration else 'N/A', 'unit': 'min'},
            'Ramp_Direction': {'value': self.ramp_params.direction if self.ramp_params.direction else 'N/A', 'unit': '-'},
            'Ramp_Curve_Type': {'value': self.ramp_params.curve_shape if self.ramp_params.curve_shape else 'N/A', 'unit': '-'},
            'Ramp_Start_Time': {'value': self.ramp_params.start_time, 'unit': 'min'},
            'Ramp_End_Time': {'value': self.ramp_params.end_time if self.ramp_params.end_time else 'N/A', 'unit': 'min'},
            
            # Calculate ramp rate: For "down" runs: -90%/ramp_time, for "up" runs: 90%/ramp_time
            'Ramp_Rate': {
                'value': self._calculate_ramp_rate(), 
                'unit': '%/min'
            },
            
            # Temperature extremes
            'Tcat_max': {'value': np.nanmax(catalyst_temp), 'unit': '°C'},
            'Tcat_min': {'value': np.nanmin(catalyst_temp), 'unit': '°C'},
            'Tcat_avg': {'value': np.nanmean(catalyst_temp), 'unit': '°C'},
            'Tcat_range': {'value': np.nanmax(catalyst_temp) - np.nanmin(catalyst_temp), 'unit': '°C'},
            'dTcat_dt_max_positive': {'value': np.nanmax(temp_derivatives), 'unit': '°C/min'},
            'dTcat_dt_max_negative': {'value': np.nanmin(temp_derivatives), 'unit': '°C/min'},
            'dTcat_dt_max_abs': {'value': np.nanmax(np.abs(temp_derivatives)), 'unit': '°C/min'},
            'dTcat_dt_rms': {'value': np.sqrt(np.nanmean(temp_derivatives**2)), 'unit': '°C/min'},
            'Steady_State_Detected': {'value': 'Yes' if self.steady_state_time is not None else 'No', 'unit': '-'},
            'Steady_State_Time': {'value': self.steady_state_time if self.steady_state_time is not None else 'N/A', 'unit': 'min'},
            'Settling_Time': {'value': (self.steady_state_time - self.ramp_params.end_time) if (self.steady_state_time is not None and self.ramp_params.end_time) else 'N/A', 'unit': 'min'},
            'Stability_RMS_Threshold': {'value': self.stability_metrics['threshold'] if self.stability_metrics else 'N/A', 'unit': '°C/min'},
            'Stability_Min_RMS_Rate': {'value': self.stability_metrics['min_rms_rate'] if self.stability_metrics else 'N/A', 'unit': '°C/min'},
            'Tcat_spatial_diff_max': {'value': np.nanmax(np.nanmax(catalyst_temp, axis=1) - np.nanmin(catalyst_temp, axis=1)), 'unit': '°C'},
            'Tcat_spatial_diff_avg': {'value': np.nanmean(np.nanmax(catalyst_temp, axis=1) - np.nanmin(catalyst_temp, axis=1)), 'unit': '°C'},
            'Data_Time_Points': {'value': self.data_package['dimensions']['n_time'], 'unit': 'count'},
            'Data_Length_Points': {'value': self.data_package['dimensions']['m_length'], 'unit': 'count'},
            'Time_Range': {'value': f"{time_vector.min():.1f}-{time_vector.max():.1f}", 'unit': 'min'},
            'Reactor_Length': {'value': self.data_package['length_vector'].max() - self.data_package['length_vector'].min(), 'unit': 'm'},
        }
        
        # Add heat transfer metrics if available
        if heat_transfer is not None:
            metrics.update({
                'Heat_Transfer_avg': {'value': np.nanmean(heat_transfer), 'unit': 'kW/m²'},
                'Heat_Transfer_max': {'value': np.nanmax(heat_transfer), 'unit': 'kW/m²'},
                'Heat_Transfer_min': {'value': np.nanmin(heat_transfer), 'unit': 'kW/m²'}
            })
        else:
            metrics.update({
                'Heat_Transfer_avg': {'value': 'N/A', 'unit': 'kW/m²'},
                'Heat_Transfer_max': {'value': 'N/A', 'unit': 'kW/m²'},
                'Heat_Transfer_min': {'value': 'N/A', 'unit': 'kW/m²'}
            })
        
        return metrics
    
    def run_full_analysis(self, options: AnalysisOptions) -> Dict[str, Any]:
        """Run complete analysis based on options"""
        if self.data_package is None:
            return {}
        
        results = {}
        
        # Always run steady state analysis
        self.run_steady_state_analysis()
        
        # Extract key metrics
        results['metrics'] = self.extract_key_metrics()
        
        # Store analysis components for plot generation
        results['data_package'] = self.data_package
        results['ramp_params'] = self.ramp_params
        results['steady_state_time'] = self.steady_state_time
        results['stability_metrics'] = self.stability_metrics
        results['options'] = options
        
        return results
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        if self.data_package is None:
            print("No data loaded for analysis")
            return
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # File information
        print(f"Source file: {os.path.basename(self.data_package['file_path'])}")
        print(f"Data dimensions: {self.data_package['dimensions']['n_time']} time points × {self.data_package['dimensions']['m_length']} positions")
        print(f"Time range: {self.data_package['time_vector'].min():.1f} - {self.data_package['time_vector'].max():.1f} min")
        print(f"Reactor length: {self.data_package['length_vector'].min():.3f} - {self.data_package['length_vector'].max():.3f} m")
        
        # Ramp parameters
        print(f"\nRamp Configuration:")
        if self.ramp_params.duration:
            print(f"  Duration: {self.ramp_params.duration} minutes")
            print(f"  Direction: {self.ramp_params.direction.upper()}")
            print(f"  Curve type: {self.ramp_params.curve_type}")
            print(f"  Period: t = {self.ramp_params.start_time:.1f} - {self.ramp_params.end_time:.1f} min")
        else:
            print(f"  Type: {self.ramp_params.analysis_title}")
            print(f"  Auto-detected from filename")
        
        # Steady state analysis
        print(f"\nSteady State Analysis:")
        if self.steady_state_time is not None:
            print(f"  Status: ✓ Detected")
            print(f"  Time: t = {self.steady_state_time:.1f} min")
            if self.ramp_params.end_time:
                settling_time = self.steady_state_time - self.ramp_params.end_time
                print(f"  Settling time: {settling_time:.1f} min")
            print(f"  RMS stability: {self.stability_metrics['min_rms_rate']:.4f} °C min⁻¹")
        else:
            print(f"  Status: ✗ Not detected")
            if self.stability_metrics:
                print(f"  Minimum RMS observed: {self.stability_metrics['min_rms_rate']:.4f} °C min⁻¹")
                print(f"  Current threshold: {self.stability_metrics['threshold']:.3f} °C min⁻¹")
        
        # Temperature statistics
        catalyst_temp = self.data_package['variables']['T_cat (°C)']
        print(f"\nTemperature Statistics:")
        print(f"  Range: {np.nanmin(catalyst_temp):.1f} - {np.nanmax(catalyst_temp):.1f} °C")
        print(f"  Average: {np.nanmean(catalyst_temp):.1f} °C")
        
        # System performance
        max_temp_change = np.nanmax(np.abs(np.diff(catalyst_temp, axis=0)))
        print(f"  Max temperature change rate: {max_temp_change:.3f} °C/time_step")
