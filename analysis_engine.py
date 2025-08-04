"""
Analysis Engine for Dynamic Reactor Ramp Analysis
================================================

Core analysis functionality separated from GUI for modularity and extensibility.
This module handles data loading, processing, analysis, and result generation.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

# Analysis Configuration Constants
DEFAULT_STEADY_STATE_THRESHOLD = 0.05  # °C/min
DEFAULT_MIN_DURATION = 10.0  # minutes
DEFAULT_RAMP_START_TIME = 10.0  # minutes
DEFAULT_MIN_CONTIGUOUS_LENGTH = 5  # data points
HOT_SPOT_STABILITY_THRESHOLD = 0.1  # meters
RAPID_CHANGE_SIGMA_MULTIPLIER = 3.0  # standard deviations

# Matrix and Array Constants
MIN_MATRIX_DIMENSIONS = 2  # Minimum dimensions for 2D temperature matrices
MIN_ARRAY_SIZE = 0  # Minimum size for arrays
MIN_POSITIVE_VALUE = 0  # Minimum value for positive parameters

@dataclass
class RampParameters:
    """
    Data class for ramp experiment parameters.
    
    Stores all parameters related to reactor ramp experiments including timing,
    direction, and curve shape characteristics.
    
    Attributes:
        duration: Ramp duration in minutes
        direction: Ramp direction ("up" or "down")
        curve_shape: Curve type ("r" for linear, "s" for sinusoidal)
        start_time: When the ramp begins (minutes from simulation start)
    """
    duration: Optional[int] = None
    direction: Optional[str] = None  # "up" or "down"
    curve_shape: Optional[str] = None  # "r" (linear) or "s" (sinusoidal)
    start_time: float = DEFAULT_RAMP_START_TIME
    
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
    """
    Data class for analysis options selected by user.
    
    Controls which analysis components and plots are generated during
    the reactor analysis process.
    
    Attributes:
        temperature_response: Generate temperature vs time plots
        stability_analysis: Perform steady-state detection analysis  
        spatial_gradients: Calculate spatial temperature gradients
        heat_transfer_3d: Generate 3D heat transfer visualizations
        temperature_difference: Calculate catalyst-bulk temperature differences
        time_limit: Optional time limit for analysis (minutes)
    """
    temperature_response: bool = True
    stability_analysis: bool = True
    spatial_gradients: bool = True
    heat_transfer_3d: bool = True
    temperature_difference: bool = True
    time_limit: Optional[float] = None

# Import data loading functionality from dedicated module
try:
    from importlib import import_module
    data_loader_module = import_module('data_loader')
    DataLoaderManager = data_loader_module.DataLoaderManager
    StandardDataPackage = data_loader_module.StandardDataPackage
    DataMetadata = data_loader_module.DataMetadata
    DataFormat = data_loader_module.DataFormat
    print("Using new modular data loader")
    
    # Simple data loading wrapper using the imported DataLoaderManager
    class DataLoader:
        """Unified DataLoader interface using modular data loader"""
        @staticmethod
        def load_data(file_path: str) -> Optional[Any]:
            loader = DataLoaderManager()
            return loader.load_data(file_path)

        @staticmethod
        def parse_ramp_parameters_from_filename(file_path: str) -> RampParameters:
            # Delegate to the imported module if available, else fallback
            if hasattr(data_loader_module, 'parse_ramp_parameters_from_filename'):
                return data_loader_module.parse_ramp_parameters_from_filename(file_path)
            # Fallback: basic parsing
            filename = os.path.basename(file_path).lower()
            ramp_params = RampParameters()
            if "up" in filename:
                ramp_params.direction = "up"
            elif "down" in filename:
                ramp_params.direction = "down"
            return ramp_params

except ImportError:
    print("Warning: Could not import new data loader, using minimal fallback")
    
    # Define minimal types for fallback
    StandardDataPackage = Any
    
    # Use single DataLoader class for both cases
    class DataLoader:
        """Minimal fallback implementation when data_loader module unavailable"""
        
        @staticmethod
        def load_data(file_path: str) -> Optional[Dict[str, Any]]:
            """Minimal fallback data loader"""
            print(f"Error: Modern data loader not available. Please check data_loader module.")
            return None
        
        @staticmethod
        def parse_ramp_parameters_from_filename(file_path: str) -> RampParameters:
            """Minimal ramp parameter parsing"""
            filename = os.path.basename(file_path).lower()
            ramp_params = RampParameters()
            
            if "up" in filename:
                ramp_params.direction = "up"
            elif "down" in filename:
                ramp_params.direction = "down"
                
            return ramp_params

def _import_module_safely(module_name: str):
    """Helper function to safely import numbered modules"""
    try:
        from importlib import import_module
        return import_module(module_name)
    except ImportError:
        return None

class SteadyStateDetector:
    """
    Handles steady state detection logic for reactor temperature data.
    
    Provides static methods for detecting when a reactor system reaches
    steady-state conditions based on temperature change rates and stability criteria.
    
    The detection algorithm:
    1. Calculates temperature derivatives over time
    2. Identifies periods where change rates are below threshold
    3. Filters out isolated stable points
    4. Finds continuous stable periods meeting minimum duration
    """
    
    @staticmethod
    def detect_steady_state(time_vector: np.ndarray, 
                          catalyst_temp_matrix: np.ndarray,
                          threshold: float = DEFAULT_STEADY_STATE_THRESHOLD,
                          min_duration: float = DEFAULT_MIN_DURATION,
                          search_start_time: Optional[float] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Detect steady state conditions in reactor temperature data.
        
        Args:
            time_vector: 1D array of time points (minutes)
            catalyst_temp_matrix: 2D array [time, position] of catalyst temperatures (°C)
            threshold: Maximum RMS temperature change rate for steady state (°C/min)
            min_duration: Minimum duration for a valid steady state period (minutes)
            search_start_time: Start searching for steady state after this time (minutes)
            
        Returns:
            Tuple containing:
            - steady_state_time: Time when steady state begins (minutes), or None if not found
            - stability_metrics: Dictionary with stability analysis results
            
        Example:
            >>> time = np.linspace(0, 100, 1000)  # 100 minutes
            >>> temps = np.random.normal(500, 1, (1000, 10))  # Stable temperatures
            >>> steady_time, metrics = detect_steady_state(time, temps)
            >>> print(f"Steady state at: {steady_time:.1f} min")
        """
        # Input validation
        if len(time_vector) == MIN_ARRAY_SIZE or catalyst_temp_matrix.size == MIN_ARRAY_SIZE:
            raise ValueError("Empty time vector or temperature matrix provided")
        
        if len(time_vector) != catalyst_temp_matrix.shape[0]:
            raise ValueError(f"Time vector length ({len(time_vector)}) must match temperature matrix first dimension ({catalyst_temp_matrix.shape[0]})")
        
        if threshold <= MIN_POSITIVE_VALUE:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        
        if min_duration <= MIN_POSITIVE_VALUE:
            raise ValueError(f"Minimum duration must be positive, got {min_duration}")
        
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
        if search_start_time is not None and search_start_time >= DEFAULT_RAMP_START_TIME:
            initial_steady_mask = time_vector < DEFAULT_RAMP_START_TIME
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
    def _remove_isolated_stable_points(mask: np.ndarray, min_contiguous_length: int = DEFAULT_MIN_CONTIGUOUS_LENGTH) -> np.ndarray:
        """
        Remove isolated stable points, keep only contiguous stable regions.
        
        Filters out brief stable periods that don't meet the minimum contiguous
        length requirement, reducing noise in steady-state detection.
        
        Args:
            mask: Boolean array indicating stable points
            min_contiguous_length: Minimum number of consecutive stable points required
            
        Returns:
            Filtered boolean mask with isolated points removed
        """
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
                if region_start is not None:
                    region_length = i - region_start
                    stable_regions.append((region_start, i, region_length))
                in_stable_region = False
        
        if in_stable_region and region_start is not None:
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
    """
    Core analysis engine that processes raw data and creates comprehensive analysis package.
    
    Architecture:
    Raw CSV → DataLoader → AnalysisEngine → Complete Analysis Package
                                  ↓
                            Saved vectors, matrices, scalars, metrics
    
    This class:
    1. Takes vectors and matrices from DataLoader
    2. Computes ALL derived variables (gradients, optima, etc.)
    3. Calculates ALL metrics (steady-state, ramp rates, etc.)
    4. Creates comprehensive data package
    5. Optionally saves complete analysis for reuse
    
    The engine handles various reactor configurations and provides flexible
    analysis through configurable parameters. It integrates steady state
    detection, spatial/temporal gradient analysis, and statistical evaluation.
    
    Attributes:
        analysis_package: Dictionary containing all computed analysis results
        detector: SteadyStateDetector instance for steady state analysis
        config: Optional configuration dictionary for analysis parameters
        
    Example:
        >>> engine = AnalysisEngine(config={'threshold': 0.05})
        >>> results = engine.run_analysis(data_package, ramp_params, options)
        >>> print(f"Analysis package contains {len(results)} components")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.analysis_package = {}
        self.save_intermediate_data = True  # Option to save analysis packages
        # Legacy attributes for compatibility with GUI
        self.data_package: Optional[Dict[str, Any]] = None
        self.ramp_params: Optional[RampParameters] = None
        self.steady_state_time: Optional[float] = None
        self.stability_metrics: Optional[Dict[str, Any]] = None
        
        # Store configuration for analysis parameters
        self.config = config or {
            'steady_state': {
                'threshold': 0.05,
                'min_duration': 10.0
            }
        }
    
    def extract_key_metrics(self) -> Dict[str, Any]:
        """
        Extract key metrics for results comparison table.
        This method provides compatibility with the GUI's results comparison system.
        """
        if not self.analysis_package and not self.data_package:
            return {}
        
        # Use analysis_package if available, otherwise fall back to data_package
        package = self.analysis_package if self.analysis_package else self.data_package
        
        metrics = {}
        
        # Add source file information
        if hasattr(self, 'data_package') and self.data_package and 'file_path' in self.data_package:
            metrics['Source_File'] = self.data_package['file_path']
        
        # Add ramp parameters if available
        if self.ramp_params:
            metrics['Ramp_Duration'] = f"{self.ramp_params.duration:.1f}" if self.ramp_params.duration else "N/A"
            metrics['Ramp_Direction'] = self.ramp_params.direction
            metrics['Ramp_Curve_Type'] = self.ramp_params.curve_type
            
            # Calculate ramp rate if we have duration and direction
            if self.ramp_params.duration and self.ramp_params.duration > 0:
                if self.ramp_params.end_time and self.ramp_params.start_time:
                    metrics['Ramp_Rate'] = f"{(self.ramp_params.end_time - self.ramp_params.start_time) / self.ramp_params.duration:.3f}"
                else:
                    metrics['Ramp_Rate'] = f"{self.ramp_params.duration:.3f}"
            else:
                metrics['Ramp_Rate'] = "N/A"
        
        # Add steady state time
        if self.steady_state_time:
            metrics['Steady_State_Time'] = f"{self.steady_state_time:.2f}"
        
        # Add stability metrics
        if self.stability_metrics:
            if 'threshold' in self.stability_metrics:
                metrics['Stability_Threshold'] = f"{self.stability_metrics['threshold']:.4f}"
            if 'min_rms_rate' in self.stability_metrics:
                metrics['Min_RMS_Rate'] = f"{self.stability_metrics['min_rms_rate']:.6f}"
        
        # Extract temperature metrics from package
        if package:
            # Try to get temperature data
            temp_matrix = None
            time_vector = None
            
            if 'catalyst_temp_matrix' in package:
                temp_matrix = package['catalyst_temp_matrix']
                time_vector = package.get('time_vector')
            elif 'variables' in package and 'T_cat (°C)' in package['variables']:
                temp_matrix = package['variables']['T_cat (°C)']
                time_vector = package.get('time_vector')
            
            # Ensure temp_matrix is a numpy array
            if temp_matrix is not None and time_vector is not None:
                # Convert to numpy array if needed
                if hasattr(temp_matrix, 'values'):
                    temp_matrix = temp_matrix.values
                elif not isinstance(temp_matrix, np.ndarray):
                    try:
                        temp_matrix = np.array(temp_matrix)
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        temp_matrix = None
                
                # Only proceed if we have a valid numpy array
                if temp_matrix is not None and isinstance(temp_matrix, np.ndarray) and temp_matrix.size > 0:
                    # Calculate basic temperature metrics
                    max_temp = np.max(temp_matrix)
                    min_temp = np.min(temp_matrix)
                    avg_temp = np.mean(temp_matrix)
                    temp_range = max_temp - min_temp
                    
                    metrics['Max_Temperature'] = f"{max_temp:.2f}"
                    metrics['Min_Temperature'] = f"{min_temp:.2f}"
                    metrics['Avg_Temperature'] = f"{avg_temp:.2f}"
                    metrics['Temperature_Range'] = f"{temp_range:.2f}"
                    
                    # Find position of maximum temperature
                    if temp_matrix.ndim >= MIN_MATRIX_DIMENSIONS:
                        max_pos = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
                        if len(package.get('length_vector', [])) > max_pos[1]:
                            max_temp_position = package['length_vector'][max_pos[1]]
                            metrics['Max_Temp_Position'] = f"{max_temp_position:.3f}"
                    
                    # Calculate final temperature (last time point, reactor exit)
                    if temp_matrix.ndim >= MIN_MATRIX_DIMENSIONS and temp_matrix.shape[0] > MIN_ARRAY_SIZE and temp_matrix.shape[1] > MIN_ARRAY_SIZE:
                        final_temp = temp_matrix[-1, -1]
                        metrics['Final_Temperature'] = f"{final_temp:.2f}"
        
        # Add units for metrics
        units = {
            'Source_File': '-',
            'Ramp_Duration': 'min',
            'Ramp_Direction': '-',
            'Ramp_Curve_Type': '-',
            'Ramp_Rate': 'min⁻¹',
            'Steady_State_Time': 'min',
            'Stability_Threshold': '°C',
            'Min_RMS_Rate': '°C/min',
            'Max_Temperature': '°C',
            'Min_Temperature': '°C',
            'Avg_Temperature': '°C', 
            'Temperature_Range': '°C',
            'Max_Temp_Position': 'm',
            'Final_Temperature': '°C'
        }
        
        # Add units to the metrics
        metrics['Units'] = units
        
        return metrics
        
    def run_complete_analysis(self, file_path: str, time_limit: Optional[float] = None, 
                            save_package: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline and return comprehensive analysis package.
        
        Returns:
            Complete analysis package with:
            - Original vectors/matrices from DataLoader
            - Derived variables (gradients, rate changes, etc.)
            - Scalar metrics (steady state time, max temps, etc.)
            - Analysis metadata
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS ENGINE")
        print("="*80)
        
        # Step 1: Load raw data using modern data loader
        print("Step 1: Loading raw data...")
        data_package_raw = DataLoader.load_data(file_path)
        if data_package_raw is None:
            raise ValueError("Failed to load data from file")
        
        # Handle both StandardDataPackage and dict types safely
        data_package = data_package_raw  # Use Any type to avoid type checking issues
        
        # Extract ramp parameters from metadata (preferred) or filename (fallback)
        try:
            # Try accessing as StandardDataPackage first
            metadata = getattr(data_package, 'metadata', None)
            if metadata and hasattr(metadata, 'ramp_parameters') and metadata.ramp_parameters:
                ramp_params = RampParameters(
                    duration=metadata.ramp_parameters.get('duration'),
                    direction=metadata.ramp_parameters.get('direction'),
                    curve_shape=metadata.ramp_parameters.get('curve_shape'),
                    start_time=metadata.ramp_parameters.get('start_time', 10.0)
                )
                print(f"✓ Ramp parameters from metadata: {ramp_params.duration}min {ramp_params.direction}-{ramp_params.curve_shape}")
            else:
                ramp_params = DataLoader.parse_ramp_parameters_from_filename(file_path)
                print("⚠ Using fallback ramp parameter parsing")
        except (AttributeError, TypeError):
            ramp_params = DataLoader.parse_ramp_parameters_from_filename(file_path)
            print("⚠ Using fallback ramp parameter parsing (error)")
        
        # Get data dimensions
        try:
            n_time = getattr(data_package, 'n_time', None)
            n_spatial = getattr(data_package, 'n_spatial', None)
            if n_time and n_spatial:
                print(f"✓ Raw data loaded: {n_time} time points, {n_spatial} positions")
            else:
                print("✓ Raw data loaded")
        except (AttributeError, TypeError):
            print("✓ Raw data loaded")
        
        # Convert to legacy format for compatibility with existing analysis methods
        try:
            # Try accessing as StandardDataPackage first
            time_vector = getattr(data_package, 'time_vector', None)
            if time_vector is not None:
                # It's a StandardDataPackage
                raw_data = {
                    'time_vector': time_vector,
                    'length_vector': getattr(data_package, 'length_vector', []),
                    'variables': getattr(data_package, 'variables', {}),
                    'file_path': file_path,
                    'dimensions': getattr(getattr(data_package, 'metadata', None), 'dimensions', {}),
                    'format_type': 'StandardDataPackage'
                }
                # Try to get format type safely
                metadata = getattr(data_package, 'metadata', None)
                if metadata:
                    format_type = getattr(metadata, 'format_type', None)
                    if format_type:
                        if hasattr(format_type, 'value'):
                            raw_data['format_type'] = format_type.value
                        else:
                            raw_data['format_type'] = str(format_type)
            else:
                # It's already a dictionary
                raw_data = data_package if isinstance(data_package, dict) else {}
        except (AttributeError, TypeError) as e:
            print(f"Warning: Error accessing data package attributes: {e}")
            # Fallback to treating as dictionary
            raw_data = data_package if isinstance(data_package, dict) else {}
        
        # Step 2: Process core variables
        print("\nStep 2: Processing core variables...")
        analysis_package = self._process_core_variables(raw_data, ramp_params, time_limit)
        
        # Step 3: Compute derived variables
        print("\nStep 3: Computing derived variables...")
        self._compute_derived_variables(analysis_package)
        
        # Step 4: Calculate scalar metrics
        print("\nStep 4: Calculating scalar metrics...")
        self._calculate_scalar_metrics(analysis_package)
        
        # Step 5: Detect key features (optima, transitions, etc.)
        print("\nStep 5: Detecting key features...")
        self._detect_key_features(analysis_package)
        
        # Step 6: Generate analysis metadata
        print("\nStep 6: Generating analysis metadata...")
        self._generate_metadata(analysis_package, file_path)
        
        # Step 7: Optionally save complete package
        if save_package:
            print("\nStep 7: Saving analysis package...")
            self._save_analysis_package(analysis_package, file_path)
        
        print("\n✓ Complete analysis finished!")
        print(f"Analysis package contains {len(analysis_package)} main categories")
        
        self.analysis_package = analysis_package
        return analysis_package
    
    def load_analysis_package(self, package_dir: str) -> Dict[str, Any]:
        """
        Load a previously saved analysis package from directory.
        
        Args:
            package_dir: Directory containing the saved analysis package files
            
        Returns:
            Complete analysis package dictionary
        """
        if not os.path.exists(package_dir):
            raise FileNotFoundError(f"Analysis package directory not found: {package_dir}")
        
        print(f"Loading analysis package from: {package_dir}")
        
        # Load vectors and matrices
        vectors_file = os.path.join(package_dir, "vectors.npz")
        matrices_file = os.path.join(package_dir, "matrices.npz") 
        derived_file = os.path.join(package_dir, "derived_variables.npz")
        
        # Load JSON files
        metrics_file = os.path.join(package_dir, "scalar_metrics.json")
        features_file = os.path.join(package_dir, "key_features.json")
        metadata_file = os.path.join(package_dir, "metadata.json")
        
        # Initialize package with consistent structure
        package = {
            'core_vectors': {},
            'core_matrices': {},
            'derived_variables': {},
            'scalar_metrics': {},
            'key_features': {},
            'metadata': {}
        }
        
        # Load numpy data
        if os.path.exists(vectors_file):
            vectors_data = np.load(vectors_file)
            package['core_vectors'] = {key: vectors_data[key] for key in vectors_data.files}
            
        if os.path.exists(matrices_file):
            matrices_data = np.load(matrices_file)
            package['core_matrices'] = {key: matrices_data[key] for key in matrices_data.files}
            
        if os.path.exists(derived_file):
            derived_data = np.load(derived_file)
            package['derived_variables'] = {key: derived_data[key] for key in derived_data.files}
        
        # Load JSON data
        import json
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                package['scalar_metrics'] = json.load(f)
                
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                package['key_features'] = json.load(f)
                
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                package['metadata'] = json.load(f)
        
        print(f"✓ Analysis package loaded successfully")
        print(f"  - Core vectors: {len(package['core_vectors'])} items")
        print(f"  - Core matrices: {len(package['core_matrices'])} items") 
        print(f"  - Derived variables: {len(package['derived_variables'])} items")
        print(f"  - Scalar metrics: {len(package['scalar_metrics'])} items")
        print(f"  - Key features: {len(package['key_features'])} items")
        
        # Store in instance
        self.analysis_package = package
        return package
    
    def _process_core_variables(self, raw_data: Dict[str, Any], ramp_params: RampParameters, 
                              time_limit: Optional[float]) -> Dict[str, Any]:
        """Process and filter core variables from raw data"""
        
        # Extract core vectors
        time_vector = raw_data['time_vector']
        length_vector = raw_data['length_vector']
        
        # Convert to numpy arrays if they're pandas objects
        if hasattr(time_vector, 'values'):
            time_vector = time_vector.values
        if hasattr(length_vector, 'values'):
            length_vector = length_vector.values
        
        # Apply time limit if specified
        if time_limit is not None:
            time_mask = time_vector <= time_limit
            time_vector = time_vector[time_mask]
        else:
            time_mask = np.ones(len(time_vector), dtype=bool)
        
        # Extract and filter matrices
        variables = raw_data['variables']
        processed_matrices = {}
        
        for var_name, matrix in variables.items():
            processed_matrices[var_name] = matrix[time_mask, :] if time_limit else matrix
            print(f"  • {var_name}: {processed_matrices[var_name].shape}")
        
        # Create analysis package structure
        package = {
            'core_vectors': {
                'time_vector': time_vector,
                'length_vector': length_vector
            },
            'core_matrices': processed_matrices,
            'ramp_parameters': ramp_params,
            'derived_variables': {},
            'scalar_metrics': {},
            'key_features': {},
            'metadata': {}
        }
        
        print(f"  • Core processing complete: {len(time_vector)} time points")
        return package
    
    def _compute_derived_variables(self, package: Dict[str, Any]):
        """Compute all derived variables (gradients, rates, differences, etc.)"""
        
        time_vector = package['core_vectors']['time_vector']
        length_vector = package['core_vectors']['length_vector']
        core_matrices = package['core_matrices']
        
        derived = package['derived_variables']
        
        # Convert vectors to numpy arrays if they're pandas Series
        time_array = time_vector.values if hasattr(time_vector, 'values') else time_vector
        length_array = length_vector.values if hasattr(length_vector, 'values') else length_vector
        
        # Calculate spacing for gradients
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.05
        dx = length_array[1] - length_array[0] if len(length_array) > 1 else 0.1
        
        # Temperature gradients
        if 'T_cat (°C)' in core_matrices:
            catalyst_temp = core_matrices['T_cat (°C)']
            
            # Spatial gradients (∂T/∂x)
            spatial_gradients = np.gradient(catalyst_temp, dx, axis=1)
            temporal_gradients = np.gradient(catalyst_temp, dt, axis=0)
            
            derived['spatial_temperature_gradient'] = spatial_gradients
            derived['temporal_temperature_gradient'] = temporal_gradients
            print(f"  • Temperature gradients: spatial ({spatial_gradients.shape}), temporal ({temporal_gradients.shape})")
            
            # Temperature differences (if bulk temp available)
            if 'T (°C)' in core_matrices:
                bulk_temp = core_matrices['T (°C)']
                temp_difference = catalyst_temp - bulk_temp
                derived['temperature_difference'] = temp_difference
                print(f"  • Temperature difference (Tcat - T): {temp_difference.shape}")
                
                # Heat transfer indication
        
        # Reaction rate analysis
        if 'Reaction Rate (kmol/m3/hr)' in core_matrices:
            reaction_rate = core_matrices['Reaction Rate (kmol/m3/hr)']
            
            # Rate gradients
            rate_spatial_grad = np.gradient(reaction_rate, dx, axis=1)
            rate_temporal_grad = np.gradient(reaction_rate, dt, axis=0)
            
            derived['reaction_rate_spatial_gradient'] = rate_spatial_grad
            derived['reaction_rate_temporal_gradient'] = rate_temporal_grad
            print(f"  • Reaction rate gradients: computed")
    
    def _calculate_scalar_metrics(self, package: Dict[str, Any]):
        """Calculate all scalar metrics and key numbers"""
        
        time_vector = package['core_vectors']['time_vector']
        length_vector = package['core_vectors']['length_vector']
        core_matrices = package['core_matrices']
        ramp_params = package['ramp_parameters']
        
        metrics = package['scalar_metrics']
        
        # Steady state detection
        if 'T_cat (°C)' in core_matrices:
            catalyst_temp = core_matrices['T_cat (°C)']
            
            search_start_time = ramp_params.end_time if ramp_params.end_time else None
            steady_state_config = self.config['steady_state']
            steady_state_time, stability_metrics = SteadyStateDetector.detect_steady_state(
                time_vector, catalyst_temp,
                threshold=steady_state_config['threshold'],
                min_duration=steady_state_config['min_duration'],
                search_start_time=search_start_time
            )
            
            metrics['steady_state_time'] = steady_state_time
            metrics['stability_metrics'] = stability_metrics
            print(f"  • Steady state time: {steady_state_time:.1f} min" if steady_state_time else "  • Steady state: not detected")
        
        # Temperature statistics
        if 'T_cat (°C)' in core_matrices:
            catalyst_temp = core_matrices['T_cat (°C)']
            
            metrics['temperature_stats'] = {
                'min_temp': np.nanmin(catalyst_temp),
                'max_temp': np.nanmax(catalyst_temp),
                'mean_temp': np.nanmean(catalyst_temp),
                'temp_range': np.nanmax(catalyst_temp) - np.nanmin(catalyst_temp),
                'final_temp': catalyst_temp[-1, :].mean() if len(catalyst_temp) > 0 else np.nan
            }
            print(f"  • Temperature range: {metrics['temperature_stats']['min_temp']:.1f} - {metrics['temperature_stats']['max_temp']:.1f} °C")
        
        # Ramp analysis
        if ramp_params.duration and ramp_params.start_time and ramp_params.end_time:
            ramp_mask = (time_vector >= ramp_params.start_time) & (time_vector <= ramp_params.end_time)
            
            if 'T_cat (°C)' in core_matrices and np.any(ramp_mask):
                catalyst_temp = core_matrices['T_cat (°C)']
                ramp_temps = catalyst_temp[ramp_mask, :]
                
                initial_temp = ramp_temps[0, :].mean()
                final_temp = ramp_temps[-1, :].mean()
                temp_change = final_temp - initial_temp
                
                # Calculate ramp rate
                ramp_rate = temp_change / ramp_params.duration
                
                metrics['ramp_analysis'] = {
                    'initial_temp': initial_temp,
                    'final_temp': final_temp,
                    'temperature_change': temp_change,
                    'ramp_rate': ramp_rate,
                    'duration': ramp_params.duration,
                    'direction': ramp_params.direction
                }
                print(f"  • Ramp rate: {ramp_rate:.3f} °C/min ({ramp_params.direction})")
        
        # Process all matrices for min/max locations
        for var_name, matrix in core_matrices.items():
            if np.isfinite(matrix).any():
                min_val = np.nanmin(matrix)
                max_val = np.nanmax(matrix)
                min_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
                max_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
                
                metrics[f'{var_name}_extrema'] = {
                    'min_value': min_val,
                    'max_value': max_val,
                    'min_location': {
                        'time': time_vector[min_idx[0]],
                        'position': length_vector[min_idx[1]]
                    },
                    'max_location': {
                        'time': time_vector[max_idx[0]], 
                        'position': length_vector[max_idx[1]]
                    }
                }
    
    def _detect_key_features(self, package: Dict[str, Any]):
        """Detect key features like optima, transitions, hot spots, etc."""
        
        time_vector = package['core_vectors']['time_vector']
        length_vector = package['core_vectors']['length_vector']
        core_matrices = package['core_matrices']
        
        features = package['key_features']
        
        # Hot spot detection
        if 'T_cat (°C)' in core_matrices:
            catalyst_temp = core_matrices['T_cat (°C)']
            
            # Find maximum temperature at each time point
            max_temp_positions = np.nanargmax(catalyst_temp, axis=1)
            max_temps = np.nanmax(catalyst_temp, axis=1)
            
            features['hot_spot_evolution'] = {
                'positions': length_vector[max_temp_positions],
                'temperatures': max_temps,
                'times': time_vector
            }
            
            # Hot spot stability
            position_std = np.std(length_vector[max_temp_positions])
            features['hot_spot_stability'] = {
                'position_std': position_std,
                'is_stable': position_std < 0.1  # Within 10cm
            }
            print(f"  • Hot spot tracking: position std = {position_std:.3f} m")
        
        # Reaction rate optima
        if 'Reaction Rate (kmol/m3/hr)' in core_matrices:
            reaction_rate = core_matrices['Reaction Rate (kmol/m3/hr)']
            
            # Find peak reaction rates
            max_rate_positions = np.nanargmax(reaction_rate, axis=1)
            max_rates = np.nanmax(reaction_rate, axis=1)
            
            features['reaction_rate_optima'] = {
                'positions': length_vector[max_rate_positions],
                'rates': max_rates,
                'times': time_vector
            }
            print(f"  • Reaction rate optima: tracked")
        
        # Phase transitions (rapid changes)
        if 'temporal_temperature_gradient' in package['derived_variables']:
            temp_grad = package['derived_variables']['temporal_temperature_gradient']
            
            # Find rapid changes (above threshold)
            rapid_change_threshold = np.nanstd(temp_grad) * 3
            rapid_changes = np.abs(temp_grad) > rapid_change_threshold
            
            # Get time points and positions of rapid changes
            rapid_indices = np.where(rapid_changes)
            if len(rapid_indices[0]) > 0:
                rapid_times = time_vector[rapid_indices[0]]
                rapid_positions = length_vector[rapid_indices[1]]
                
                features['rapid_transitions'] = {
                    'times': rapid_times,
                    'positions': rapid_positions,
                    'threshold': rapid_change_threshold,
                    'count': len(rapid_times)
                }
                print(f"  • Rapid transitions: {len(rapid_times)} detected")
    
    def _generate_metadata(self, package: Dict[str, Any], file_path: str):
        """Generate comprehensive metadata for the analysis"""
        
        import datetime
        
        metadata = package['metadata']
        
        # File information
        metadata['source_file'] = {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Data dimensions
        time_vector = package['core_vectors']['time_vector']
        length_vector = package['core_vectors']['length_vector']
        
        metadata['dimensions'] = {
            'time_points': len(time_vector),
            'spatial_points': len(length_vector),
            'time_range': (float(time_vector[0]), float(time_vector[-1])),
            'length_range': (float(length_vector[0]), float(length_vector[-1]))
        }
        
        # Variable summary
        metadata['variables'] = {
            'core_variables': list(package['core_matrices'].keys()),
            'derived_variables': list(package['derived_variables'].keys()),
            'scalar_metrics': list(package['scalar_metrics'].keys()),
            'key_features': list(package['key_features'].keys())
        }
        
        # Analysis summary
        ramp_params = package['ramp_parameters']
        metadata['analysis_summary'] = {
            'ramp_detected': bool(ramp_params.duration),
            'ramp_type': f"{ramp_params.duration}min_{ramp_params.direction}_{ramp_params.curve_shape}" if ramp_params.duration else "none",
            'steady_state_detected': package['scalar_metrics'].get('steady_state_time') is not None,
            'total_variables_computed': len(package['core_matrices']) + len(package['derived_variables'])
        }
        
        print(f"  • Analysis metadata: complete")
        print(f"    - Variables: {metadata['analysis_summary']['total_variables_computed']} total")
        print(f"    - Ramp type: {metadata['analysis_summary']['ramp_type']}")
    
    def _save_analysis_package(self, package: Dict[str, Any], file_path: str):
        """Save complete analysis package to disk for reuse"""
        
        if not self.save_intermediate_data:
            return
        
        # Create analysis output directory
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), f"{base_name}_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectors
        vectors_file = os.path.join(output_dir, "vectors.npz")
        np.savez(vectors_file, **package['core_vectors'])
        
        # Save matrices 
        matrices_file = os.path.join(output_dir, "matrices.npz")
        np.savez(matrices_file, **package['core_matrices'])
        
        # Save derived variables
        if package['derived_variables']:
            derived_file = os.path.join(output_dir, "derived_variables.npz")
            np.savez(derived_file, **package['derived_variables'])
        
        # Save scalar metrics as JSON
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, str):
                return str(obj)
            return obj
        
        metrics_file = os.path.join(output_dir, "scalar_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(convert_numpy(package['scalar_metrics']), f, indent=2)
        
        features_file = os.path.join(output_dir, "key_features.json") 
        with open(features_file, 'w') as f:
            json.dump(convert_numpy(package['key_features']), f, indent=2)
        
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(convert_numpy(package['metadata']), f, indent=2)
        
        # Create summary report
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("DYNAMIC REACTOR ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            metadata = package['metadata']
            f.write(f"Source File: {metadata['source_file']['filename']}\n")
            f.write(f"Analysis Time: {metadata['source_file']['timestamp']}\n")
            f.write(f"Data Dimensions: {metadata['dimensions']['time_points']} x {metadata['dimensions']['spatial_points']}\n")
            f.write(f"Time Range: {metadata['dimensions']['time_range'][0]:.1f} - {metadata['dimensions']['time_range'][1]:.1f} min\n")
            f.write(f"Length Range: {metadata['dimensions']['length_range'][0]:.3f} - {metadata['dimensions']['length_range'][1]:.3f} m\n\n")
            
            f.write("Variables Computed:\n")
            for var in metadata['variables']['core_variables']:
                f.write(f"  • {var}\n")
            
            f.write(f"\nDerived Variables: {len(metadata['variables']['derived_variables'])}\n")
            f.write(f"Scalar Metrics: {len(metadata['variables']['scalar_metrics'])}\n")
            f.write(f"Key Features: {len(metadata['variables']['key_features'])}\n")
        
        print(f"  • Analysis package saved: {output_dir}")
        print(f"    - Files: vectors.npz, matrices.npz, derived_variables.npz")
        print(f"    - Metrics: scalar_metrics.json, key_features.json, metadata.json")
        print(f"    - Summary: analysis_summary.txt")


class DynamicRampAnalyzer:
    """
    Main analyzer class that coordinates all analysis components for reactor ramp tests.
    
    This class serves as the primary interface for conducting comprehensive dynamic
    ramp analysis of reactor systems. It coordinates data loading, analysis execution,
    and results management while handling configuration and plotting setup.
    
    The analyzer integrates multiple components:
    - AnalysisEngine for core computational analysis
    - ConfigManager for matplotlib and analysis settings
    - Results management for output organization
    - Error handling and recovery mechanisms
    
    Attributes:
        config: Dictionary containing analysis and plotting configuration
        analysis_engine: AnalysisEngine instance for data processing
        
    Example:
        >>> analyzer = DynamicRampAnalyzer()
        >>> analyzer.run_analysis('reactor_data.csv', options)
        >>> # Results automatically saved with timestamp
        
    Note:
        This class handles missing dependencies gracefully and provides
        default configurations when external modules are unavailable.
    """
    
    def __init__(self):
        # Update matplotlib settings if ConfigManager is available
        results_manager_module = _import_module_safely('results_manager')
        if results_manager_module:
            try:
                ConfigManager = results_manager_module.ConfigManager
                ConfigManager.update_matplotlib_settings()
                self.config = ConfigManager.get_config()
            except AttributeError:
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when ConfigManager is not available"""
        return {
            'steady_state': {
                'threshold': 0.05,
                'min_duration': 5.0
            }
        }
    
    def run_data_processing_only(self, file_path: str, options) -> bool:
        """Run data processing only (no plotting) for threading"""
        try:
            print("="*60)
            print("DYNAMIC REACTOR RAMP ANALYSIS")
            print("="*60)
            
            # Use AnalysisEngine for all data processing to avoid duplication
            print("Initializing analysis engine...")
            engine = AnalysisEngine(config=self.config)
            
            # Run complete analysis - this does everything once and correctly
            analysis_package = engine.run_complete_analysis(
                file_path=file_path, 
                time_limit=getattr(options, 'time_limit', None),
                save_package=True
            )
            
            # Extract key information from analysis package for backward compatibility
            ramp_params = analysis_package['ramp_parameters']
            time_vector = analysis_package['core_vectors']['time_vector']
            length_vector = analysis_package['core_vectors']['length_vector']
            catalyst_temp_matrix = analysis_package['core_matrices']['T_cat (°C)']
            bulk_temp_matrix = analysis_package['core_matrices'].get('T (°C)')
            
            # Extract computed metrics
            steady_state_time = analysis_package['scalar_metrics'].get('steady_state_time')
            stability_metrics = analysis_package['scalar_metrics'].get('stability_metrics')
            
            # Print results summary
            if steady_state_time:
                print(f"✓ Steady state detected at t = {steady_state_time:.1f} min")
            else:
                print("⚠ No steady state detected in analysis period")
            
            # Generate analysis report
            results_manager_module = _import_module_safely('results_manager')
            if results_manager_module:
                try:
                    AnalysisReporter = results_manager_module.AnalysisReporter
                    
                    # Create legacy data package for reporter compatibility
                    legacy_data_package = {
                        'time_vector': time_vector,
                        'length_vector': length_vector,
                        'variables': analysis_package['core_matrices'],
                        'file_path': file_path,
                        'dimensions': analysis_package['metadata']['dimensions'],
                        'format_type': 'StandardDataPackage'
                    }
                    
                    AnalysisReporter.print_analysis_summary(
                        legacy_data_package, ramp_params, steady_state_time, stability_metrics
                    )
                except AttributeError:
                    print("Analysis reporter not available")
            else:
                print("Results manager module not available")
            
            # Update results comparison file
            print("\nUpdating results comparison file...")
            results_manager_module = _import_module_safely('results_manager')
            if results_manager_module:
                try:
                    # Set up engine for metrics extraction
                    engine.data_package = {
                        'time_vector': time_vector,
                        'length_vector': length_vector,
                        'variables': analysis_package['core_matrices'],
                        'file_path': file_path
                    }
                    engine.ramp_params = ramp_params
                    engine.steady_state_time = steady_state_time
                    engine.stability_metrics = stability_metrics
                    
                    metrics = engine.extract_key_metrics()
                    
                    ResultsComparison = results_manager_module.ResultsComparison
                    comparison_file = ResultsComparison.update_comparison_file(metrics)
                    print(f"Results comparison file updated: {os.path.basename(comparison_file)}")
                except Exception as e:
                    print(f"Warning: Could not update results comparison file: {e}")
            else:
                print("Warning: Results manager not available for comparison file update")
            
            # Store processed data for main thread plotting
            self.processed_data = {
                'time_vector': time_vector,
                'catalyst_temp_matrix': catalyst_temp_matrix,
                'bulk_temp_matrix': bulk_temp_matrix,
                'length_vector': length_vector,
                'ramp_params': ramp_params,
                'steady_state_time': steady_state_time,
                'stability_metrics': stability_metrics,
                'analysis_package': analysis_package  # Store complete analysis for advanced users
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Data processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_analysis(self, file_path: str, options) -> bool:
        """
        Run complete analysis workflow with automated plotting.
        
        This method executes the full analysis pipeline including data processing,
        steady state detection, gradient analysis, and automatic plot generation.
        Designed for standalone script usage where GUI is not available.
        
        Args:
            file_path: Absolute path to the CSV data file to analyze
            options: AnalysisOptions object containing analysis configuration
            
        Returns:
            bool: True if analysis and plotting completed successfully, False otherwise
            
        Side Effects:
            - Creates analysis package accessible via self.processed_data
            - Generates and saves plots to timestamped directory
            - Prints progress and results to console
            
        Example:
            >>> engine = AnalysisEngine()
            >>> options = AnalysisOptions(temperature_threshold=0.05)
            >>> success = engine.run_analysis('data/reactor_test.csv', options)
            >>> if success:
            ...     print("Analysis and plots completed successfully")
        """
        success = self.run_data_processing_only(file_path, options)
        if not success:
            return False
        
        # Generate plots if data processing was successful
        plot_generator_module = _import_module_safely('plot_generator')
        if plot_generator_module:
            try:
                PlotGen = plot_generator_module.PlotGenerator
                data = self.processed_data
                
                if options.temperature_response:
                    fig = PlotGen.create_temperature_response_plots(
                        data['time_vector'], data['catalyst_temp_matrix'], data['length_vector'],
                        data['ramp_params'], data['steady_state_time'], file_path, 
                        getattr(options, 'time_limit', None)
                    )
                    if fig:
                        import matplotlib.pyplot as plt
                        plt.show()
                
                # Add other plot types as needed...
                
            except Exception as e:
                print(f"Error generating plots: {e}")
                return False
        else:
            print("Warning: Plot generator module not available")
        
        return True
    
    def save_analysis_results(self, data_package: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save analysis results to files and return timestamp"""
        try:
            from importlib import import_module
            results_manager_module = import_module('results_manager')
            DataExporter = results_manager_module.DataExporter
            timestamp = DataExporter.save_data_structure(data_package, output_dir)
            return timestamp
        except ImportError:
            print("Warning: DataExporter not available - results not saved")
            return "unknown"
