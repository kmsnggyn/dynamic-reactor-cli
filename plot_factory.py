"""
Plot Factory Module for Dynamic Reactor Analysis
==============================================

This module defines a plot factory for generating various plots based on analysis data.
It uses a registration pattern to make it easy to add new plots. The factory pattern
provides a clean separation of concerns and makes the system highly extensible.

Features:
- Registry-based plot generation system
- Automatic dependency checking for required data
- Consistent error handling and logging
- Easy addition of new plot types

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

from typing import List, Dict, Any, Tuple, Callable, Optional
import matplotlib.figure
from plot_generator import PlotGenerator

# Plot Factory Configuration Constants
PLOT_REGISTRY_VERSION = "1.0"  # Version for tracking registry changes
DEFAULT_ERROR_MESSAGE_PREFIX = "Error generating"  # Prefix for error messages
DEFAULT_SUCCESS_MESSAGE_SUFFIX = "plots completed"  # Suffix for success messages
DEFAULT_SKIP_MESSAGE_PREFIX = "Skipping"  # Prefix for skip messages
REQUIRED_DATA_CHECK_ENABLED = True  # Enable/disable data validation

# Plot Registry
# =============
# This registry maps plot IDs to their metadata and generation functions.
# The registry pattern allows for easy extension and modification of available plots.
#
# Registry Structure:
# - 'label': Human-readable name displayed in GUI checkboxes
# - 'function': Reference to the PlotGenerator method that creates the plot
# - 'required_data': List of data keys that must be present for plot generation
# - 'description': Brief description of what the plot shows (optional)
#
# Adding New Plots:
# 1. Implement the generation function in PlotGenerator class
# 2. Add a new entry to this registry with a unique ID
# 3. Specify all required data dependencies
# 4. Update any GUI components that reference plot types
#
# Data Validation:
# The factory automatically checks for required data before attempting
# plot generation, preventing runtime errors and providing clear feedback.

PLOT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "temperature_response": {
        "label": "Temperature Response",
        "function": PlotGenerator.create_temperature_response_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params', 'steady_state_time'],
        "description": "Catalyst temperature vs time with change rate analysis"
    },
    "stability_analysis": {
        "label": "Stability Analysis", 
        "function": PlotGenerator.create_stability_analysis_plots,
        "required_data": ['time_vector', 'stability_metrics', 'ramp_params', 'steady_state_time'],
        "description": "System stability metrics and steady state detection"
    },
    "spatial_gradients": {
        "label": "Spatial Gradients",
        "function": PlotGenerator.create_spatial_gradient_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params'],
        "description": "Temperature gradients and hot spot migration analysis"
    },
    "heat_transfer_3d": {
        "label": "3D Heat Transfer",
        "function": PlotGenerator.create_3d_heat_transfer_plots,
        "required_data": ['time_vector', 'heat_transfer_matrix', 'length_vector', 'ramp_params', 'steady_state_time'],
        "description": "3D visualization of heat transfer with coolant"
    },
    "temperature_difference": {
        "label": "Temperature Difference",
        "function": PlotGenerator.create_temperature_difference_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params'],
        "description": "Analysis of temperature differences between catalyst and bulk phases"
    }
}

def validate_plot_registry() -> bool:
    """
    Validate the plot registry structure for consistency and completeness.
    
    Returns:
        bool: True if registry is valid, False otherwise
    """
    required_keys = {'label', 'function', 'required_data'}
    
    for plot_id, plot_info in PLOT_REGISTRY.items():
        if not all(key in plot_info for key in required_keys):
            print(f"Warning: Plot '{plot_id}' missing required registry keys")
            return False
        
        if not callable(plot_info['function']):
            print(f"Warning: Plot '{plot_id}' function is not callable")
            return False
            
        if not isinstance(plot_info['required_data'], list):
            print(f"Warning: Plot '{plot_id}' required_data must be a list")
            return False
    
    return True

def check_data_availability(required_data: List[str], processed_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if all required data is available in processed_data.
    
    Args:
        required_data: List of required data keys
        processed_data: Dictionary containing processed analysis data
        
    Returns:
        Tuple of (all_available: bool, missing_keys: List[str])
    """
    missing_keys = [key for key in required_data if key not in processed_data]
    return len(missing_keys) == 0, missing_keys

def generate_plots(selected_plot_ids: List[str], 
                  processed_data: Dict[str, Any], 
                  file_path: str, 
                  time_limit: Optional[float], 
                  add_terminal_output: Callable[[str], None]) -> List[Tuple[str, matplotlib.figure.Figure]]:
    """
    Generate all plots selected by the user using the registry pattern.
    
    This function serves as the main entry point for plot generation. It validates
    data requirements, handles errors gracefully, and provides comprehensive logging
    for debugging and user feedback.
    
    Args:
        selected_plot_ids: List of plot IDs corresponding to PLOT_REGISTRY keys
        processed_data: Dictionary containing all processed analysis data
        file_path: Path to the source data file (used for plot titles and metadata)
        time_limit: Optional time limit for plot data (None for no limit)
        add_terminal_output: Callback function for logging messages to GUI terminal
        
    Returns:
        List of tuples containing (plot_label, matplotlib_figure) for successful plots
        
    Raises:
        KeyError: If a plot_id is not found in the registry (handled gracefully)
        Exception: Plot generation errors (caught and logged)
        
    Example:
        >>> plot_ids = ['temperature_response', 'stability_analysis']
        >>> figures = generate_plots(plot_ids, data, 'test.csv', 60.0, print)
        >>> print(f"Generated {len(figures)} plots successfully")
        
    Note:
        The function automatically skips plots when required data is missing,
        providing clear feedback about why specific plots were not generated.
    """
    generated_figs = []
    
    # Validate registry at runtime (optional check)
    if REQUIRED_DATA_CHECK_ENABLED and not validate_plot_registry():
        add_terminal_output("Warning: Plot registry validation failed")
    
    for plot_id in selected_plot_ids:
        if plot_id not in PLOT_REGISTRY:
            add_terminal_output(f"{DEFAULT_SKIP_MESSAGE_PREFIX} '{plot_id}': Unknown plot type")
            continue
            
        plot_info = PLOT_REGISTRY[plot_id]
        label = plot_info['label']
        plot_function = plot_info['function']
        required_data = plot_info['required_data']

        # Check if all required data is available
        data_available, missing_keys = check_data_availability(required_data, processed_data)
        if not data_available:
            missing_str = ", ".join(missing_keys)
            add_terminal_output(f"{DEFAULT_SKIP_MESSAGE_PREFIX} '{label}': Missing required data: {missing_str}")
            continue

        add_terminal_output(f"Generating {label} plots...")
        try:
                # Note: This assumes all plot functions in PlotGenerator will be refactored
                # to accept a consistent set of arguments. For now, we'll pass what we have.
                # A better refactoring would make them all accept (processed_data, file_path, time_limit)
                
                # This is a temporary adaptation to the existing function signatures.
                # A full refactor of PlotGenerator would be the next step.
                if plot_id == 'temperature_response':
                    fig = plot_function(
                        processed_data['time_vector'], processed_data['catalyst_temp_matrix'], processed_data['length_vector'],
                        processed_data['ramp_params'], processed_data['steady_state_time'], file_path, time_limit
                    )
                elif plot_id == 'stability_analysis':
                    fig = plot_function(
                        processed_data['time_vector'], processed_data['stability_metrics'], processed_data['ramp_params'],
                        processed_data['steady_state_time'], file_path, time_limit
                    )
                elif plot_id == 'spatial_gradients':
                     fig = plot_function(
                        processed_data['time_vector'], processed_data['catalyst_temp_matrix'], processed_data['length_vector'],
                        processed_data['ramp_params'], file_path, time_limit
                    )
                elif plot_id == 'heat_transfer_3d':
                    fig = plot_function(
                        processed_data['time_vector'], processed_data['heat_transfer_matrix'], processed_data['length_vector'],
                        processed_data['ramp_params'], processed_data['steady_state_time'], file_path, time_limit
                    )
                elif plot_id == 'temperature_difference':
                    fig = plot_function(
                        processed_data['time_vector'], processed_data['catalyst_temp_matrix'], 
                        processed_data.get('bulk_temp_matrix'), processed_data['length_vector'],
                        processed_data['ramp_params'], None, file_path, time_limit
                    )
                else:
                    # Fallback for any other plots, assuming a standard signature
                    fig = plot_function(processed_data, file_path, time_limit)

                generated_figs.append((label, fig))
                add_terminal_output(f"   {label} {DEFAULT_SUCCESS_MESSAGE_SUFFIX}")
        except Exception as e:
            add_terminal_output(f"   {DEFAULT_ERROR_MESSAGE_PREFIX} {label} plots: {e}")
    
    return generated_figs

def get_available_plots() -> Dict[str, str]:
    """
    Get a dictionary of available plot IDs and their labels.
    
    Returns:
        Dictionary mapping plot_id -> label for GUI display
    """
    return {plot_id: info['label'] for plot_id, info in PLOT_REGISTRY.items()}

def get_plot_description(plot_id: str) -> Optional[str]:
    """
    Get the description for a specific plot type.
    
    Args:
        plot_id: The plot identifier
        
    Returns:
        Plot description string or None if not found
    """
    plot_info = PLOT_REGISTRY.get(plot_id)
    return plot_info.get('description') if plot_info else None

def get_plot_requirements(plot_id: str) -> Optional[List[str]]:
    """
    Get the data requirements for a specific plot type.
    
    Args:
        plot_id: The plot identifier
        
    Returns:
        List of required data keys or None if plot not found
    """
    plot_info = PLOT_REGISTRY.get(plot_id)
    return plot_info.get('required_data') if plot_info else None
