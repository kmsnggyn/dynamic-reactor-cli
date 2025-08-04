"""
This module defines a plot factory for generating various plots based on analysis data.
It uses a registration pattern to make it easy to add new plots.
"""

from plot_generator import PlotGenerator

# Plot Registry
# Maps a plot's ID to its metadata and generation function.
# To add a new plot:
# 1. Add the generation function to the PlotGenerator class.
# 2. Add a new entry to this dictionary with a unique ID.
#    - 'label': The text that will appear in the GUI checkbox.
#    - 'function': A reference to the generation function in PlotGenerator.
#    - 'required_data': A list of keys that must be present in the processed_data 
#                       dictionary for this plot to be generated. This prevents errors
#                       if the necessary data is missing.

PLOT_REGISTRY = {
    "temperature_response": {
        "label": "Temperature Response",
        "function": PlotGenerator.create_temperature_response_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params', 'steady_state_time']
    },
    "stability_analysis": {
        "label": "Stability Analysis",
        "function": PlotGenerator.create_stability_analysis_plots,
        "required_data": ['time_vector', 'stability_metrics', 'ramp_params', 'steady_state_time']
    },
    "spatial_gradients": {
        "label": "Spatial Gradients",
        "function": PlotGenerator.create_spatial_gradient_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params']
    },
    "heat_transfer_3d": {
        "label": "3D Heat Transfer",
        "function": PlotGenerator.create_3d_heat_transfer_plots,
        "required_data": ['time_vector', 'heat_transfer_matrix', 'length_vector', 'ramp_params', 'steady_state_time']
    },
    "temperature_difference": {
        "label": "Temperature Difference",
        "function": PlotGenerator.create_temperature_difference_plots,
        "required_data": ['time_vector', 'catalyst_temp_matrix', 'length_vector', 'ramp_params']
    },
}

def generate_plots(selected_plot_ids, processed_data, file_path, time_limit, add_terminal_output):
    """
    Generates all plots selected by the user.

    Args:
        selected_plot_ids (list): A list of strings corresponding to the keys in PLOT_REGISTRY.
        processed_data (dict): The dictionary containing all processed data from the analysis.
        file_path (str): The path to the source data file.
        time_limit (float or None): The time limit for the plots.
        add_terminal_output (function): A function to log messages to the GUI terminal.

    Returns:
        list: A list of tuples, where each tuple is (plot_label, plot_figure).
    """
    generated_figs = []
    for plot_id in selected_plot_ids:
        if plot_id in PLOT_REGISTRY:
            plot_info = PLOT_REGISTRY[plot_id]
            label = plot_info['label']
            plot_function = plot_info['function']
            required_data = plot_info['required_data']

            # Check if all required data is available
            if not all(key in processed_data for key in required_data):
                add_terminal_output(f"   Skipping '{label}': Missing required data.")
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
                add_terminal_output(f"   {label} plots completed")
            except Exception as e:
                add_terminal_output(f"   Error generating {label} plots: {e}")
    
    return generated_figs
