import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This script expects to be run in a folder containing the output files from ramp_cli.py
# Usage: python plot_results.py <results_folder>

def plot_results(results_folder):
    # Load metadata and analysis data
    metadata_path = os.path.join(results_folder, 'metadata.json')
    analysis_data_path = os.path.join(results_folder, 'analysis_data.json')
    time_vector_path = os.path.join(results_folder, 'time_vector.csv')
    length_vector_path = os.path.join(results_folder, 'length_vector.csv')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(analysis_data_path, 'r') as f:
        analysis_data = json.load(f)
    time_vector = pd.read_csv(time_vector_path)['time_min'].values
    length_vector = None
    if os.path.exists(length_vector_path):
        length_vector = pd.read_csv(length_vector_path)['length_m'].values

    # Plot all variables
    variables_dir = results_folder
    variable_files = [f for f in os.listdir(variables_dir) if f.endswith('.csv') and f not in ['time_vector.csv', 'length_vector.csv']]
    for var_file in variable_files:
        var_name = var_file.replace('.csv', '')
        arr = pd.read_csv(os.path.join(variables_dir, var_file)).values
        plt.figure(figsize=(8, 5))
        if arr.ndim == 2 and arr.shape[1] > 1 and length_vector is not None:
            extent = (float(time_vector[0]), float(time_vector[-1]), float(length_vector[0]), float(length_vector[-1]))
            plt.imshow(arr.T, aspect='auto', origin='lower', extent=extent)
            plt.colorbar(label=var_name)
            plt.xlabel('Time (min)')
            plt.ylabel('Position (m)')
            plt.title(f'{var_name} (2D)')
        else:
            arr_flat = arr.flatten()
            # Ensure both arrays are numpy arrays for matplotlib
            arr_flat = arr_flat if isinstance(arr_flat, (np.ndarray, list)) else np.array(arr_flat)
            tvec = np.array(time_vector)
            plt.plot(tvec, arr_flat)
            plt.xlabel('Time (min)')
            plt.ylabel(var_name)
            plt.title(f'{var_name} (1D)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{var_name}_plot.png'))
        plt.close()

    # Plot summary metrics if available
    if 'dTcat_dt_extreme_1min_extreme' in analysis_data:
        labels = ['1min', '5min', '10min', 'ramptime', 'stabtime']
        values = [
            analysis_data.get('dTcat_dt_extreme_1min_extreme', 0),
            analysis_data.get('dTcat_dt_extreme_5min_extreme', 0),
            analysis_data.get('dTcat_dt_extreme_10min_extreme', 0),
            analysis_data.get('dTcat_dt_extreme_ramptime_extreme', 0),
            analysis_data.get('dTcat_dt_extreme_stabtime_extreme', 0)
        ]
        # Determine ramp direction
        ramp_direction = analysis_data.get('ramp_direction', None)
        if ramp_direction is None:
            # Try to load from metadata if not present
            metadata_path = os.path.join(results_folder, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                ramp_direction = metadata.get('ramp_direction', None)
        # Set line value
        if ramp_direction == 'up':
            line_value = 0.5
        else:
            line_value = -0.58
        plt.figure()
        plt.bar(labels, values)
        plt.axhline(line_value, color='red', linestyle='--', label=f'{line_value} °C/min')
        plt.ylabel('Extreme dTcat/dt')
        plt.title('Extreme dTcat/dt for Different Windows')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'dTcat_dt_extreme_summary.png'))
        plt.close()

    print(f'Plots saved in {results_folder}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python plot_results.py <results_folder>')
        sys.exit(1)
    plot_results(sys.argv[1])
