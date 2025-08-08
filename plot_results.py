"""
Dynamic Reactor Ramp Analysis - Plotting Module
==============================================

Automated plotting script for dynamic reactor ramp analysis results.
Generates comprehensive visualizations of analysis outputs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


class PlotGenerator:
    """Handles all plotting operations for reactor analysis results."""
    
    def __init__(self, results_folder: str):
        """Initialize with results folder path."""
        self.results_folder = results_folder
        self.metadata = self._load_metadata()
        self.analysis_data = self._load_analysis_data()
        self.time_vector = self._load_time_vector()
        self.length_vector = self._load_length_vector()
        self.ramp_direction = self._get_ramp_direction()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        path = os.path.join(self.results_folder, 'metadata.json')
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_analysis_data(self) -> Dict[str, Any]:
        """Load analysis data from JSON file."""
        path = os.path.join(self.results_folder, 'analysis_data.json')
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_time_vector(self) -> Optional[np.ndarray]:
        """Load time vector from CSV file."""
        path = os.path.join(self.results_folder, 'time_vector.csv')
        try:
            return np.array(pd.read_csv(path)['time_min'].values)
        except (FileNotFoundError, KeyError):
            return None
    
    def _load_length_vector(self) -> Optional[np.ndarray]:
        """Load length vector from CSV file."""
        path = os.path.join(self.results_folder, 'length_vector.csv')
        try:
            return np.array(pd.read_csv(path)['length_m'].values)
        except (FileNotFoundError, KeyError):
            return None
    
    def _get_ramp_direction(self) -> str:
        """Get ramp direction from analysis data or metadata."""
        return (self.analysis_data.get('ramp_direction') or 
                self.metadata.get('ramp_direction', 'unknown'))
    
    def _save_plot(self, filename: str) -> None:
        """Save current plot with consistent formatting."""
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, filename))
        plt.close()
    
    def plot_extreme_dTcat_dt_vs_position(self) -> None:
        """Plot extreme dTcat/dt vs position for 2D data."""
        dTcat_dt_path = os.path.join(self.results_folder, 'dTcat_dt.csv')
        
        if not (os.path.exists(dTcat_dt_path) and self.length_vector is not None):
            return
        
        dTcat_dt = pd.read_csv(dTcat_dt_path).values
        
        if dTcat_dt.ndim == 2 and len(self.length_vector) == dTcat_dt.shape[1]:
            extreme_func = np.nanmin if self.ramp_direction == 'down' else np.nanmax
            extreme_per_pos = extreme_func(dTcat_dt, axis=0)
            
            plt.figure()
            plt.plot(self.length_vector, extreme_per_pos, marker='o')
            plt.xlabel('Position (m)')
            plt.ylabel('Max |dTcat/dt| (°C/min)')
            plt.title('Extreme dTcat/dt vs. Position (global)')
            plt.grid(True)
            self._save_plot('dTcat_dt_extreme_vs_position.png')
    
    def plot_extreme_dTcat_dt_vs_time(self) -> None:
        """Plot extreme dTcat/dt over time across all positions."""
        dTcat_dt_path = os.path.join(self.results_folder, 'dTcat_dt.csv')
        
        if not os.path.exists(dTcat_dt_path) or self.time_vector is None:
            return
        
        dTcat_dt = pd.read_csv(dTcat_dt_path).values
        
        if dTcat_dt.ndim == 2:
            extreme_func = np.nanmin if self.ramp_direction == 'down' else np.nanmax
            extreme_over_time = extreme_func(dTcat_dt, axis=1)
            
            plt.figure()
            plt.plot(self.time_vector, extreme_over_time)
            plt.xlabel('Time (min)')
            plt.ylabel('Extreme dTcat/dt (°C/min)')
            plt.title('Extreme dTcat/dt vs. Time (global across positions)')
            plt.grid(True)
            self._save_plot('dTcat_dt_extreme_vs_time.png')
    
    def plot_variables(self) -> None:
        """Plot all variables from CSV files."""
        if self.time_vector is None:
            return
            
        csv_files = [f for f in os.listdir(self.results_folder) 
                    if f.endswith('.csv') and f not in ['time_vector.csv', 'length_vector.csv']]
        
        for csv_file in csv_files:
            var_name = csv_file.replace('.csv', '')
            data = pd.read_csv(os.path.join(self.results_folder, csv_file)).values
            
            plt.figure(figsize=(8, 5))
            
            # Check if this is a position-based extreme variable
            if 'dTcat_dt_extreme_' in var_name and self.length_vector is not None:
                # This is position-based data - plot against length vector
                data_flat = data.flatten()
                if len(data_flat) == len(self.length_vector):
                    plt.plot(self.length_vector, data_flat, marker='o')
                    plt.xlabel('Position (m)')
                    plt.ylabel('Extreme dTcat/dt (°C/min)')
                    plt.title(f'{var_name} vs Position')
                    plt.grid(True)
                else:
                    print(f"⚠ Skipping {var_name}: dimension mismatch (position: {len(self.length_vector)}, data: {len(data_flat)})")
                    plt.close()
                    continue
            elif (data.ndim == 2 and data.shape[1] > 1 and self.length_vector is not None):
                # 2D plot (time vs position)
                extent = (float(self.time_vector[0]), float(self.time_vector[-1]),
                         float(self.length_vector[0]), float(self.length_vector[-1]))
                plt.imshow(data.T, aspect='auto', origin='lower', extent=extent)
                plt.colorbar(label=var_name)
                plt.xlabel('Time (min)')
                plt.ylabel('Position (m)')
                plt.title(f'{var_name} (2D)')
            else:
                # 1D plot (time-based) - handle different data shapes
                if data.ndim == 2:
                    # For 2D data without length vector, take first column or average
                    if data.shape[1] == 1:
                        data_to_plot = data[:, 0]
                    else:
                        # Take average across spatial dimension for 1D representation
                        data_to_plot = np.mean(data, axis=1)
                else:
                    data_to_plot = data.flatten()
                
                # Ensure we have matching dimensions
                if len(data_to_plot) != len(self.time_vector):
                    print(f"⚠ Skipping {var_name}: dimension mismatch (time: {len(self.time_vector)}, data: {len(data_to_plot)})")
                    plt.close()
                    continue
                
                plt.plot(self.time_vector, data_to_plot)
                plt.xlabel('Time (min)')
                plt.ylabel(var_name)
                plt.title(f'{var_name} (1D)')
                plt.grid(True)
            
            self._save_plot(f'{var_name}_plot.png')
    
    def plot_summary_metrics(self) -> None:
        """Plot summary bar chart of extreme dTcat/dt for different windows."""
        if 'dTcat_dt_extreme_1min_extreme' not in self.analysis_data:
            return
        
        labels = ['1min', '5min', '10min', 'ramptime', 'stabtime']
        values = [
            self.analysis_data.get(f'dTcat_dt_extreme_{label}_extreme', 0)
            for label in labels
        ]
        
        # Set threshold line based on ramp direction
        threshold = 0.5 if self.ramp_direction == 'up' else -0.58
        
        plt.figure()
        plt.bar(labels, values)
        plt.axhline(threshold, color='red', linestyle='--', 
                   label=f'{threshold} °C/min')
        plt.ylabel('Extreme dTcat/dt')
        plt.title('Extreme dTcat/dt for Different Windows')
        plt.legend()
        self._save_plot('dTcat_dt_extreme_summary.png')
    
    def generate_all_plots(self) -> None:
        """Generate all available plots."""
        self.plot_extreme_dTcat_dt_vs_position()
        self.plot_extreme_dTcat_dt_vs_time()
        self.plot_variables()
        self.plot_summary_metrics()
        print(f'Plots saved in {self.results_folder}')


def plot_results(results_folder: str) -> None:
    """Main plotting function - creates all plots for analysis results."""
    generator = PlotGenerator(results_folder)
    generator.generate_all_plots()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python plot_results.py <results_folder>')
        sys.exit(1)
    plot_results(sys.argv[1])
