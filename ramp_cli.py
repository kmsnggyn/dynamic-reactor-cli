#!/usr/bin/env python3
"""
Dynamic Reactor Ramp Analysis CLI
================================

Command-line tool for comprehensive dynamic reactor ramp analysis.
Features automatic file selection, analysis execution, and visualization.

Usage:
    python ramp_cli.py [csv_file] [options]
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from data_loader import DataLoaderManager
from analysis_engine_v2 import run_analysis, AnalysisConfig


class DataFileSelector:
    """Handles file selection from data directory."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def get_ramp_direction(self) -> str:
        """Get user selection for ramp direction."""
        while True:
            ramp_direction = input("Select ramp direction to analyze ([down]/up): ").strip().lower()
            if ramp_direction == '':
                return 'down'
            if ramp_direction in ['down', 'up']:
                return ramp_direction
            print("Please enter 'down' or 'up'.")
    
    def filter_files_by_direction(self, ramp_direction: str) -> List[str]:
        """Filter CSV files by ramp direction."""
        all_csv_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.csv')]
        filtered_files = []
        
        for filename in all_csv_files:
            if '-' in filename:
                after_hyphen = filename.split('-', 1)[1].lower()
                if ramp_direction in after_hyphen:
                    filtered_files.append(filename)
        
        return filtered_files
    
    def select_file(self) -> str:
        """Interactive file selection process."""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")
        
        ramp_direction = self.get_ramp_direction()
        filtered_files = self.filter_files_by_direction(ramp_direction)
        
        if not filtered_files:
            raise FileNotFoundError(f"No CSV files found for ramp direction '{ramp_direction}'")
        
        print(f"\nAvailable CSV files for ramp direction '{ramp_direction}':")
        for idx, filename in enumerate(filtered_files, 1):
            print(f"  [{idx}] {filename}")
        
        while True:
            try:
                selection = input(f"Select a file [1-{len(filtered_files)}]: ")
                file_idx = int(selection)
                if 1 <= file_idx <= len(filtered_files):
                    return os.path.join(self.data_dir, filtered_files[file_idx - 1])
                else:
                    print("Invalid selection. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Try again.")


class ResultsSaver:
    """Handles saving of analysis results to disk."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save all analysis results to appropriate files."""
        # Save basic vectors
        self._save_time_vector(results.get('time_vector'))
        self._save_length_vector(results.get('length_vector'))
        
        # Save variables
        if 'variables' in results:
            self._save_variables(results['variables'])
        
        # Save derived arrays
        self._save_derived_arrays(results)
        
        # Save JSON data
        self._save_json_data(results)
    
    def _save_time_vector(self, time_vector: Optional[np.ndarray]) -> None:
        """Save time vector to CSV."""
        if time_vector is not None:
            df = pd.DataFrame({'time_min': time_vector})
            df.to_csv(os.path.join(self.output_dir, 'time_vector.csv'), index=False)
    
    def _save_length_vector(self, length_vector: Optional[np.ndarray]) -> None:
        """Save length vector to CSV."""
        if length_vector is not None:
            df = pd.DataFrame({'length_m': length_vector})
            df.to_csv(os.path.join(self.output_dir, 'length_vector.csv'), index=False)
    
    def _save_variables(self, variables: Dict[str, np.ndarray]) -> None:
        """Save variable arrays to CSV files."""
        for var_name, data in variables.items():
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.output_dir, f'{var_name}.csv'), index=False)
    
    def _save_derived_arrays(self, results: Dict[str, Any]) -> None:
        """Save derived arrays (like dTcat_dt) to CSV files."""
        skip_keys = {'variables', 'time_vector', 'length_vector', 'metadata'}
        
        for key, value in results.items():
            if key not in skip_keys and isinstance(value, np.ndarray):
                df = pd.DataFrame(value)
                df.to_csv(os.path.join(self.output_dir, f'{key}.csv'), index=False)
    
    def _save_json_data(self, results: Dict[str, Any]) -> None:
        """Save analysis data and metadata to JSON files."""
        # Analysis data (excluding arrays)
        analysis_data = {
            k: v for k, v in results.items() 
            if k not in {'variables', 'time_vector', 'length_vector', 'metadata'} 
            and not isinstance(v, np.ndarray)
        }
        
        with open(os.path.join(self.output_dir, 'analysis_data.json'), 'w') as f:
            json.dump(self._make_serializable(analysis_data), f, indent=2)
        
        # Metadata
        if 'metadata' in results:
            with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
                json.dump(self._make_serializable(results['metadata']), f, indent=2)


def run_plotting_script(output_dir: str) -> None:
    """Execute the plotting script on the results folder."""
    plot_script = os.path.join(os.path.dirname(__file__), 'plot_results.py')
    if not os.path.exists(plot_script):
        print("⚠ Plotting script not found")
        return
    
    print(f"\n▶ Generating plots for results in: {os.path.abspath(output_dir)}")
    try:
        subprocess.run(['python', plot_script, output_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠ Plotting script failed: {e}")
    except Exception as e:
        print(f"⚠ Error running plotting script: {e}")


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Dynamic Reactor Ramp Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to CSV file (if not provided, select from data folder)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: <csv_basename>_analysis)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Maximum time [min] to include in analysis"
    )
    
    args = parser.parse_args()
    
    # Determine input file
    if args.csv_file:
        csv_path = args.csv_file
        if not os.path.isfile(csv_path):
            print(f"❌ Error: File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive file selection
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        try:
            selector = DataFileSelector(data_dir)
            csv_path = selector.select_file()
        except FileNotFoundError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
    
    # Verify data can be loaded
    print(f"\n▶ Loading data from: {os.path.basename(csv_path)}")
    loader = DataLoaderManager()
    if loader.load_data(csv_path) is None:
        print("❌ Data loading failed.", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    print("▶ Running analysis pipeline…")
    cfg = AnalysisConfig(time_limit=args.time_limit)
    try:
        results = run_analysis(csv_path, cfg)
    except Exception as e:
        print(f"❌ Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save results
    output_dir = args.output_dir or (Path(csv_path).stem + "_analysis")
    print(f"▶ Saving results to: {output_dir}")
    
    saver = ResultsSaver(output_dir)
    saver.save_results(results)
    
    # Generate plots
    run_plotting_script(output_dir)
    
    # Final message
    print(f"\n✅ Analysis complete! Results saved in:\n   {os.path.abspath(output_dir)}\n")


if __name__ == "__main__":
    main()

