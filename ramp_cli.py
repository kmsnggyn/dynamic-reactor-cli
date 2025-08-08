#!/usr/bin/env python3
"""
ramp_cli.py

Command-line tool for Dynamic Reactor Ramp Analysis (no plots).

Usage:
    ./ramp_cli.py path/to/your_data.csv
"""
import argparse
import os
import sys

from data_loader import DataLoaderManager
from analysis_engine_v2 import run_analysis, AnalysisConfig
import pandas as pd
import json
import numpy as np

def main():

    parser = argparse.ArgumentParser(
        description="CLI: Dynamic Reactor Ramp Analysis (vectors + summary only)"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to Aspen Plus Dynamics CSV export (if not provided, select from data folder)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent folder for results (defaults to <csv_basename>_analysis)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="(Optional) Max time [min] to include in analysis"
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if csv_path is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.isdir(data_dir):
            print(f"❌ Data folder not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        # Ask user for ramp direction first
        ramp_direction = None
        while ramp_direction not in ['down', 'up']:
            ramp_direction = input("Select ramp direction to analyze ([down]/up): ").strip().lower()
            if ramp_direction == '':
                ramp_direction = 'down'
            if ramp_direction not in ['down', 'up']:
                print("Please enter 'down' or 'up'.")
        # Filter files by ramp direction in filename (after first hyphen)
        all_csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
        filtered_files = []
        for f in all_csv_files:            
            parts = f.split('-', 1)
            if len(parts) > 1:
                after_hyphen = parts[1].lower()
                if ramp_direction in after_hyphen:
                    filtered_files.append(f)
        if not filtered_files:
            print(f"❌ No CSV files found in {data_dir} for ramp direction '{ramp_direction}'", file=sys.stderr)
            sys.exit(1)
        print(f"Available CSV files in 'data' for ramp direction '{ramp_direction}':")
        for idx, fname in enumerate(filtered_files, 1):
            print(f"  [{idx}] {fname}")
        while True:
            try:
                selection = input(f"Select a file [1-{len(filtered_files)}]: ")
                file_idx = int(selection)
                if 1 <= file_idx <= len(filtered_files):
                    csv_path = os.path.join(data_dir, filtered_files[file_idx - 1])
                    break
                else:
                    print("Invalid selection. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Try again.")

    if not os.path.isfile(csv_path):
        print(f"❌ Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 1) Verify data can load
    loader = DataLoaderManager()
    if loader.load_data(csv_path) is None:
        print("❌ Data loading failed.", file=sys.stderr)
        sys.exit(1)

    # 2) Run full analysis pipeline using the new v2 API
    cfg = AnalysisConfig(time_limit=args.time_limit)
    print("\n▶ Running analysis pipeline …")
    package = run_analysis(csv_path, cfg)

    # 3) Prepare flat dict for saving (adapted to new structure)
    data = package['data']
    metadata = package['metadata']
    flat_pkg = {
        'file_path': metadata.get('source_file', csv_path),
        'time_vector': data.get('time_vector'),
        'length_vector': data.get('length_vector'),
        'variables': data.get('variables'),
        'metadata': metadata,
        'dimensions': {
            'n_time': len(data.get('time_vector')) if data.get('time_vector') is not None else 0,
            'm_length': len(data.get('length_vector')) if data.get('length_vector') is not None else 0
        }
    }

    # 4) Save results (minimal, no DataExporter)
    print("\n▶ Saving results to disk …")
    output_dir = args.output_dir or os.path.splitext(os.path.basename(csv_path))[0] + "_analysis"
    os.makedirs(output_dir, exist_ok=True)


    # Save time_vector
    pd.DataFrame({'time_min': flat_pkg['time_vector']}).to_csv(os.path.join(output_dir, 'time_vector.csv'), index=False)
    # Save length_vector if present
    if flat_pkg['length_vector'] is not None:
        pd.DataFrame({'length_m': flat_pkg['length_vector']}).to_csv(os.path.join(output_dir, 'length_vector.csv'), index=False)
    # Save variables (each as CSV)
    for var, arr in flat_pkg['variables'].items():
        pd.DataFrame(arr).to_csv(os.path.join(output_dir, f'{var}.csv'), index=False)

    # Helper to convert numpy arrays in dicts to lists for JSON
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return obj

    # Save all data (except variables, time_vector, length_vector, dimensions) as a single JSON
    data_to_save = {k: v for k, v in package['data'].items() if k not in ['variables', 'time_vector', 'length_vector']}
    with open(os.path.join(output_dir, 'analysis_data.json'), 'w') as f:
        json.dump(to_serializable(data_to_save), f, indent=2)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(to_serializable(package['metadata']), f, indent=2)
    with open(os.path.join(output_dir, 'dimensions.json'), 'w') as f:
        json.dump(flat_pkg['dimensions'], f, indent=2)

    # 5) Final message
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = output_dir

    # 6) Automatically run plotting script on the output folder
    import subprocess
    plot_script = os.path.join(os.path.dirname(__file__), 'plot_results.py')
    print(f"\n▶ Running plotting script on results folder: {os.path.abspath(out_dir)}")
    try:
        subprocess.run(['python', plot_script, out_dir], check=True)
        
    except Exception as e:
        print(f"⚠ Plotting script failed: {e}")

    print(f"\nALL RESULTS SAVED IN FOLDER:\n   {os.path.abspath(out_dir)}\n")

if __name__ == "__main__":
    main()

