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
        help="Path to Aspen Plus Dynamics CSV export"
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

    # 3) Prepare flat dict for saving
    flat_pkg = {
        'file_path': package.metadata.get('source_file', csv_path),
        'time_vector': package.time_vector,
        'length_vector': package.length_vector,
        'variables': package.variables,
        'metadata': package.metadata,
        'dimensions': {
            'n_time': len(package.time_vector),
            'm_length': len(package.length_vector) if package.length_vector is not None else 0
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

    with open(os.path.join(output_dir, 'derived.json'), 'w') as f:
        json.dump(to_serializable(package.derived), f, indent=2)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(to_serializable(package.metrics), f, indent=2)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(to_serializable(package.metadata), f, indent=2)
    with open(os.path.join(output_dir, 'dimensions.json'), 'w') as f:
        json.dump(flat_pkg['dimensions'], f, indent=2)

    # 5) Final message
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = output_dir
    print(f"\n✅ All results saved in folder:\n   {os.path.abspath(out_dir)}\n")

if __name__ == "__main__":
    main()
