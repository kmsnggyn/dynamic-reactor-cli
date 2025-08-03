# Data Directory

Place your Aspen CSV files in this directory for analysis.

## Supported File Formats

The tool supports CSV files exported from Aspen Plus with the following structure:
- Row 1: Headers (column names)
- Row 2: Units (or empty/NaN for timestamp columns)
- Row 3+: Data values

## Example File Naming Convention

For best results, use descriptive filenames that include:
- Duration (e.g., `20min`, `30min`)
- Direction (e.g., `up`, `down`) 
- Curve type (e.g., `r` for linear, `s` for sinusoidal)
- Timestamp (e.g., `20250801-120000`)

Example: `36min-down-s-20250803-233838.csv`

## File Organization

```
data/
├── README.md                           # This file
├── 20min-down-r-20250801-120000.csv  # Example ramp data
├── 30min-up-s-20250801-130000.csv    # Example ramp data
└── results/                           # Analysis outputs (auto-generated)
    ├── plots/                         # Generated plots
    └── Ramp_Analysis_Results_Comparison.csv  # Comparison table
```

> **Note**: All CSV files and generated results are automatically excluded from version control via `.gitignore`.
