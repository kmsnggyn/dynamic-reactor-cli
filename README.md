
# Dynamic Reactor Ramp Analysis Tool (CLI-Only)

A minimal, scriptable command-line tool for analyzing Aspen Plus dynamic reactor simulation data. This version is headless (no GUI, no plotting) and designed for batch or automated workflows.

**Author:** Seonggyun Kim (seonggyun.kim@outlook.com)  
**License:** MIT  
**Python:** 3.7+

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis on your data
python ramp_cli.py path/to/your_data.csv
```

---

## Usage

### 1. Prepare Data in Aspen Plus
1. Create a **Flowsheet Form** with:
    - **X-axis**: `L_Profile(*)` (reactor length)
    - **Y-axis**: `T_cat(*)~1`, `T(*)~1`, `Q_flux(*)~1`, `Q_cat(*)~1`, reaction rates
2. Run dynamic simulation
3. Export or copy/paste table data to a CSV file
4. Name the file: `{duration}-{direction}-{curve}.csv` (e.g., `30-up-r.csv`)

### 2. Run CLI Analysis
```bash
python ramp_cli.py path/to/your_data.csv
# Optional arguments:
#   --output-dir <folder>   # Output directory (default: <csv_basename>_analysis)
#   --time-limit <minutes>  # Max time to include in analysis
```

The tool will load the CSV, run the analysis, and save results (vectors, summary) to disk.

---

## File Requirements

**Expected CSV format from Aspen Plus:**
- Row 0: "Time" header
- Row 2: Time + reactor positions (`L_Profile(*)`)
- Rows 3-7: Variables (`T_cat`, `T`, `Q_cat`, `Q_flux`, reaction rates)
- Pattern repeats for each time point

**Naming convention:** `{duration}-{direction}-{curve}.csv`
- Duration: minutes (e.g., `30`, `45`)
- Direction: `up` or `down`
- Curve: `r` (linear) or `s` (sinusoidal)

---

## Troubleshooting

- Check console output for detailed errors
- Verify CSV format matches Aspen Plus export
- Ensure Python and dependencies are installed

---

**Contact:** Seonggyun Kim (seonggyun.kim@outlook.com) | KTH Royal Institute of Technology
