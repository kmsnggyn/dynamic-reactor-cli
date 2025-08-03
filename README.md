# Dynamic Reactor Ramp Analysis Tool

A GUI tool for analyzing Aspen Plus dynamic reactor simulation data with automated ramp detection and comprehensive visualization.

**Author:** Seonggyun Kim (seonggyun.kim@outlook.com) | **License:** MIT | **Python:** 3.7+

## Quick Start

```bash
# Install dependencies
pip install matplotlib pandas numpy seaborn scipy

# Test installation
python test_setup.py

# Launch application
python main_gui.py
```

## Usage Workflow

### 1. Prepare Data in Aspen Plus
1. Create **Flowsheet Form** with:
   - **X-axis**: `L_Profile(*)` (reactor length)
   - **Y-axis**: `T_cat(*)~1`, `T(*)~1`, `Q_flux(*)~1`, `Q_cat(*)~1`, reaction rates
2. Run dynamic simulation
3. Copy/paste table data to CSV file
4. Name file: `{duration}-{direction}-{curve}.csv` (e.g., `30-up-r.csv`)

### 2. Run Analysis
1. Launch GUI: `python main_gui.py`
2. Select CSV file
3. Choose plot types and options
4. Click "Run Analysis"

## Features

- **4 Analysis Types**: Temperature response, stability, spatial gradients, 3D heat transfer
- **Auto Ramp Detection**: From filename (`30-up-r.csv` = 30-min linear ramp-up)
- **Export Options**: CSV matrices, vectors, summary reports
- **Modular Architecture**: Separate GUI, analysis, plotting, and results modules

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

## Troubleshooting

**Aspen Plus:**
- Include all required variables in flowsheet form
- Ensure `L_Profile(*)` is set as reactor length coordinate
- Complete dynamic simulation before export

**Application:**
- Check console output for detailed errors
- Verify CSV format matches Aspen Plus export
- Select at least one plot type

---

**Contact:** Seonggyun Kim (seonggyun.kim@outlook.com) | KTH Royal Institute of Technology
