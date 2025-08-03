# Dynamic Reactor Ramp Analysis Tool

## Overview
The Dynamic Reactor Ramp Analysis Tool is a comprehensive, user-friendly application for analyzing Aspen Plus dynamic simulation data. It features a modern GUI interface with customizable plot generation and modular code architecture.

**Author:** Seonggyun Kim (seonggyun.kim@outlook.com)  
**License:** MIT  
**Python Version:** 3.7+

## Architecture Overview

The tool is built with a modular architecture for maintainability and extensibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ANALYSIS WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │  main_gui   │───▶│  analysis_engine │───▶│  plot_generator │ │
│  │             │    │                  │    │                 │ │
│  │ • GUI       │    │ • Data loading   │    │ • Temperature   │ │
│  │ • Controls  │    │ • Ramp detection │    │ • Stability     │ │
│  │ • Progress  │    │ • Steady state   │    │ • Spatial       │ │
│  │ • Display   │    │ • Calculations   │    │ • 3D plots      │ │
│  └─────────────┘    └──────────────────┘    └─────────────────┘ │
│         │                                             │          │
│         │            ┌──────────────────┐            │          │
│         └───────────▶│ results_manager  │◀───────────┘          │
│                      │                  │                       │
│                      │ • CSV export     │                       │
│                      │ • Comparison     │                       │
│                      │ • File management│                       │
│                      │ • Configuration  │                       │
│                      └──────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main_gui.py` | **Main Application** | GUI interface, user controls, progress tracking |
| `analysis_engine.py` | **Core Analysis** | Data loading, ramp detection, steady state analysis |
| `plot_generator.py` | **Visualization** | Temperature plots, stability charts, 3D visualization |
| `results_manager.py` | **Data Management** | Export results, comparison tables, configuration |
| `test_setup.py` | **Testing** | Verify installation and module compatibility |

## Installation

### Prerequisites
- Python 3.7 or higher
- Required packages: `tkinter`, `matplotlib`, `pandas`, `numpy`, `seaborn`, `scipy`

### Setup
1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install matplotlib pandas numpy seaborn scipy
   ```
3. Test the installation:
   ```bash
   python test_setup.py
   ```
4. Launch the application:
   ```bash
   python main_gui.py
   ```

## Key Features

### **User-Friendly GUI**
- **File Selection**: Easy-to-use file browser for selecting Aspen CSV files
- **Plot Selection**: Choose which analysis plots to generate (all selected by default)
- **Time Limiting**: Option to limit analysis to specific time ranges
- **Auto-Save**: Option to automatically save analysis results
- **Progress Tracking**: Real-time progress updates and status messages

### **Analysis Capabilities**
1. **Temperature Response Analysis**
   - Catalyst temperature change rates during ramp periods
   - Maximum response tracking with position information
   - Ramp period visualization with timing analysis

2. **Stability Analysis**
   - Steady state detection with customizable thresholds
   - RMS and maximum change rate tracking
   - Stability timeline visualization
   - Comprehensive isolated point filtering

3. **Spatial Temperature Gradients**
   - Hot/cold spot tracking and migration analysis
   - Spatial gradient heatmaps (dT/dx)
   - Zone-based gradient statistics
   - Temperature extrema tracking over time

4. **3D Heat Transfer Analysis**
   - 3D surface plots of heat transfer with coolant
   - Temporal and spatial visualization
   - Ramp period and steady state indicators

### **Modular Architecture**
The code is now organized into clean, reusable modules:

- **`DataLoader`**: Handles CSV parsing and ramp parameter detection
- **`SteadyStateDetector`**: Advanced steady state detection algorithms
- **`PlotGenerator`**: All plotting functionality with consistent styling
- **`AnalysisReporter`**: Comprehensive analysis summaries and statistics
- **`DataExporter`**: Save results and data structures to files
- **`ConfigManager`**: Centralized configuration management
- **`AnalysisGUI`**: Modern tkinter interface with threading support
- **`DynamicRampAnalyzer`**: Main coordinator class

## How to Use

### 1. **Launch the Application**
```bash
python main_gui.py
```

### 2. **Select Your Data File**
- Click "Select Aspen CSV File"
- Choose your Aspen Plus dynamic simulation CSV export
- The filename will be displayed once selected

### 3. **Configure Analysis Options**
- **Time Limit**: Enter a maximum time (in minutes) to analyze, or leave empty for full range
- **Save Results**: Check to automatically save analysis results to files
- **Plot Selection**: Choose which plot types to generate:
  - **Temperature Response Analysis** (recommended)
  - **Stability Analysis** (recommended) 
  - **Spatial Temperature Gradients** (comprehensive analysis)
  - **3D Heat Transfer Analysis** (if heat transfer data available)

### 4. **Run Analysis**
- Click "Run Analysis" to start
- Progress bar and status updates will show analysis progress
- Multiple plot windows will open when complete

### 5. **Save Results** (Optional)
- If auto-save was enabled, results are saved automatically
- Use "Save Last Results" button to save results from previous analysis
- Saved files include:
  - Time and position vectors (CSV)
  - All variable matrices (CSV)
  - Analysis summary report (TXT)

## File Format Requirements

The tool expects Aspen Plus CSV exports with the following structure:
- **Row 0**: "Time" header
- **Row 2**: Time value + position headers
- **Rows 3-7**: 5 variable rows (T_cat, T, Reaction Rate, Heat Transfer to Catalyst, Heat Transfer with coolant)
- **Pattern repeats** every 6 rows for each time point

## Automatic Ramp Detection

The tool automatically detects ramp parameters from filenames using the convention:
```
{duration}-{direction}-{curve_shape}.csv
```

Examples:
- `30-up-r.csv` - 30-minute linear ramp-up
- `45-down-s.csv` - 45-minute sinusoidal ramp-down

Supported parameters:
- **Duration**: Any integer (minutes)
- **Direction**: `up` or `down`
- **Curve Shape**: `r` (linear) or `s` (sinusoidal)

## Output Files (when saving enabled)

Results are saved to a folder named `{filename}_ramp_analysis/` containing:

1. **Vector Files**:
   - `{filename}_time_vector_{timestamp}.csv`
   - `{filename}_length_vector_{timestamp}.csv`

2. **Matrix Files** (for each variable):
   - `{filename}_T_cat_matrix_{timestamp}.csv`
   - `{filename}_Heat_Transfer_with_coolant_matrix_{timestamp}.csv`
   - etc.

3. **Summary Report**:
   - `{filename}_ramp_data_summary_{timestamp}.txt`

## Advanced Features

### Steady State Detection
- **Threshold**: 0.05 C/min (configurable)
- **Minimum Duration**: 10 minutes (configurable)
- **Isolated Point Removal**: Filters out numerical noise
- **Post-Ramp Search**: Focuses detection after ramp completion

### Plot Customization
- **Professional Styling**: Seaborn themes with viridis color palette
- **Consistent Time Axes**: All plots start at actual data start time
- **Interactive Elements**: Zoom, pan, and save capabilities
- **Multiple Figure Support**: Each analysis type in separate window

### Error Handling
- **Graceful Degradation**: Missing data variables handled appropriately
- **User Feedback**: Clear error messages and status updates
- **Thread Safety**: GUI remains responsive during analysis

## Troubleshooting

### Common Issues:
1. **"No file selected"**: Click "Select Aspen CSV File" first
2. **"No plots selected"**: Check at least one plot type checkbox
3. **"Analysis failed"**: Check console output for detailed error messages
4. **"Heat transfer data not available"**: 3D plot requires heat transfer variable in CSV

### Performance Tips:
- Use time limits for large datasets (>1000 time points)
- Close previous plot windows to free memory
- Save results incrementally for long analyses

## Code Architecture Benefits

### For Users:
- **Faster Analysis**: Optimized algorithms and parallel processing
- **Better Reliability**: Modular design with comprehensive error handling
- **Easier Usage**: Intuitive GUI with helpful status messages
- **Flexible Options**: Customizable analysis parameters

### For Developers:
- **Maintainable Code**: Clear separation of concerns
- **Extensible Design**: Easy to add new analysis types
- **Testable Components**: Individual modules can be tested independently
- **Configuration Management**: Centralized settings and parameters

## Future Enhancements

The modular architecture makes it easy to add:
- **Custom Analysis Modules**: New analysis types
- **Export Formats**: Additional file formats (Excel, HDF5, etc.)
- **Batch Processing**: Multiple file analysis
- **Parameter Optimization**: Automated parameter tuning
- **Web Interface**: Browser-based version

---

## Support

For questions or issues:
1. Check the console output for detailed error messages
2. Verify your CSV file format matches Aspen Plus export structure
3. Try with a smaller time range if experiencing performance issues
4. Review the analysis summary in the console for parameter detection results

**Happy Analyzing!**

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Contact
- **Author:** Seonggyun Kim
- **Email:** seonggyun.kim@outlook.com
- **Institution:** KTH Royal Institute of Technology

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
