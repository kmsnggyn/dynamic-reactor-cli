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
│              ┌─────────────────────────────────────┐            │
│              │             main_gui                │            │
│              │ • User interface & controls         │            │
│              │ • File selection & options          │            │
│              │ • Progress tracking & display       │            │
│              └─────────────┬───────────────────────┘            │
│                            │ coordinates analysis               │
│                            │                                    │
│    ┌──────────────────┐    │    ┌─────────────────┐             │
│    │ analysis_engine  │<───┼───>│ plot_generator  │             │
│    │                  │    │    │                 │             │
│    │ • Data loading   │    │    │ • Temperature   │             │
│    │ • Ramp detection │    │    │ • Stability     │             │
│    │ • Steady state   │    │    │ • Spatial       │             │
│    │ • Calculations   │    │    │ • 3D plots      │             │
│    └─────────┬────────┘    │    └─────────────────┘             │
│              │             │                                    │
│              │    ┌────────┴────────┐                           │ 
│              └───>│ results_manager │                           │
│                   │                 │                           │
│                   │ • CSV export    │                           │
│                   │ • Comparison    │                           │
│                   │ • File mgmt     │                           │
│                   │ • Config        │                           │
│                   └─────────────────┘                           │
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

### 1. **Prepare Data in Aspen Plus**

Before using this tool, you need to set up a flowsheet form in Aspen Plus and export the data:

#### **Step 1.1: Create Flowsheet Form**
1. In Aspen Plus, create a **Flowsheet Form** (Profile Plot or Table)
2. Set up the form with:
   - **X-axis**: Reactor length profile `L_Profile(*)`
   - **Y-axis variables** (add as separate series):
     - `T_cat(*)~1` - Catalyst temperature profile
     - `T(*)~1` - Gas temperature profile  
     - `Q_flux(*)~1` - Heat flux profile
     - `Q_cat(*)~1` - Heat transfer to catalyst profile
     - `iRxnModel.Rate(1,*).("<reaction-name>")` - Reaction rate profile
3. Configure the form to display all reactor positions along the length

#### **Step 1.2: Run Dynamic Simulation**
1. Execute your dynamic simulation with the desired ramp conditions
2. Ensure the simulation completes successfully
3. The flowsheet form will populate with time-series data

#### **Step 1.3: Export Data to CSV**
1. Open the completed flowsheet form/table
2. **Select All** data in the table (Ctrl+A)
3. **Copy** the entire table (Ctrl+C)
4. **Paste** into a new CSV file using a text editor or Excel
5. **Save** the file with the naming convention: `{duration}-{direction}-{curve}.csv`

**Example filenames:**
- `30-up-r.csv` - 30-minute linear ramp-up
- `45-down-s.csv` - 45-minute sinusoidal ramp-down
- `60-up-s.csv` - 60-minute sinusoidal ramp-up

### 2. **Launch the Application**
```bash
python main_gui.py
```

### 3. **Select Your Data File**
- Click "Select Aspen CSV File"
- Choose your exported Aspen Plus CSV file
- The filename will be displayed once selected

### 4. **Configure Analysis Options**
- **Time Limit**: Enter a maximum time (in minutes) to analyze, or leave empty for full range
- **Save Results**: Check to automatically save analysis results to files
- **Plot Selection**: Choose which plot types to generate:
  - **Temperature Response Analysis** (recommended)
  - **Stability Analysis** (recommended) 
  - **Spatial Temperature Gradients** (comprehensive analysis)
  - **3D Heat Transfer Analysis** (if heat transfer data available)

### 5. **Run Analysis**
- Click "Run Analysis" to start
- Progress bar and status updates will show analysis progress
- Multiple plot windows will open when complete

### 6. **Save Results** (Optional)
- If auto-save was enabled, results are saved automatically
- Use "Save Last Results" button to save results from previous analysis
- Saved files include:
  - Time and position vectors (CSV)
  - All variable matrices (CSV)
  - Analysis summary report (TXT)

## File Format Requirements

The tool expects CSV files exported from Aspen Plus Flowsheet Forms with the following structure:

### **Required Data Structure:**
- **Row 0**: "Time" header
- **Row 2**: Time value + reactor position headers (`L_Profile(*)`)
- **Rows 3-7**: Variable data rows containing:
  - `T_cat(*)~1` - Catalyst temperature profile
  - `T(*)~1` - Gas temperature profile
  - `iRxnModel.Rate(1,*).("<reaction-name>")` - Reaction rate profile
  - `Q_cat(*)~1` - Heat transfer to catalyst profile
  - `Q_flux(*)~1` - Heat transfer with coolant profile
- **Pattern repeats** every 6 rows for each time point

### **Variable Names Expected:**
The tool automatically detects these variable patterns:
- **Temperature**: `T_cat`, `T` (catalyst and gas temperatures)
- **Reaction Rate**: `Rate`, `iRxnModel.Rate` (reaction kinetics)
- **Heat Transfer**: `Q_cat`, `Q_flux` (heat transfer profiles)

### **Export Requirements:**
1. **Complete time series**: Include all time points from your dynamic simulation
2. **All spatial positions**: Include all reactor length positions
3. **Consistent formatting**: Use the standard Aspen Plus table export format
4. **Proper filename**: Follow the naming convention for automatic parameter detection

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

### **Aspen Plus Data Export Issues:**
1. **"Variable not found"**: Ensure all required variables are included in your flowsheet form
2. **"Inconsistent data structure"**: Verify the CSV export includes time, position, and all variable profiles
3. **"No reactor positions detected"**: Check that `L_Profile(*)` is properly set as the spatial coordinate
4. **"Missing time data"**: Ensure the dynamic simulation completed and time series data is available

### **Common Application Issues:**
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

### **Aspen Plus Setup:**
1. Verify your flowsheet form includes all required variables (`T_cat`, `T`, `Q_cat`, `Q_flux`, reaction rates)
2. Ensure `L_Profile(*)` is properly configured as the reactor length coordinate
3. Check that your dynamic simulation completed successfully before data export
4. Confirm the CSV export includes complete time-series data

### **Application Issues:**
1. Check the console output for detailed error messages
2. Verify your CSV file format matches the expected Aspen Plus export structure
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
