# Data Loader Module: Extensible CSV Parsing System

## Overview

The new `data_loader.py` module provides a flexible, extensible framework for handling multiple CSV formats in your Dynamic Reactor Analysis Tool. This separation improves modularity, maintainability, and allows easy addition of new data formats.

## Key Benefits

### 1. **Format Extensibility**
- Easy to add new CSV formats without modifying analysis code
- Support for different header styles, comment formats, and data structures
- Automatic format detection and parsing

### 2. **Modular Design** 
- Clear separation between data loading and analysis logic
- Standardized data output format for all parsers
- Independent testing and maintenance of parsing logic

### 3. **Backward Compatibility**
- Existing code continues to work unchanged
- Legacy functions provide compatibility layer
- Gradual migration path to new system

### 4. **Robust Error Handling**
- Graceful fallback when formats don't match
- Detailed parsing statistics and error reporting
- Non-breaking updates when adding new parsers

## Current Supported Formats

### 1. **Aspen Plus Dynamics** (Primary Format)
```
Time,,,,,,,,,,,
Minutes,,,,,,,,,,,
12.05,0,0.1,0.2,0.3,...,5.0
,235.992,244.024,249.158,...     # T_cat data
,235.992,244.024,249.159,...     # T data  
,337.531,233.17,190.04,...       # Reaction Rate data
,-1.64E-05,6.40E-05,1.05E-04,... # Heat Transfer data
,-1.99953,-2.95102,-3.55962,...  # Coolant Heat Transfer data
```

**Features:**
- Spatial + temporal data (2D matrices)
- 5 predefined variables with units
- Every 6 rows = 1 time point + 5 variable rows

### 2. **Generic Time-Series** (Fallback Format)
```
Time,Temperature,Pressure,Flow_Rate
0.0,25.0,1.01,100.0
1.0,26.2,1.02,102.0
2.0,27.5,1.03,105.0
```

**Features:**
- Standard CSV with headers
- Time-series data (1D vectors)
- Any number of data columns

## Usage Examples

### Basic Usage (Existing Code)
```python
# Your existing code works unchanged
from analysis_engine import DataLoader

data_package = DataLoader.load_and_parse_aspen_data(file_path)
# Returns same format as before
```

### Advanced Usage (New Features)
```python
from data_loader import DataLoaderManager, DataFormat

# Create data loader with format detection
loader = DataLoaderManager()

# Load with automatic format detection
data = loader.load_data(file_path)

# Access standardized data structure
if data:
    print(f"Format: {data.metadata.format_type}")
    print(f"Time points: {data.n_time}")
    print(f"Spatial points: {data.n_spatial}")
    print(f"Variables: {list(data.variables.keys())}")
```

### Adding Custom Formats
```python
from data_loader import BaseDataParser, StandardDataPackage

class MyCustomParser(BaseDataParser):
    def can_parse(self, file_path: str) -> bool:
        # Check if file matches your format
        return file_path.endswith('_custom.csv')
    
    def parse(self, file_path: str) -> StandardDataPackage:
        # Parse your custom format
        # Return standardized data package
        pass
    
    def get_format_description(self) -> str:
        return "My custom CSV format"

# Add to loader
loader = DataLoaderManager()
loader.add_parser(MyCustomParser())
```

## Data Structure Standardization

All parsers output the same standardized format:

```python
@dataclass
class StandardDataPackage:
    time_vector: np.ndarray          # 1D time array
    length_vector: Optional[np.ndarray]  # 1D spatial array (or None)
    variables: Dict[str, np.ndarray]     # Variable name -> data matrix
    metadata: DataMetadata               # Format info, units, etc.
```

**Benefits:**
- Consistent interface for all analysis functions
- Easy to add metadata (units, format info, parsing notes)
- Future-proof for new data structures

## Integration with Analysis Engine

The `analysis_engine.py` automatically detects and uses the new data loader:

```python
# New modular approach (preferred)
try:
    from data_loader import DataLoaderManager
    # Uses new extensible system
except ImportError:
    # Falls back to legacy implementation
    # No disruption to existing functionality
```

## File Organization

```
Dynamic_Reactor_Analysis_Tool/
├── data_loader.py              # New extensible data loader
├── analysis_engine.py          # Core analysis (updated)
├── main_gui.py                 # GUI (compatible)
├── plot_generator.py           # Plotting (unchanged)
├── results_manager.py          # Results (unchanged)
├── example_custom_parser.py    # Example extension
└── README.md                   # Documentation
```

## Future Extensibility Examples

### Excel File Support
```python
class ExcelParser(BaseDataParser):
    def can_parse(self, file_path: str) -> bool:
        return file_path.endswith(('.xlsx', '.xls'))
    
    def parse(self, file_path: str) -> StandardDataPackage:
        df = pd.read_excel(file_path)
        # Convert to standard format
```

### JSON Time-Series
```python
class JSONTimeSeriesParser(BaseDataParser):
    def can_parse(self, file_path: str) -> bool:
        return file_path.endswith('.json')
    
    def parse(self, file_path: str) -> StandardDataPackage:
        import json
        with open(file_path) as f:
            data = json.load(f)
        # Convert to standard format
```

### Database Integration
```python
class DatabaseParser(BaseDataParser):
    def can_parse(self, file_path: str) -> bool:
        return file_path.startswith('db://')
    
    def parse(self, file_path: str) -> StandardDataPackage:
        # Connect to database and fetch data
        # Convert to standard format
```

## Migration Strategy

1. **Phase 1 (Current)**: New data loader works alongside existing code
2. **Phase 2**: Gradually update analysis functions to use StandardDataPackage
3. **Phase 3**: Remove legacy compatibility layer
4. **Phase 4**: Add advanced features (caching, validation, etc.)

## Performance Considerations

- **Memory Efficient**: Only loads data once per file
- **Format Caching**: Remembers successful formats for faster detection
- **Lazy Loading**: Can be extended for large file streaming
- **Error Recovery**: Graceful handling of malformed data

## Conclusion

The separated data loader provides a solid foundation for handling diverse CSV formats while maintaining backward compatibility. The modular design makes it easy to:

- Add support for new experimental data formats
- Handle variations in existing formats
- Integrate with different data sources
- Maintain clean separation of concerns

This architecture ensures your analysis tool can grow and adapt to new data requirements without disrupting existing functionality.
