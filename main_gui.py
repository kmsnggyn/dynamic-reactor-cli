"""
Dynamic Reactor Ramp Analysis Tool with GUI
===========================================

A comprehensive analysis tool for Aspen Plus Dynamics simulation data with user-friendly GUI.
Analyzes catalyst temperature response during flow ramp experiments.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import sys
import re
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# Import modular classes and functions
try:
    from results_manager import ResultsComparison, DataExporter, ConfigManager, AnalysisReporter
    from analysis_engine import AnalysisOptions, DataLoader, SteadyStateDetector, RampParameters
    from plot_generator import PlotGenerator as PlotGen
    
    # Create aliases for shorter names
    DataExp = DataExporter
    ConfigMgr = ConfigManager
    AnalysisRep = AnalysisReporter
    
    print("✓ Imported modular components successfully")
except ImportError as e:
    print(f"Warning: Could not import modular components: {e}")
    print("Please ensure analysis_engine.py, plot_generator.py, and results_manager.py are available.")
    # Create placeholder classes to prevent crashes
    class ResultsComparison:
        COMPARISON_FILE = "Ramp_Analysis_Results_Comparison.csv"
        @staticmethod
        def extract_key_metrics(*args, **kwargs): return {}
        @staticmethod
        def update_comparison_file(*args, **kwargs): return ""
    
    class AnalysisOptions:
        def __init__(self, **kwargs): pass
    
    class DataLoader:
        @staticmethod
        def load_and_parse_aspen_data(*args, **kwargs): return {}
        @staticmethod
        def parse_ramp_parameters_from_filename(*args, **kwargs): return None
    
    class SteadyStateDetector:
        @staticmethod
        def detect_steady_state(*args, **kwargs): return None, {}
    
    class RampParameters:
        def __init__(self, **kwargs):
            self.duration = None
            self.start_time = 0
            self.end_time = 0
            self.direction = "none"
            self.curve_type = "none"
            self.analysis_title = "Analysis"
    
    class DataExporter:
        @staticmethod
        def save_data_structure(*args, **kwargs): return ""
    
    class ConfigManager:
        @staticmethod
        def get_config(*args, **kwargs): return {}
        @staticmethod
        def update_matplotlib_settings(*args, **kwargs): pass
    
    class AnalysisReporter:
        @staticmethod
        def print_analysis_summary(*args, **kwargs): pass
    
    # Create aliases
    DataExp = DataExporter
    ConfigMgr = ConfigManager
    AnalysisRep = AnalysisReporter
    PlotGen = None
from matplotlib.patches import Patch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from typing import Optional, Dict, List, Tuple, Any
import contextlib
import io
import shutil
import seaborn as sns

# Configure matplotlib settings
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette('viridis')
except:
    # Fallback if seaborn style is not available
    plt.style.use('default')

# GUI classes start here

class AnalysisGUI:
    """Main GUI application for dynamic ramp analysis"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dynamic Reactor Ramp Analysis Tool")
        self.root.geometry("500x650")  # Set to match base_width
        
        # Expansion state variables
        self.base_width = 500  # Increased from 400 to make collapsed window wider
        self.expanded_width = 940  # Increased from 900 to account for wider terminal (500 + 420 + 20 padding)
        self.total_expanded_width = 1740  # Increased from 1700 to account for wider terminal
        self.current_width = self.base_width
        self.terminal_panel_visible = False
        self.comparison_panel_visible = False
        
        # Initially allow resizing for dynamic expansion, but control it manually
        self.root.resizable(False, False)  # Disable auto-resizing
        
        # Variables
        self.selected_file = tk.StringVar()
        self.time_limit_var = tk.StringVar(value="")
        self.save_results_var = tk.BooleanVar(value=False)
        
        # Plot selection variables
        self.plot_vars = {
            'temperature_response': tk.BooleanVar(value=True),
            'stability_analysis': tk.BooleanVar(value=True),
            'spatial_gradients': tk.BooleanVar(value=True),
            'heat_transfer_3d': tk.BooleanVar(value=True)
        }
        
        # Store data for saving
        self.last_data_package = None
        
        # Store timestamp from selected file for comparison table
        self.current_file_timestamp = None
        
        # Terminal output capture
        self.terminal_output = []
        
        # Filter variables for comparison table
        self.filter_duration = tk.StringVar(value="All")
        self.filter_direction = tk.StringVar(value="All") 
        self.filter_curve = tk.StringVar(value="All")
        self.filter_active = tk.BooleanVar(value=False)
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Create main container with horizontal layout
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Main control panel - positioning will be handled dynamically
        self.main_frame = ttk.Frame(container, padding="10", width=480)  # Increased from 380 to match new base width
        self.main_frame.pack_propagate(False)  # Prevent frame from shrinking to content
        
        # Terminal panel (middle) - initially hidden, increased width for scrollbar
        self.terminal_frame = ttk.Frame(container, padding="10", width=420)  # Increased from 380 to 420
        self.terminal_frame.pack_propagate(False)  # Always maintain fixed width
        
        # Comparison panel (right) - initially hidden, much wider for better table viewing
        self.comparison_frame = ttk.Frame(container, padding="10", width=780)
        self.comparison_frame.pack_propagate(False)  # Always maintain fixed width
        
        self.setup_main_panel()
        self.setup_terminal_panel()
        self.setup_comparison_panel()
        
        # Position main frame initially (centered when alone)
        self._update_main_frame_position()
        
        # Ensure initial window size is correct
        self.update_window_size()
    
    def setup_main_panel(self):
        """Setup the main control panel"""
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Dynamic Reactor Ramp Analysis", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(self.main_frame, text="Data File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Aspen CSV File", 
                  command=self.select_file).grid(row=0, column=0, sticky=tk.W)
        
        self.file_label = ttk.Label(file_frame, textvariable=self.selected_file, 
                                   foreground="blue", font=('Arial', 9))
        self.file_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Analysis options section
        options_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Time limit
        ttk.Label(options_frame, text="Time Limit (min):").grid(row=0, column=0, sticky=tk.W)
        time_entry = ttk.Entry(options_frame, textvariable=self.time_limit_var, width=10)
        time_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        ttk.Label(options_frame, text="(leave empty for full range)", 
                 font=('Arial', 8)).grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Save results option
        ttk.Checkbutton(options_frame, text="Save analysis results to files", 
                       variable=self.save_results_var).grid(row=1, column=0, columnspan=3, 
                                                           sticky=tk.W, pady=(10, 0))
        
        # Info about automatic comparison file
        info_label = ttk.Label(options_frame, 
                              text="Note: Key results are automatically saved to 'Ramp_Analysis_Results_Comparison.csv'",
                              font=('Arial', 8), foreground='gray')
        info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Plot selection section
        plots_frame = ttk.LabelFrame(self.main_frame, text="Select Plots to Generate", padding="10")
        plots_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Select all/none buttons
        button_frame = ttk.Frame(plots_frame)
        button_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(button_frame, text="Select All", 
                  command=self.select_all_plots).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Select None", 
                  command=self.select_no_plots).pack(side=tk.LEFT)
        
        # Plot checkboxes
        plot_descriptions = {
            'temperature_response': 'Temperature Response Analysis',
            'stability_analysis': 'Stability Analysis',
            'spatial_gradients': 'Spatial Temperature Gradients',
            'heat_transfer_3d': '3D Heat Transfer Analysis'
        }
        
        row = 1
        for key, description in plot_descriptions.items():
            ttk.Checkbutton(plots_frame, text=description, 
                           variable=self.plot_vars[key]).grid(row=row, column=0, 
                                                            sticky=tk.W, pady=2)
            row += 1
        
        # Control buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
        
        self.analyze_button = ttk.Button(button_frame, text="Run Analysis", 
                                        command=self.run_analysis, state="disabled")
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="Save Last Results", 
                                     command=self.save_last_results, state="disabled")
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.LEFT)
        
        # View panels buttons - moved to bottom right in separate rows
        view_frame = ttk.Frame(self.main_frame)
        view_frame.grid(row=8, column=1, pady=(10, 0), sticky=(tk.E, tk.S))
        
        # Terminal panel toggle
        self.terminal_toggle_button = ttk.Button(view_frame, text="Terminal Output ▶", width=18,
                                               command=self.toggle_terminal_panel)
        self.terminal_toggle_button.grid(row=0, column=0, sticky=tk.E, pady=(0, 2))
        
        # Comparison panel toggle
        self.comparison_toggle_button = ttk.Button(view_frame, text="Comparison Table ▶", width=18,
                                                 command=self.toggle_comparison_panel)
        self.comparison_toggle_button.grid(row=1, column=0, sticky=tk.E)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready. Please select a file to begin.")
        self.status_label.grid(row=7, column=0, columnspan=2, pady=(10, 0))
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(0, weight=1)
    
    def setup_terminal_panel(self):
        """Setup the terminal output panel"""
        
        # Create a main LabelFrame for the entire Terminal Output section
        terminal_main_frame = ttk.LabelFrame(self.terminal_frame, text="Terminal Output", padding="10")
        terminal_main_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Terminal output text widget with scrollbar
        terminal_text_frame = ttk.Frame(terminal_main_frame)
        terminal_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.terminal_text = tk.Text(terminal_text_frame, 
                                    width=48, height=30,  # Reduced width from 50 to 48 to fit scrollbar better
                                    font=('Consolas', 9),
                                    bg='white', fg='black',
                                    wrap=tk.WORD)
        terminal_scrollbar = ttk.Scrollbar(terminal_text_frame, orient=tk.VERTICAL, 
                                         command=self.terminal_text.yview)
        self.terminal_text.configure(yscrollcommand=terminal_scrollbar.set)
        
        self.terminal_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        terminal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Terminal controls
        terminal_controls = ttk.Frame(terminal_main_frame)
        terminal_controls.pack(pady=(10, 0))
        
        ttk.Button(terminal_controls, text="Clear", 
                  command=self.clear_terminal).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(terminal_controls, text="Save Log", 
                  command=self.save_terminal_log).pack(side=tk.LEFT)
    
    def setup_comparison_panel(self):
        """Setup the comparison results panel"""
        
        # Create a main LabelFrame for the entire Results Comparison section
        results_frame = ttk.LabelFrame(self.comparison_frame, text="Results Comparison", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Comparison controls at the top
        comparison_controls = ttk.Frame(results_frame)
        comparison_controls.pack(pady=(0, 10), fill=tk.X)
        
        # Data Filters section - moved to top
        filters_frame = ttk.LabelFrame(comparison_controls, text="Data Filters", padding="10")
        filters_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Filter controls
        filter_content = ttk.Frame(filters_frame)
        filter_content.pack(fill=tk.X)
        
        ttk.Checkbutton(filter_content, text="Enable Filters", 
                       variable=self.filter_active,
                       command=self.apply_filters).grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        
        ttk.Label(filter_content, text="Duration:").grid(row=0, column=1, padx=(10, 5), sticky=tk.W)
        self.duration_filter = ttk.Combobox(filter_content, textvariable=self.filter_duration, 
                                          width=8, state="readonly")
        self.duration_filter.grid(row=0, column=2, padx=(0, 15))
        self.duration_filter.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        ttk.Label(filter_content, text="Direction:").grid(row=0, column=3, padx=(10, 5), sticky=tk.W)
        self.direction_filter = ttk.Combobox(filter_content, textvariable=self.filter_direction,
                                           width=8, state="readonly")
        self.direction_filter.grid(row=0, column=4, padx=(0, 15))
        self.direction_filter.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        ttk.Label(filter_content, text="Curve:").grid(row=0, column=5, padx=(10, 5), sticky=tk.W)
        self.curve_filter = ttk.Combobox(filter_content, textvariable=self.filter_curve,
                                       width=10, state="readonly")
        self.curve_filter.grid(row=0, column=6, padx=(0, 15))
        self.curve_filter.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        ttk.Button(filter_content, text="Clear All Filters", width=15,
                  command=self.clear_filters).grid(row=0, column=7, padx=(10, 0))
        
        # Comparison table frame with scrollbars
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview widget for table display with resizable columns
        self.comparison_tree = ttk.Treeview(table_frame)
        
        # Configure header styling - bold font with bottom border
        style = ttk.Style()
        style.configure("Treeview.Heading", 
                       font=('Arial', 10, 'bold'),  # Bold font
                       relief="raised",              # Raised border for better visibility
                       borderwidth=2)                # Thicker border width
        
        # Configure row styling to ensure proper text alignment
        style.configure("Treeview", 
                       fieldbackground="white",      # Background color
                       rowheight=20)                 # Row height
        
        # Configure the TreeView to remove all tree-related styling
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        
        # Configure custom style to completely eliminate indentation
        style.configure("NoIndent.Treeview", 
                       indent=0,                     # Remove tree indentation
                       fieldbackground="white",
                       rowheight=20)
        
        # Also configure the layout for NoIndent style
        style.layout("NoIndent.Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        
        # Right-click context menu for removing columns
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Remove Column", command=self.remove_selected_column)
        
        # Bind right-click to show context menu
        self.comparison_tree.bind("<Button-3>", self.show_context_menu)  # Right-click on Windows
        self.comparison_tree.bind("<Control-Button-1>", self.show_context_menu)  # Ctrl+click on Mac
        
        # Scrollbars for the table
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.comparison_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.comparison_tree.xview)
        self.comparison_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack the treeview and scrollbars using grid for better control
        self.comparison_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights to make table resizable
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bottom controls section
        bottom_controls = ttk.Frame(results_frame)
        bottom_controls.pack(pady=(10, 0), fill=tk.X)
        
        # Table Actions section
        actions_frame = ttk.LabelFrame(bottom_controls, text="Table Actions", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(actions_frame, text="Refresh", 
                  command=self.refresh_comparison_table).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(actions_frame, text="Export CSV", 
                  command=self.export_comparison_table).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(actions_frame, text="Remove Column", 
                  command=self.remove_selected_column).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(actions_frame, text="Check File Status", 
                  command=self.check_file_status).pack(side=tk.LEFT, padx=(0, 10))
        
        # Column Order section
        order_frame = ttk.LabelFrame(bottom_controls, text="Column Order", padding="10")
        order_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Column order controls in horizontal layout
        order_content = ttk.Frame(order_frame)
        order_content.pack(fill=tk.X)
        
        self.column_listbox = tk.Listbox(order_content, height=3, width=25, font=('Arial', 8))
        self.column_listbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Column ordering buttons - horizontal layout
        order_buttons = ttk.Frame(order_content)
        order_buttons.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(order_buttons, text="↑", width=3,
                  command=self.move_column_up).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(order_buttons, text="↓", width=3,
                  command=self.move_column_down).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(order_buttons, text="Apply Order", width=12,
                  command=self.apply_column_order).pack(side=tk.LEFT)
        
        # Instructions label
        instructions = ttk.Label(bottom_controls, 
                                text="Right-click column header to remove • Data columns are resizable • Metric/Units columns are fixed width • Use filters to show/hide specific experiment types",
                                font=('Arial', 8), foreground='gray')
        instructions.pack(pady=(10, 0))
    
    def toggle_terminal_panel(self):
        """Toggle the terminal output panel visibility"""
        if not self.terminal_panel_visible:
            # Show terminal panel
            self.terminal_panel_visible = True
            self.terminal_toggle_button.config(text="Terminal Output ◀")
        else:
            # Hide terminal panel
            self.terminal_frame.pack_forget()
            self.terminal_panel_visible = False
            self.terminal_toggle_button.config(text="Terminal Output ▶")
        
        # Ensure main frame stays fixed width and update positioning
        self.main_frame.pack_propagate(False)
        self._update_main_frame_position()
        self.update_window_size()
    
    def toggle_comparison_panel(self):
        """Toggle the comparison table panel visibility"""
        if not self.comparison_panel_visible:
            # Show comparison panel
            self.comparison_panel_visible = True
            self.comparison_toggle_button.config(text="Comparison Table ◀")
            self.refresh_comparison_table()  # Load data when showing
        else:
            # Hide comparison panel
            self.comparison_frame.pack_forget()
            self.comparison_panel_visible = False
            self.comparison_toggle_button.config(text="Comparison Table ▶")
        
        # Ensure main frame stays fixed width and update positioning
        self.main_frame.pack_propagate(False)
        self._update_main_frame_position()
        self.update_window_size()
    
    def update_window_size(self):
        """Update window size based on visible panels"""
        # Calculate the required width based on visible panels
        if self.terminal_panel_visible and self.comparison_panel_visible:
            new_width = self.total_expanded_width  # 1740px for all three panels
        elif self.comparison_panel_visible:
            new_width = 1300  # 500px main + 780px comparison panel + 20px padding
        elif self.terminal_panel_visible:
            new_width = self.expanded_width  # 940px for main + terminal
        else:
            new_width = self.base_width  # 500px main panel only
        
        # Only update if the width actually changed
        if new_width != self.current_width:
            self.current_width = new_width
            # Set the window geometry with fixed height
            self.root.geometry(f"{new_width}x650")
            # Ensure the window doesn't auto-resize based on content
            self.root.update_idletasks()
            # Force the window to stay at the specified size
            self.root.minsize(new_width, 650)
            self.root.maxsize(new_width, 650)
    
    def _update_main_frame_position(self):
        """Update main frame position: center when alone, left when panels are visible"""
        any_panel_visible = self.terminal_panel_visible or self.comparison_panel_visible
        
        # Always pack main frame first to ensure it stays on the left
        self.main_frame.pack_forget()
        if any_panel_visible:
            # Position on the left when other panels are visible
            self.main_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        else:
            # Center when it's the only visible panel - use expand=True for true centering
            self.main_frame.pack(expand=True, fill=tk.Y)
        
        # Re-pack other panels in the correct order after main frame
        if self.terminal_panel_visible:
            self.terminal_frame.pack_forget()
            self.terminal_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
            self.terminal_frame.pack_propagate(False)
        
        if self.comparison_panel_visible:
            self.comparison_frame.pack_forget()
            self.comparison_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
            self.comparison_frame.pack_propagate(False)
    
    def add_terminal_output(self, text):
        """Add text to terminal output"""
        if hasattr(self, 'terminal_text'):
            self.terminal_text.insert(tk.END, text + "\n")
            self.terminal_text.see(tk.END)
        self.terminal_output.append(text)
    
    def clear_terminal(self):
        """Clear terminal output"""
        if hasattr(self, 'terminal_text'):
            self.terminal_text.delete(1.0, tk.END)
        self.terminal_output.clear()
    
    def save_terminal_log(self):
        """Save terminal output to file"""
        if not self.terminal_output:
            messagebox.showinfo("No Data", "No terminal output to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Terminal Log"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("\n".join(self.terminal_output))
                messagebox.showinfo("Success", f"Terminal log saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {str(e)}")
    
    def _extract_timestamp_from_column(self, column_name):
        """Extract timestamp from column name with format like '20-down-s-20250801-231457'"""
        try:
            # Pattern to match YYYYMMDD-HHMMSS at the end of the column name
            pattern = r'(\d{8})-(\d{6})(?:\.csv)?$'
            match = re.search(pattern, column_name)
            
            if match:
                date_part = match.group(1)  # YYYYMMDD
                time_part = match.group(2)  # HHMMSS
                
                # Convert to readable format
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                
                hour = time_part[:2]
                minute = time_part[2:4]
                second = time_part[4:6]
                
                formatted_date = f"{year}-{month}-{day}"
                formatted_time = f"{hour}:{minute}:{second}"
                
                return formatted_date, formatted_time
            else:
                # Fallback: check if column name looks like a file path or has identifiable patterns
                if '.csv' in column_name.lower() or '\\' in column_name or '/' in column_name:
                    # Extract just the filename if it's a full path
                    filename = column_name.split('\\')[-1].split('/')[-1]
                    # Try pattern matching on the filename
                    pattern_match = re.search(r'(\d{8})-(\d{6})', filename)
                    if pattern_match:
                        date_part = pattern_match.group(1)
                        time_part = pattern_match.group(2)
                        
                        year = date_part[:4]
                        month = date_part[4:6]
                        day = date_part[6:8]
                        hour = time_part[:2]
                        minute = time_part[2:4]
                        second = time_part[4:6]
                        
                        return f"{year}-{month}-{day}", f"{hour}:{minute}:{second}"
                
                # Final fallback: return column name truncated for date and N/A for time
                return column_name[:10] if len(column_name) >= 10 else column_name, "N/A"
                
        except Exception:
            # Fallback: return column name for date and N/A for time
            return column_name[:10] if len(column_name) >= 10 else column_name, "N/A"
    
    def _curve_code_to_display(self, curve_code):
        """Convert curve code to display name (s -> Sinusoidal, r -> Linear)"""
        if curve_code == 's':
            return 'Sinusoidal'
        elif curve_code == 'r':
            return 'Linear'
        else:
            return curve_code
    
    def _curve_display_to_code(self, curve_display):
        """Convert display name to curve code (Sinusoidal -> s, Linear -> r)"""
        if curve_display == 'Sinusoidal':
            return 's'
        elif curve_display == 'Linear':
            return 'r'
        else:
            return curve_display
    
    def _column_name_for_display(self, column_name):
        """Return original column name without conversion"""
        return column_name
    
    def _format_for_display(self, value, metric_name=None):
        """Format values for display with special handling for specific metrics"""
        try:
            # Handle special metric types with specific value conversions
            if metric_name == 'Ramp_Curve_Type':
                # Convert curve codes to display names: s -> Sinusoidal, r -> Linear
                if value == 's':
                    return 'Sinusoidal'
                elif value == 'r':
                    return 'Linear'
                else:
                    return str(value) if value not in ['', 'N/A', '-'] else value
            
            elif metric_name == 'Ramp_Direction':
                # Convert direction codes to display names: d -> Down, u -> Up, down -> Down, up -> Up
                if value in ['d', 'down']:
                    return 'Down'
                elif value in ['u', 'up']:
                    return 'Up'
                else:
                    return str(value) if value not in ['', 'N/A', '-'] else value
            
            # Handle empty values first
            if pd.isna(value) or value == '' or str(value).strip() == '':
                return ''
            
            # Handle 'N/A' and similar string values
            if isinstance(value, str) and value.strip().lower() in ['n/a', 'na', 'nan', 'none', '-']:
                return value
            
            # Try to convert to float for numeric formatting
            num_value = float(value)
            
            # Handle zero
            if num_value == 0:
                return '0'
            
            # Format to 5 significant figures
            # Use scientific notation for very large or very small numbers
            abs_value = abs(num_value)
            if abs_value >= 1e6 or abs_value < 1e-3:
                # Scientific notation with 4 decimal places (5 sig figs total)
                formatted = f"{num_value:.4e}"
            else:
                # Determine number of decimal places needed for 5 sig figs
                import math
                if abs_value >= 1:
                    # For numbers >= 1, decimal places = 5 - number of digits before decimal
                    digits_before_decimal = len(str(int(abs_value)))
                    decimal_places = max(0, 5 - digits_before_decimal)
                    formatted = f"{num_value:.{decimal_places}f}"
                else:
                    # For numbers < 1, use g format which handles sig figs well
                    formatted = f"{num_value:.4g}"
            
            # Clean up trailing zeros for non-scientific notation
            if 'e' not in formatted.lower():
                formatted = formatted.rstrip('0').rstrip('.')
            
            return formatted
            
        except (ValueError, TypeError):
            # If conversion fails, return original value as string
            return str(value)

    def refresh_comparison_table(self):
        """Refresh the comparison table with latest data, respecting active filters"""
        # If filters are active, delegate to apply_filters which will refresh with filters applied
        if hasattr(self, 'filter_active') and self.filter_active.get():
            self.apply_filters()
            return
        
        # Otherwise, refresh with full data (no filters)
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            # Clear the table and show message
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            self.comparison_tree.heading("#0", text="No comparison data available")
            return
        
        try:
            # Read the comparison file, skipping comment lines
            df = pd.read_csv(comparison_file, index_col=0, comment='#')
            
            # Clean up empty rows (metrics with no data)
            rows_before = len(df.index)
            
            # Find rows where all data columns (excluding Units) are empty, NaN, or 'N/A'
            data_columns = [col for col in df.columns if col != 'Units']
            if data_columns:
                # Check for rows that are completely empty across all data columns
                # Handle various empty representations: NaN, 'N/A', empty string, whitespace
                empty_mask = df[data_columns].isna().all(axis=1) | \
                           (df[data_columns] == 'N/A').all(axis=1) | \
                           (df[data_columns] == '').all(axis=1)
                
                # Check for whitespace-only strings for each column separately
                for col in data_columns:
                    empty_mask = empty_mask | (df[col].astype(str).str.strip() == '')
                
                # Get list of rows that will be removed for logging
                empty_rows = df.index[empty_mask].tolist()
                
                # Remove empty rows
                df_cleaned = df[~empty_mask]
                
                rows_after = len(df_cleaned.index)
                removed_rows = rows_before - rows_after
                
                if removed_rows > 0:
                    try:
                        # Save the cleaned data back to file
                        df_cleaned.to_csv(comparison_file)
                        
                        # Add metadata header comment back
                        with open(comparison_file, 'r') as f:
                            content = f.read()
                        
                        header_comment = f"""# Ramp Analysis Results Comparison
# Generated by Dynamic Reactor Ramp Analysis Tool
# Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
# 
# Each column represents one ramp test configuration: Duration_Direction_CurveType
# Each row represents a key metric from the analysis
# Units column shows the measurement units for each metric
# Same ramp configurations will overwrite previous results
#
"""
                        
                        with open(comparison_file, 'w') as f:
                            f.write(header_comment + content)
                        
                        # Update df to use the cleaned version
                        df = df_cleaned
                        
                        # Log the cleanup action with details
                        self.add_terminal_output(f"🧹 Cleaned up {removed_rows} empty metric row(s) from comparison table")
                        if len(empty_rows) <= 5:  # Show names if not too many
                            for row_name in empty_rows:
                                self.add_terminal_output(f"   • Removed: {row_name}")
                        else:
                            self.add_terminal_output(f"   • Removed {len(empty_rows)} metrics (too many to list)")
                        
                    except Exception as cleanup_error:
                        self.add_terminal_output(f"Warning: Could not save cleaned data: {cleanup_error}")
                        # Continue with original data if cleanup fails
                
            else:
                # No data columns, use original dataframe
                pass
            
            # Clear existing data
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            
            # Configure columns - use headings only to eliminate tree structure
            columns = ['Metric'] + list(df.columns)
            self.comparison_tree['columns'] = columns  # Include all columns
            self.comparison_tree['show'] = 'headings'  # Only show headings, no tree structure
            
            # Configure the treeview to minimize indentation for the first column
            self.comparison_tree.configure(style="NoIndent.Treeview")
            
            # Set column headings and widths - make columns resizable
            self.comparison_tree.heading("Metric", text="Metric", anchor=tk.W)
            
            # Calculate optimal width for Metric column based on content (excluding Source_File)
            if len(df.index) > 0:
                display_metrics = [metric for metric in df.index if metric != 'Source_File']
                if display_metrics:
                    # Calculate width based on longest metric name, exported date/time, and add some padding
                    metric_names = display_metrics + ["Exported Date", "Exported Time", "Ramp_Rate"]
                    metric_width = max([len(str(metric)) for metric in metric_names]) * 8 + 30
                else:
                    metric_width = len("Exported Date") * 8 + 30  # Use frozen row as minimum
            else:
                metric_width = len("Exported Date") * 8 + 30  # Use frozen row as minimum
            
            # Apply 30% reduction as requested
            metric_width = int(metric_width * 0.7)
            
            # Set reasonable bounds for metric column width  
            metric_width = max(85, min(metric_width, 140))  # Between 85px and 140px (30% smaller)
            self.comparison_tree.column("Metric", width=metric_width, anchor=tk.W, minwidth=metric_width, stretch=False)  # Fixed width metric column
            
            # Calculate optimal width for Units column based on content (excluding Source_File)
            units_width = 60  # Default minimum
            if "Units" in df.columns and len(df.index) > 0:
                try:
                    display_metrics = [metric for metric in df.index if metric != 'Source_File']
                    if display_metrics:
                        # Get all unit values and calculate max width needed
                        unit_values = [str(df.loc[metric, "Units"]) for metric in display_metrics if metric in df.index]
                        unit_values.append("Units")  # Include header
                        max_unit_length = max([len(val) for val in unit_values])
                        units_width = max(units_width, max_unit_length * 8 + 20)
                except:
                    units_width = 80  # Fallback if there's an error
                    
                # Apply 30% reduction as requested
                units_width = int(units_width * 0.7)
                units_width = min(units_width, 70)  # Cap at reasonable maximum (30% smaller)
            else:
                # Apply 30% reduction to default as well
                units_width = int(units_width * 0.7)  # Make default narrower too
            
            # Reorder columns to ensure Units is always second (after Metric)
            if "Units" in df.columns:
                data_columns = [col for col in df.columns if col != "Units"]
                ordered_columns = ["Units"] + data_columns  # Units first among df.columns, then data columns
            else:
                ordered_columns = list(df.columns)
            
            for col in ordered_columns:
                display_name = self._column_name_for_display(col)
                self.comparison_tree.heading(col, text=display_name, anchor=tk.W)
                # Make columns resizable with minimum widths
                if col == "Units":
                    self.comparison_tree.column(col, width=units_width, anchor=tk.W, minwidth=units_width, stretch=False)  # Fixed width units column
                else:
                    self.comparison_tree.column(col, width=180, anchor=tk.W, minwidth=120, stretch=True)  # Resizable data columns
            
            # Add frozen date and time rows first, extracting timestamp from Source_File for each column
            date_values = ["Exported Date"]  # Start with metric name
            time_values = ["Exported Time"]  # Start with metric name
            for col in ordered_columns:  # Use the same column order as headers
                if col == "Units":
                    date_values.append("-")
                    time_values.append("-")
                else:
                    # Extract timestamp from the Source_File value for this column
                    if 'Source_File' in df.index:
                        source_file = df.loc['Source_File', col]
                        extracted_date, extracted_time = self._extract_timestamp_from_column(source_file)
                    else:
                        # Fallback to column name if Source_File not available
                        extracted_date, extracted_time = self._extract_timestamp_from_column(col)
                    date_values.append(extracted_date)
                    time_values.append(extracted_time)
            
            # Insert date and time rows at the top as normal rows
            date_item = self.comparison_tree.insert("", 0, values=date_values)
            time_item = self.comparison_tree.insert("", 1, values=time_values)
            
            # Add Ramp_Rate row first as a priority metric
            next_row = 2
            if 'Ramp_Rate' in df.index:
                formatted_values = ['Ramp_Rate']  # Start with metric name as first column
                for col in ordered_columns:  # Use the same column order as headers
                    raw_value = df.loc['Ramp_Rate', col]
                    if col == "Units":
                        # Don't format units column
                        formatted_values.append(str(raw_value))
                    else:
                        # Format values with metric-specific handling
                        formatted_values.append(self._format_for_display(raw_value, 'Ramp_Rate'))
                
                self.comparison_tree.insert("", next_row, values=formatted_values)
                next_row += 1
            
            # Add remaining data rows with formatted display values (5 significant figures)
            # Exclude Source_File and Ramp_Rate (already displayed) from display but keep them in the data for timestamp extraction
            for metric in df.index:
                if metric in ['Source_File', 'Ramp_Rate']:
                    continue  # Skip displaying Source_File row and Ramp_Rate (already shown)
                    
                formatted_values = [metric]  # Start with metric name as first column
                for col in ordered_columns:  # Use the same column order as headers
                    raw_value = df.loc[metric, col]
                    if col == "Units":
                        # Don't format units column
                        formatted_values.append(str(raw_value))
                    else:
                        # Format values with metric-specific handling
                        formatted_values.append(self._format_for_display(raw_value, metric))
                
                self.comparison_tree.insert("", tk.END, values=formatted_values)
            
            # Update column order listbox (exclude Units column from reordering)
            self.update_column_order_listbox(ordered_columns)
            
            # Update filter options based on available data
            self.update_filter_options(df)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load comparison data: {str(e)}")
            # Clear the table on error
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            self.comparison_tree.heading("#0", text="Error loading data")
    
    def update_column_order_listbox(self, columns):
        """Update the column order listbox with current data columns"""
        self.column_listbox.delete(0, tk.END)
        # Exclude Units from reordering since it's now always second
        if isinstance(columns, list) and "Units" in columns:
            data_columns = [col for col in columns if col != "Units"]
        else:
            data_columns = [col for col in columns if col != "Units"]
        for col in data_columns:
            self.column_listbox.insert(tk.END, col)
    
    def move_column_up(self):
        """Move selected column up in the order"""
        selection = self.column_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        
        idx = selection[0]
        # Get the item and remove it
        item = self.column_listbox.get(idx)
        self.column_listbox.delete(idx)
        # Insert it one position up
        self.column_listbox.insert(idx - 1, item)
        # Keep it selected
        self.column_listbox.selection_set(idx - 1)
    
    def move_column_down(self):
        """Move selected column down in the order"""
        selection = self.column_listbox.curselection()
        if not selection or selection[0] == self.column_listbox.size() - 1:
            return
        
        idx = selection[0]
        # Get the item and remove it
        item = self.column_listbox.get(idx)
        self.column_listbox.delete(idx)
        # Insert it one position down
        self.column_listbox.insert(idx + 1, item)
        # Keep it selected
        self.column_listbox.selection_set(idx + 1)
    
    def apply_column_order(self):
        """Apply the new column order to the CSV file and refresh table"""
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            messagebox.showwarning("No Data", "No comparison file to reorder.")
            return
        
        try:
            # Get new column order from listbox
            new_order = []
            for i in range(self.column_listbox.size()):
                new_order.append(self.column_listbox.get(i))
            
            if not new_order:
                return
            
            # Read current CSV
            df = pd.read_csv(comparison_file, index_col=0, comment='#')
            
            # Create new column order (always keep Units as the first column, followed by new data order)
            if "Units" in df.columns:
                new_columns = ["Units"] + new_order  # Units first, then reordered data columns
            else:
                new_columns = new_order
            
            # Reorder DataFrame columns
            df_reordered = df[new_columns]
            
            # Save back to file with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df_reordered.to_csv(comparison_file)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        retry_result = messagebox.askretrycancel(
                            "File Access Error", 
                            f"Cannot access the comparison file. It may be open in Excel or another program.\n\n"
                            f"Please close the file and click 'Retry', or click 'Cancel' to abort.\n\n"
                            f"File: {os.path.basename(comparison_file)}\n"
                            f"Attempt: {attempt + 1}/{max_retries}"
                        )
                        if not retry_result:
                            return
                    else:
                        messagebox.showerror(
                            "File Access Error", 
                            f"Unable to modify the comparison file after {max_retries} attempts.\n\n"
                            f"The file may be open in Excel or another program.\n"
                            f"Please close all applications using this file and try again.\n\n"
                            f"File: {comparison_file}"
                        )
                        return
            
            # Add metadata header comment back
            with open(comparison_file, 'r') as f:
                content = f.read()
            
            header_comment = f"""# Ramp Analysis Results Comparison
# Generated by Dynamic Reactor Ramp Analysis Tool
# Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
# 
# Each column represents one ramp test configuration: Duration_Direction_CurveType
# Each row represents a key metric from the analysis
# Units column shows the measurement units for each metric
# Same ramp configurations will overwrite previous results
#
"""
            
            with open(comparison_file, 'w') as f:
                f.write(header_comment + content)
            
            # Refresh the display
            self.refresh_comparison_table()
            
            # Show success message
            messagebox.showinfo("Success", "Column order has been updated successfully.")
            self.add_terminal_output(f"Applied new column order: {' → '.join(new_order)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reorder columns: {str(e)}")
            print(f"Error reordering columns: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load comparison data: {str(e)}")
            # Clear the table on error
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            self.comparison_tree.heading("#0", text="Error loading data")
    
    def export_comparison_table(self):
        """Export the comparison table to a new CSV file"""
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            messagebox.showinfo("No Data", "No comparison data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Comparison Table"
        )
        
        if file_path:
            try:
                # Copy the comparison file to the new location
                import shutil
                shutil.copy2(comparison_file, file_path)
                messagebox.showinfo("Success", f"Comparison table exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export table: {str(e)}")
    
    def extract_column_parameters(self, column_name):
        """Extract ramp parameters from column name (e.g., '20min_down_s' -> {'duration': '20min', 'direction': 'down', 'curve': 'Sinusoidal'})"""
        if column_name == "Units":
            return None
        
        # Parse column name format: {duration}_{direction}_{curve}
        parts = column_name.split('_')
        if len(parts) >= 3:
            return {
                'duration': parts[0],
                'direction': parts[1], 
                'curve': self._curve_code_to_display(parts[2])  # Convert to display name
            }
        return None
    
    def update_filter_options(self, df):
        """Update filter dropdown options based on available data"""
        if df.empty:
            return
        
        # Get all data columns (excluding Units)
        data_columns = [col for col in df.columns if col != 'Units']
        
        # Extract parameters from column names
        durations = set()
        directions = set()
        curves = set()
        
        for col in data_columns:
            params = self.extract_column_parameters(col)
            if params:
                durations.add(params['duration'])
                directions.add(params['direction'])
                curves.add(params['curve'])
        
        # Update filter comboboxes
        duration_values = ['All'] + sorted(list(durations))
        direction_values = ['All'] + sorted(list(directions))
        curve_values = ['All'] + sorted(list(curves))
        
        self.duration_filter['values'] = duration_values
        self.direction_filter['values'] = direction_values
        self.curve_filter['values'] = curve_values
        
        # Reset to 'All' if current selection is not available
        if self.filter_duration.get() not in duration_values:
            self.filter_duration.set('All')
        if self.filter_direction.get() not in direction_values:
            self.filter_direction.set('All')
        if self.filter_curve.get() not in curve_values:
            self.filter_curve.set('All')
    
    def apply_filters(self):
        """Apply filters to the comparison table"""
        if not self.filter_active.get():
            # If filters are disabled, refresh the full table
            self.refresh_comparison_table()
            return
        
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            return
        
        try:
            # Read the comparison file
            df = pd.read_csv(comparison_file, index_col=0, comment='#')
            
            if df.empty:
                return
            
            # Get filter values
            filter_duration = self.filter_duration.get()
            filter_direction = self.filter_direction.get()
            filter_curve = self.filter_curve.get()
            
            # Get columns that match the filters
            data_columns = [col for col in df.columns if col != 'Units']
            filtered_columns = []
            
            for col in data_columns:
                params = self.extract_column_parameters(col)
                if params:
                    # Check if this column matches all active filters
                    if (filter_duration == 'All' or params['duration'] == filter_duration) and \
                       (filter_direction == 'All' or params['direction'] == filter_direction) and \
                       (filter_curve == 'All' or params['curve'] == filter_curve):
                        filtered_columns.append(col)
            
            # Create filtered dataframe
            if filtered_columns:
                if 'Units' in df.columns:
                    filtered_df = df[['Units'] + filtered_columns]
                else:
                    filtered_df = df[filtered_columns]
            else:
                # No columns match filters, show empty table
                filtered_df = pd.DataFrame()
            
            # Update the display with filtered data
            self._display_filtered_data(filtered_df)
            
            # Update terminal output
            if filtered_columns:
                self.add_terminal_output(f"Applied filters: Duration={filter_duration}, Direction={filter_direction}, Curve={filter_curve}")
                self.add_terminal_output(f"   Showing {len(filtered_columns)} column(s): {', '.join(filtered_columns)}")
            else:
                self.add_terminal_output(f"No columns match the current filters")
                
        except Exception as e:
            messagebox.showerror("Filter Error", f"Failed to apply filters: {str(e)}")
    
    def _display_filtered_data(self, df):
        """Display filtered data in the comparison tree"""
        # Clear existing data
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        if df.empty:
            self.comparison_tree.heading("Metric", text="No data matches filters")
            return
        
        # Configure columns - use headings only to eliminate tree structure
        columns = ['Metric'] + list(df.columns)
        self.comparison_tree['columns'] = columns  # Include all columns
        self.comparison_tree['show'] = 'headings'  # Only show headings, no tree structure
        
        # Configure the treeview to minimize indentation for the first column
        self.comparison_tree.configure(style="NoIndent.Treeview")
        
        # Set column headings and widths
        self.comparison_tree.heading("Metric", text="Metric", anchor=tk.W)
        
        # Calculate optimal width for Metric column (excluding Source_File)
        display_metrics = [metric for metric in df.index if metric != 'Source_File']
        if display_metrics:
            metric_width = max(
                len("Metric") * 8,
                max([len(str(metric)) for metric in display_metrics] + [len("Exported Date"), len("Exported Time")]) * 8 + 20
            )
        else:
            metric_width = len("Exported Date") * 8 + 20
            
        # Apply 30% reduction as requested
        metric_width = int(metric_width * 0.7)
        metric_width = min(metric_width, 175)  # 30% smaller cap
        self.comparison_tree.column("Metric", width=metric_width, anchor=tk.W, minwidth=metric_width, stretch=False)
        
        # Calculate optimal width for Units column
        units_width = 60
        if "Units" in df.columns and display_metrics:
            try:
                max_unit_length = max([len(str(df.loc[metric, "Units"])) for metric in display_metrics] + [len("Units"), 3])
                units_width = max(units_width, max_unit_length * 8 + 15)
            except:
                units_width = 80
                
            # Apply 30% reduction as requested
            units_width = int(units_width * 0.7)
            units_width = min(units_width, 84)  # 30% smaller cap
        else:
            # Apply 30% reduction to default as well
            units_width = int(units_width * 0.7)
        
        # Reorder columns to ensure Units is always second (after Metric)
        if "Units" in df.columns:
            data_columns = [col for col in df.columns if col != "Units"]
            ordered_columns = ["Units"] + data_columns  # Units first among df.columns, then data columns
        else:
            ordered_columns = list(df.columns)
        
        for col in ordered_columns:
            display_name = self._column_name_for_display(col)
            self.comparison_tree.heading(col, text=display_name, anchor=tk.W)
            if col == "Units":
                self.comparison_tree.column(col, width=units_width, anchor=tk.W, minwidth=units_width, stretch=False)
            else:
                self.comparison_tree.column(col, width=180, anchor=tk.W, minwidth=120, stretch=True)
        
        # Add frozen date and time rows
        date_values = ["Exported Date"]  # Start with metric name
        time_values = ["Exported Time"]  # Start with metric name
        for col in ordered_columns:  # Use the same column order as headers
            if col == "Units":
                date_values.append("-")
                time_values.append("-")
            else:
                if 'Source_File' in df.index:
                    source_file = df.loc['Source_File', col]
                    extracted_date, extracted_time = self._extract_timestamp_from_column(source_file)
                else:
                    extracted_date, extracted_time = self._extract_timestamp_from_column(col)
                date_values.append(extracted_date)
                time_values.append(extracted_time)
        
        # Insert date and time rows as normal rows
        self.comparison_tree.insert("", 0, values=date_values)
        self.comparison_tree.insert("", 1, values=time_values)
        
        # Add Ramp_Rate row first as a priority metric
        next_row = 2
        if 'Ramp_Rate' in df.index:
            formatted_values = ['Ramp_Rate']  # Start with metric name as first column
            for col in ordered_columns:  # Use the same column order as headers
                raw_value = df.loc['Ramp_Rate', col]
                if col == "Units":
                    formatted_values.append(str(raw_value))
                else:
                    formatted_values.append(self._format_for_display(raw_value, 'Ramp_Rate'))
            
            self.comparison_tree.insert("", next_row, values=formatted_values)
            next_row += 1
        
        # Add remaining data rows (excluding Source_File and Ramp_Rate)
        for metric in df.index:
            if metric in ['Source_File', 'Ramp_Rate']:
                continue
                
            formatted_values = [metric]  # Start with metric name as first column
            for col in ordered_columns:  # Use the same column order as headers
                raw_value = df.loc[metric, col]
                if col == "Units":
                    formatted_values.append(str(raw_value))
                else:
                    formatted_values.append(self._format_for_display(raw_value, metric))
            
            self.comparison_tree.insert("", tk.END, values=formatted_values)
        
        # Update column order listbox with filtered columns
        self.update_column_order_listbox(ordered_columns)
    
    def clear_filters(self):
        """Clear all filters and show all data"""
        self.filter_duration.set('All')
        self.filter_direction.set('All') 
        self.filter_curve.set('All')
        self.filter_active.set(False)
        self.refresh_comparison_table()
        self.add_terminal_output("Cleared all filters - showing all data")
    
    def show_context_menu(self, event):
        """Show context menu on right-click"""
        try:
            # Identify which column was clicked
            region = self.comparison_tree.identify_region(event.x, event.y)
            
            if region == "heading":
                # Get the column identifier - identify_column only takes x coordinate
                column = self.comparison_tree.identify_column(event.x)
                
                # Convert column number to column name
                if column == "#0":
                    # Can't remove the metric column
                    return
                
                # Get column name from the column number
                column_names = self.comparison_tree['columns']
                if column_names and column:
                    try:
                        col_index = int(column.replace('#', '')) - 1
                        if 0 <= col_index < len(column_names):
                            self.selected_column = column_names[col_index]
                            
                            # Don't allow removing the Units column
                            if self.selected_column == "Units":
                                return
                            
                            # Show context menu
                            self.context_menu.post(event.x_root, event.y_root)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            # Silently handle any errors in context menu
            print(f"Context menu error: {e}")
    
    def check_file_status(self):
        """Check if the comparison file is accessible and provide status info"""
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            messagebox.showinfo("File Status", "Comparison file does not exist yet.\nRun an analysis to create it.")
            return
        
        # Try to open the file for writing to check if it's locked
        try:
            # Test write access
            with open(comparison_file, 'r+') as f:
                pass  # Just test if we can open for writing
            
            # Get file info
            import stat
            file_stats = os.stat(comparison_file)
            file_size = file_stats.st_size
            modified_time = pd.Timestamp.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Count columns
            try:
                df = pd.read_csv(comparison_file, index_col=0, comment='#')
                num_columns = len([col for col in df.columns if col != 'Units'])
                num_metrics = len(df.index)
                
                status_msg = f"File Status: ACCESSIBLE\n\n" \
                           f"File: {os.path.basename(comparison_file)}\n" \
                           f"Size: {file_size:,} bytes\n" \
                           f"Modified: {modified_time}\n" \
                           f"Test configurations: {num_columns}\n" \
                           f"Metrics tracked: {num_metrics}\n\n" \
                           f"The file can be modified safely."
            except:
                status_msg = f"File Status: ACCESSIBLE\n\n" \
                           f"File: {os.path.basename(comparison_file)}\n" \
                           f"Size: {file_size:,} bytes\n" \
                           f"Modified: {modified_time}\n\n" \
                           f"The file can be modified safely."
            
            messagebox.showinfo("File Status", status_msg)
            
        except PermissionError:
            status_msg = f"File Status: LOCKED\n\n" \
                       f"File: {os.path.basename(comparison_file)}\n" \
                       f"Location: {comparison_file}\n\n" \
                       f"The file is currently open in another program\n" \
                       f"(likely Excel or a text editor).\n\n" \
                       f"To fix this:\n" \
                       f"• Close Excel or other programs using this file\n" \
                       f"• Save and close any text editors with this file open\n" \
                       f"• Wait a moment and try again"
            
            messagebox.showwarning("File Status", status_msg)
            
        except Exception as e:
            messagebox.showerror("File Status", f"Error checking file status:\n{str(e)}")
    
    def remove_selected_column(self):
        """Remove the selected column from both display and CSV file"""
        if not hasattr(self, 'selected_column') or not self.selected_column:
            messagebox.showwarning("No Selection", "Please right-click on a column header to select it for removal.")
            return
        
        # Confirm deletion
        result = messagebox.askyesno("Confirm Deletion", 
                                   f"Are you sure you want to remove the column '{self.selected_column}'?\n\n"
                                   f"This will permanently delete it from the CSV file.")
        
        if not result:
            return
        
        comparison_file = os.path.join(os.getcwd(), ResultsComparison.COMPARISON_FILE)
        
        if not os.path.exists(comparison_file):
            messagebox.showerror("Error", "Comparison file not found.")
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(comparison_file, index_col=0, comment='#')
            
            # Check if column exists
            if self.selected_column not in df.columns:
                messagebox.showerror("Error", f"Column '{self.selected_column}' not found in the data.")
                return
            
            # Remove the column
            df = df.drop(columns=[self.selected_column])
            
            # Save back to file with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df.to_csv(comparison_file)
                    break  # Success, exit retry loop
                except PermissionError:
                    if attempt < max_retries - 1:
                        # Ask user to close the file and retry
                        retry_result = messagebox.askretrycancel(
                            "File Access Error", 
                            f"Cannot access the comparison file. It may be open in Excel or another program.\n\n"
                            f"Please close the file and click 'Retry', or click 'Cancel' to abort.\n\n"
                            f"File: {os.path.basename(comparison_file)}\n"
                            f"Attempt: {attempt + 1}/{max_retries}"
                        )
                        if not retry_result:
                            return  # User cancelled
                    else:
                        # Final attempt failed
                        messagebox.showerror(
                            "File Access Error", 
                            f"Unable to modify the comparison file after {max_retries} attempts.\n\n"
                            f"The file may be open in Excel or another program.\n"
                            f"Please close all applications using this file and try again.\n\n"
                            f"File: {comparison_file}"
                        )
                        return
            
            # Add metadata header comment back
            with open(comparison_file, 'r') as f:
                content = f.read()
            
            header_comment = f"""# Ramp Analysis Results Comparison
# Generated by Dynamic Reactor Ramp Analysis Tool
# Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
# 
# Each column represents one ramp test configuration: Duration_Direction_CurveType
# Each row represents a key metric from the analysis
# Units column shows the measurement units for each metric
# Same ramp configurations will overwrite previous results
#
"""
            
            with open(comparison_file, 'w') as f:
                f.write(header_comment + content)
            
            # Refresh the display
            self.refresh_comparison_table()
            
            # Show success message
            messagebox.showinfo("Success", f"Column '{self.selected_column}' has been removed successfully.")
            self.add_terminal_output(f"Removed column '{self.selected_column}' from comparison table")
            
            # Clear selection
            self.selected_column = None
            
        except PermissionError as pe:
            messagebox.showerror(
                "Permission Error", 
                f"Cannot access the comparison file. Please ensure:\n\n"
                f"• The file is not open in Excel or another program\n"
                f"• You have write permissions to the file\n"
                f"• The file is not read-only\n\n"
                f"File: {comparison_file}\n"
                f"Error: {str(pe)}"
            )
            print(f"Permission error removing column: {pe}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove column: {str(e)}")
            print(f"Error removing column: {e}")
    
    def select_file(self):
        """File selection dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Aspen Plus CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if file_path:
            filename = os.path.basename(file_path)
            self.selected_file.set(filename)
            self.file_path = file_path
            
            # Extract timestamp from filename for use in comparison table
            self.current_file_timestamp = self._extract_timestamp_from_column(filename)
            
            self.analyze_button.config(state="normal")
            self.status_label.config(text=f"File selected: {filename}")
            
            # Show extracted timestamp in status if found
            if self.current_file_timestamp and self.current_file_timestamp[1] != "N/A":
                self.status_label.config(text=f"File selected: {filename} | Timestamp: {self.current_file_timestamp[0]} {self.current_file_timestamp[1]}")
        
    def select_all_plots(self):
        """Select all plot options"""
        for var in self.plot_vars.values():
            var.set(True)
    
    def select_no_plots(self):
        """Deselect all plot options"""
        for var in self.plot_vars.values():
            var.set(False)
    
    def get_analysis_options(self) -> AnalysisOptions:
        """Get analysis options from GUI"""
        time_limit = None
        if self.time_limit_var.get().strip():
            try:
                time_limit = float(self.time_limit_var.get())
            except ValueError:
                pass
        
        return AnalysisOptions(
            temperature_response=self.plot_vars['temperature_response'].get(),
            stability_analysis=self.plot_vars['stability_analysis'].get(),
            spatial_gradients=self.plot_vars['spatial_gradients'].get(),
            heat_transfer_3d=self.plot_vars['heat_transfer_3d'].get(),
            time_limit=time_limit
        )
    
    def save_last_results(self):
        """Save the last analysis results"""
        if self.last_data_package is None:
            messagebox.showwarning("Warning", "No analysis results to save. Please run an analysis first.")
            return
        
        try:
            analyzer = DynamicRampAnalyzer()
            timestamp = analyzer.save_analysis_results(self.last_data_package)
            messagebox.showinfo("Success", f"Analysis results saved with timestamp: {timestamp}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def run_analysis(self):
        """Run the analysis in a separate thread"""
        if not hasattr(self, 'file_path'):
            messagebox.showerror("Error", "Please select a file first.")
            return
        
        # Check if any plots are selected
        if not any(var.get() for var in self.plot_vars.values()):
            messagebox.showwarning("Warning", "Please select at least one plot type.")
            return
        
        self.analyze_button.config(state="disabled")
        self.progress.start()
        self.status_label.config(text="Running analysis...")
        
        # Run analysis in separate thread to prevent GUI freezing
        analysis_thread = threading.Thread(target=self._run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _show_plot_choice_dialog(self, generated_plots: List[Tuple[str, plt.Figure]]) -> str:
        """Show dialog asking user what to do with the generated plots"""
        
        # Create custom dialog - same size as main window
        dialog = tk.Toplevel(self.root)
        dialog.title("Analysis Complete!")
        dialog.geometry("400x600")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()  # Make it modal
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Result variable
        result = tk.StringVar()
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Success message - centered
        success_label = ttk.Label(main_frame, 
                                 text="Analysis Completed Successfully!", 
                                 font=('Arial', 14, 'bold'),
                                 anchor='center')
        success_label.pack(pady=(0, 20))
        
        # Plot count message - left aligned
        plot_count_label = ttk.Label(main_frame,
                                    text=f"Generated {len(generated_plots)} plot set(s):",
                                    font=('Arial', 11),
                                    anchor='w')
        plot_count_label.pack(anchor=tk.W, pady=(0, 8))
        
        # List of generated plots - left aligned
        for plot_name, _ in generated_plots:
            plot_item = ttk.Label(main_frame, text=f"  • {plot_name}", 
                                 font=('Arial', 10), anchor='w')
            plot_item.pack(anchor=tk.W, pady=2)
        
        # Question - left aligned
        question_label = ttk.Label(main_frame,
                                  text="\nWhat would you like to do with the plots?",
                                  font=('Arial', 12, 'bold'),
                                  anchor='w')
        question_label.pack(anchor=tk.W, pady=(25, 20))
        
        # Option buttons
        def set_result(value):
            result.set(value)
            dialog.destroy()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        # Option 1: View and Save
        btn1 = ttk.Button(button_frame, 
                         text="View & Save", 
                         width=30,
                         command=lambda: set_result("view_and_save"))
        btn1.pack(pady=3, fill=tk.X)
        
        # Option 2: View Only (No Save)
        btn2 = ttk.Button(button_frame, 
                         text="View Only", 
                         width=30,
                         command=lambda: set_result("view_only"))
        btn2.pack(pady=3, fill=tk.X)
        
        # Option 3: Save Only (No View)
        btn3 = ttk.Button(button_frame, 
                         text="Save Only", 
                         width=30,
                         command=lambda: set_result("save_only"))
        btn3.pack(pady=3, fill=tk.X)
        
        # Option 4: Neither
        btn4 = ttk.Button(button_frame, 
                         text="Nothing", 
                         width=30,
                         command=lambda: set_result("neither"))
        btn4.pack(pady=3, fill=tk.X)
        
        # Default option on Enter key
        dialog.bind('<Return>', lambda e: set_result("view_and_save"))
        dialog.focus_set()
        
        # Wait for user choice
        dialog.wait_window()
        
        return result.get() or "view_and_save"  # Default if dialog closed
    
    def _save_plots_to_files(self, generated_plots: List[Tuple[str, plt.Figure]], base_filename: str = None) -> str:
        """Save all generated plots to image files"""
        if not base_filename:
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), f"{base_filename}_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each plot
        saved_files = []
        for plot_name, fig in generated_plots:
            # Clean plot name for filename
            clean_name = plot_name.replace(" ", "_").replace("/", "_").lower()
            filename = f"{base_filename}_{clean_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                saved_files.append(filename)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
        
        print(f"\nPlots saved to: {output_dir}")
        return output_dir

    def _run_analysis_thread(self):
        """Run analysis in separate thread"""
        # Clear terminal output for new analysis
        self.root.after(0, self.clear_terminal)
        
        try:
            # Capture stdout to show in terminal panel
            captured_output = io.StringIO()
            
            with contextlib.redirect_stdout(captured_output):
                options = self.get_analysis_options()
                analyzer = DynamicRampAnalyzer()
                
                # Update status and add to terminal
                self.root.after(0, lambda: self.status_label.config(text="Loading data..."))
                self.root.after(0, lambda: self.add_terminal_output("Starting analysis..."))
                self.root.after(0, lambda: self.add_terminal_output(""))
                self.root.after(0, lambda: self.add_terminal_output(f"Loading file: {os.path.basename(self.file_path)}"))
                
                # Load data first to store for potential saving
                data_package = DataLoader.load_and_parse_aspen_data(self.file_path)
                if data_package:
                    self.last_data_package = data_package
                    self.root.after(0, lambda: self.save_button.config(state="normal"))
                    self.root.after(0, lambda: self.add_terminal_output("Data loaded successfully"))
                else:
                    self.root.after(0, lambda: self.add_terminal_output("Failed to load data"))
                
                self.root.after(0, lambda: self.add_terminal_output(""))
                # Do data processing only (no plotting) in background thread
                self.root.after(0, lambda: self.add_terminal_output("Processing data..."))
                success = analyzer.run_data_processing_only(self.file_path, options)
                
                # Store processed data for plotting in main thread
                if success and hasattr(analyzer, 'processed_data'):
                    self.processed_data = analyzer.processed_data
                    self.analysis_options = options
                    self.root.after(0, lambda: self.add_terminal_output("Data processing completed"))
                    self.root.after(0, lambda: self.add_terminal_output(""))
                    self.root.after(0, self._create_and_display_plots)
                else:
                    self.root.after(0, lambda: self.add_terminal_output("Data processing failed"))
                
                # Auto-save if option is selected
                if success and self.save_results_var.get() and data_package:
                    self.root.after(0, lambda: self.status_label.config(text="Saving results..."))
                    self.root.after(0, lambda: self.add_terminal_output("Saving results..."))
                    timestamp = analyzer.save_analysis_results(data_package)
                    save_msg = f" Results saved with timestamp: {timestamp}"
                    self.root.after(0, lambda: self.add_terminal_output(f"Results saved with timestamp: {timestamp}"))
                else:
                    save_msg = ""
            
            # Add captured output to terminal
            output_text = captured_output.getvalue()
            if output_text.strip():
                for line in output_text.strip().split('\n'):
                    if line.strip():
                        self.root.after(0, lambda l=line: self.add_terminal_output(l))
            
            if success:
                self.root.after(0, lambda: self.status_label.config(text=f"Analysis completed successfully!{save_msg}"))
                self.root.after(0, lambda: self.add_terminal_output(""))
                self.root.after(0, lambda: self.add_terminal_output("Analysis completed successfully!"))
                
                # If results comparison is enabled, update comparison data
                if hasattr(self, 'processed_data') and self.processed_data:
                    self.root.after(0, lambda: self.add_terminal_output(""))
                    self.root.after(0, lambda: self.add_terminal_output("Updating results comparison..."))
                    try:
                        # Extract metrics from processed data
                        data_dict = {
                            'time_vector': self.processed_data['time_vector'],
                            'variables': {'T_cat (°C)': self.processed_data['catalyst_temp_matrix']},
                            'length_vector': self.processed_data['length_vector'],
                            'dimensions': {'n_time': len(self.processed_data['time_vector']), 
                                         'm_length': len(self.processed_data['length_vector'])},
                            'file_path': self.file_path
                        }
                        
                        # Add heat transfer if available
                        if 'heat_transfer_matrix' in self.processed_data:
                            data_dict['variables']['Heat Transfer with coolant (kW/m2)'] = self.processed_data['heat_transfer_matrix']
                        
                        # Create analysis engine and extract metrics
                        from analysis_engine import AnalysisEngine
                        engine = AnalysisEngine()
                        engine.data_package = data_dict
                        engine.ramp_params = self.processed_data['ramp_params']
                        engine.steady_state_time = self.processed_data.get('steady_state_time')
                        engine.stability_metrics = self.processed_data.get('stability_metrics', {'threshold': 0.05, 'min_rms_rate': 0.0})
                        
                        metrics = engine.extract_key_metrics()
                        
                        ResultsComparison.update_comparison_file(metrics)
                        self.root.after(0, lambda: self.add_terminal_output("Results comparison updated successfully"))
                    except Exception as e:
                        self.root.after(0, lambda: self.add_terminal_output(f"Failed to update comparison: {str(e)}"))
                        print(f"Comparison update error: {e}")  # For debugging
            else:
                self.root.after(0, lambda: self.status_label.config(text="Analysis failed. Check terminal for errors."))
                self.root.after(0, lambda: self.add_terminal_output("Analysis failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Analysis failed. Check terminal for error details."))
                
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text="Analysis failed."))
            self.root.after(0, lambda: self.add_terminal_output(f"ERROR: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            print(f"Analysis error: {e}")
        finally:
            self.root.after(0, self._analysis_finished)
    
    def _create_and_display_plots(self):
        """Create and display plots in main thread using processed data"""
        try:
            if hasattr(self, 'processed_data') and hasattr(self, 'analysis_options'):
                self.status_label.config(text="Creating plots...")
                self.add_terminal_output("Creating plots...")
                self.add_terminal_output("")
                
                data = self.processed_data
                options = self.analysis_options
                
                generated_plots = []
                
                if options.temperature_response:
                    self.add_terminal_output("Generating temperature response plots...")
                    try:
                        fig = PlotGen.create_temperature_response_plots(
                            data['time_vector'], data['catalyst_temp_matrix'], data['length_vector'],
                            data['ramp_params'], data['steady_state_time'], self.file_path, options.time_limit
                        )
                        generated_plots.append(("Temperature Response", fig))
                        self.add_terminal_output("   Temperature response plots completed")
                    except Exception as e:
                        self.add_terminal_output(f"   Error generating temperature response plots: {e}")
                
                if options.stability_analysis and data.get('stability_metrics'):
                    self.add_terminal_output("Generating stability analysis plots...")
                    try:
                        fig = PlotGen.create_stability_analysis_plots(
                            data['time_vector'], data['stability_metrics'], data['ramp_params'],
                            data['steady_state_time'], self.file_path, options.time_limit
                        )
                        generated_plots.append(("Stability Analysis", fig))
                        self.add_terminal_output("   Stability analysis plots completed")
                    except Exception as e:
                        self.add_terminal_output(f"   Error generating stability plots: {e}")
                
                if options.spatial_gradients:
                    self.add_terminal_output("Generating spatial gradient analysis plots...")
                    try:
                        fig = PlotGen.create_spatial_gradient_plots(
                            data['time_vector'], data['catalyst_temp_matrix'], data['length_vector'],
                            data['ramp_params'], self.file_path, options.time_limit
                        )
                        generated_plots.append(("Spatial Gradients", fig))
                        self.add_terminal_output("   Spatial gradient plots completed")
                    except Exception as e:
                        self.add_terminal_output(f"   Error generating spatial gradient plots: {e}")
                
                if options.heat_transfer_3d and data.get('heat_transfer_matrix') is not None:
                    self.add_terminal_output("Generating 3D heat transfer plots...")
                    try:
                        fig = PlotGen.create_3d_heat_transfer_plots(
                            data['time_vector'], data['heat_transfer_matrix'], data['length_vector'],
                            data['ramp_params'], data['steady_state_time'], self.file_path, options.time_limit
                        )
                        generated_plots.append(("3D Heat Transfer", fig))
                        self.add_terminal_output("   3D heat transfer plots completed")
                    except Exception as e:
                        self.add_terminal_output(f"   Error generating 3D heat transfer plots: {e}")
                
                # Store generated plots
                self.generated_plots = generated_plots
                
                # Show user choice dialog and handle accordingly
                if generated_plots:
                    self.add_terminal_output("")
                    self.add_terminal_output(f"Plot generation completed! Generated {len(generated_plots)} plot set(s):")
                    for plot_name, _ in generated_plots:
                        self.add_terminal_output(f"   {plot_name}")
                    self.add_terminal_output("")
                    
                    # Show choice dialog
                    user_choice = self._show_plot_choice_dialog(generated_plots)
                    
                    # Handle user choice
                    if user_choice == "view_and_save":
                        self.add_terminal_output("Displaying plots...")
                        plt.show()
                        self.add_terminal_output("Saving plot images...")
                        saved_dir = self._save_plots_to_files(generated_plots)
                        messagebox.showinfo("Success", 
                                          f"Analysis completed!\n\n"
                                          f"• Plots displayed\n"
                                          f"• Images saved to: {os.path.basename(saved_dir)}")
                        
                    elif user_choice == "view_only":
                        self.add_terminal_output("Displaying plots...")
                        plt.show()
                        messagebox.showinfo("Success", "Analysis completed! Plots displayed.")
                        
                    elif user_choice == "save_only":
                        self.add_terminal_output("Saving plot images...")
                        saved_dir = self._save_plots_to_files(generated_plots)
                        # Close all figures to free memory
                        for _, fig in generated_plots:
                            plt.close(fig)
                        messagebox.showinfo("Success", 
                                          f"Analysis completed!\n\n"
                                          f"• Plot images saved to: {os.path.basename(saved_dir)}")
                        
                    elif user_choice == "neither":
                        self.add_terminal_output("Analysis completed - plots generated but not displayed or saved.")
                        # Close all figures to free memory
                        for _, fig in generated_plots:
                            plt.close(fig)
                        messagebox.showinfo("Success", "Analysis completed!")
                        
                else:
                    self.add_terminal_output("No plots were generated")
                    messagebox.showwarning("Warning", "No plots were generated.")
                    
        except Exception as e:
            print(f"Error creating plots: {e}")
            messagebox.showerror("Error", f"Error creating plots: {str(e)}")

    def _display_plots(self):
        """Display plots in main thread"""
        if hasattr(self, 'generated_plots') and self.generated_plots:
            print(f"\nAnalysis completed! Generated {len(self.generated_plots)} plot set(s):")
            for plot_name, _ in self.generated_plots:
                print(f"  {plot_name}")
            
            print("\nDisplaying plots...")
            # Use plt.show() to display all figures
            plt.show()
    
    def _analysis_finished(self):
        """Clean up after analysis"""
        self.progress.stop()
        self.analyze_button.config(state="normal")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

class DynamicRampAnalyzer:
    """Main analyzer class that coordinates all analysis components"""
    
    def __init__(self):
        # Update matplotlib settings if ConfigManager is available
        if ConfigMgr:
            ConfigMgr.update_matplotlib_settings()
            self.config = ConfigMgr.get_config()
        else:
            self.config = {}
    
    def run_data_processing_only(self, file_path: str, options: AnalysisOptions) -> bool:
        """Run data processing only (no plotting) for threading"""
        try:
            print("="*60)
            print("DYNAMIC REACTOR RAMP ANALYSIS")
            print("="*60)
            
            # Load data
            print("Loading and parsing data...")
            data_package = DataLoader.load_and_parse_aspen_data(file_path)
            if data_package is None:
                print("❌ Failed to load data")
                return False
            
            print("✓ Data loaded successfully")
            
            # Parse ramp parameters
            ramp_params = DataLoader.parse_ramp_parameters_from_filename(file_path)
            
            if ramp_params.duration and ramp_params.direction and ramp_params.curve_shape:
                print(f"✓ Ramp parameters detected: {ramp_params.duration}min {ramp_params.direction}-{ramp_params.curve_shape}")
            else:
                print("⚠ Using fallback ramp detection")
            
            # Extract data
            time_vector = data_package['time_vector']
            catalyst_temp_matrix = data_package['variables']['T_cat (°C)']
            length_vector = data_package['length_vector']
            
            # Detect steady state
            print("\nDetecting steady state conditions...")
            search_start_time = ramp_params.end_time if ramp_params.end_time else None
            steady_state_config = self.config['steady_state']
            
            steady_state_time, stability_metrics = SteadyStateDetector.detect_steady_state(
                time_vector, catalyst_temp_matrix, 
                threshold=steady_state_config['threshold'],
                min_duration=steady_state_config['min_duration'],
                search_start_time=search_start_time
            )
            
            if steady_state_time:
                print(f"✓ Steady state detected at t = {steady_state_time:.1f} min")
            else:
                print("⚠ No steady state detected in analysis period")
            
            # Generate analysis report
            if AnalysisRep:
                AnalysisRep.print_analysis_summary(
                    data_package, ramp_params, steady_state_time, stability_metrics
                )
            
            # Update results comparison file
            print("\nUpdating results comparison file...")
            try:
                # Create analysis engine and extract metrics  
                from analysis_engine import AnalysisEngine
                engine = AnalysisEngine()
                engine.data_package = data_package
                engine.ramp_params = ramp_params
                engine.steady_state_time = steady_state_time
                engine.stability_metrics = stability_metrics
                
                metrics = engine.extract_key_metrics()
                comparison_file = ResultsComparison.update_comparison_file(metrics)
                print(f"Results comparison file updated: {os.path.basename(comparison_file)}")
            except Exception as e:
                print(f"Warning: Could not update results comparison file: {e}")
            
            # Store processed data for main thread plotting
            self.processed_data = {
                'time_vector': time_vector,
                'catalyst_temp_matrix': catalyst_temp_matrix,
                'length_vector': length_vector,
                'ramp_params': ramp_params,
                'steady_state_time': steady_state_time,
                'stability_metrics': stability_metrics,
                'heat_transfer_matrix': data_package['variables'].get('Heat Transfer with coolant (kW/m2)')
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Data processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_analysis(self, file_path: str, options: AnalysisOptions) -> bool:
        """Run complete analysis based on options"""
        success = self.run_analysis_no_display(file_path, options)
        
        # Show results if successful
        if success and hasattr(self, 'generated_plots') and self.generated_plots:
            print(f"\nAnalysis completed! Generated {len(self.generated_plots)} plot set(s):")
            for plot_name, _ in self.generated_plots:
                print(f"  {plot_name}")
            
            print("\nDisplaying plots...")
            # Display all figures
            plt.show()
        
        return success
    
    def run_analysis_no_display(self, file_path: str, options: AnalysisOptions) -> bool:
        """Run complete analysis based on options but don't display plots (for threading)"""
        try:
            print("="*60)
            print("DYNAMIC REACTOR RAMP ANALYSIS")
            print("="*60)
            
            # Load data
            print("Loading and parsing data...")
            data_package = DataLoader.load_and_parse_aspen_data(file_path)
            if data_package is None:
                print("❌ Failed to load data")
                return False
            
            print("✓ Data loaded successfully")
            
            # Parse ramp parameters
            ramp_params = DataLoader.parse_ramp_parameters_from_filename(file_path)
            
            if ramp_params.duration and ramp_params.direction and ramp_params.curve_shape:
                print(f"✓ Ramp parameters detected: {ramp_params.duration}min {ramp_params.direction}-{ramp_params.curve_shape}")
            else:
                print("⚠ Using fallback ramp detection")
            
            # Extract data
            time_vector = data_package['time_vector']
            catalyst_temp_matrix = data_package['variables']['T_cat (°C)']
            length_vector = data_package['length_vector']
            
            # Detect steady state
            print("\nDetecting steady state conditions...")
            search_start_time = ramp_params.end_time if ramp_params.end_time else None
            steady_state_config = self.config['steady_state']
            
            steady_state_time, stability_metrics = SteadyStateDetector.detect_steady_state(
                time_vector, catalyst_temp_matrix, 
                threshold=steady_state_config['threshold'],
                min_duration=steady_state_config['min_duration'],
                search_start_time=search_start_time
            )
            
            if steady_state_time:
                print(f"✓ Steady state detected at t = {steady_state_time:.1f} min")
            else:
                print("⚠ No steady state detected in analysis period")
            
            # Generate analysis report
            if AnalysisRep:
                AnalysisRep.print_analysis_summary(
                    data_package, ramp_params, steady_state_time, stability_metrics
                )
            
            # Generate selected plots
            print("\n" + "="*40)
            print("GENERATING PLOTS")
            print("="*40)
            
            self.generated_plots = []
            
            if options.temperature_response:
                print("📊 Generating temperature response plots...")
                try:
                    fig = PlotGen.create_temperature_response_plots(
                        time_vector, catalyst_temp_matrix, length_vector,
                        ramp_params, steady_state_time, file_path, options.time_limit
                    )
                    self.generated_plots.append(("Temperature Response", fig))
                    print("✓ Temperature response plots completed")
                except Exception as e:
                    print(f"❌ Error generating temperature response plots: {e}")
            
            if options.stability_analysis and stability_metrics:
                print("📊 Generating stability analysis plots...")
                try:
                    fig = PlotGen.create_stability_analysis_plots(
                        time_vector, stability_metrics, ramp_params,
                        steady_state_time, file_path, options.time_limit
                    )
                    self.generated_plots.append(("Stability Analysis", fig))
                    print("✓ Stability analysis plots completed")
                except Exception as e:
                    print(f"❌ Error generating stability plots: {e}")
            
            if options.spatial_gradients:
                print("📊 Generating spatial gradient analysis plots...")
                try:
                    fig = PlotGen.create_spatial_gradient_plots(
                        time_vector, catalyst_temp_matrix, length_vector,
                        ramp_params, file_path, options.time_limit
                    )
                    self.generated_plots.append(("Spatial Gradients", fig))
                    print("✓ Spatial gradient plots completed")
                except Exception as e:
                    print(f"❌ Error generating spatial gradient plots: {e}")
            
            if options.heat_transfer_3d:
                print("📊 Generating 3D heat transfer plots...")
                try:
                    heat_transfer_matrix = data_package['variables'].get('Heat Transfer with coolant (kW/m2)')
                    if heat_transfer_matrix is not None:
                        fig = PlotGen.create_3d_heat_transfer_plots(
                            time_vector, heat_transfer_matrix, length_vector,
                            ramp_params, steady_state_time, file_path, options.time_limit
                        )
                        self.generated_plots.append(("3D Heat Transfer", fig))
                        print("✓ 3D heat transfer plots completed")
                    else:
                        print("⚠ Heat transfer data not available for 3D plotting")
                except Exception as e:
                    print(f"❌ Error generating 3D heat transfer plots: {e}")
            
            # Check if any plots were generated
            if self.generated_plots:
                return True
            else:
                print("⚠ No plots were generated")
                return False
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_analysis_results(self, data_package: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save analysis results to files"""
        if DataExp:
            return DataExp.save_data_structure(data_package, output_dir)
        else:
            print("DataExporter not available - cannot save results")
            return ""

def main():
    """Main entry point"""
    app = AnalysisGUI()
    app.run()

if __name__ == "__main__":
    main()
