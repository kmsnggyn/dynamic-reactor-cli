"""
Plot Generation Module for Dynamic Reactor Ramp Analysis
=======================================================

Handles all plot generation functionality for the analysis results.
Separated from main analysis for modularity and easier customization.

Author: Seonggyun Kim (seonggyun.kim@outlook.com)
Date: August 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from typing import Optional, Dict, Any, List, Tuple, Union
from analysis_engine import RampParameters

# Type alias for cleaner code
Figure = matplotlib.figure.Figure

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Plot Configuration Constants
DEFAULT_TIME_LIMIT_FOR_DISPLAY = 60.0  # minutes
DEFAULT_PLOT_DPI = 100  # dots per inch
DEFAULT_FIGSIZE_LARGE = (20, 12)  # inches
DEFAULT_FIGSIZE_MEDIUM = (16, 12)  # inches
DEFAULT_FIGSIZE_SMALL = (12, 8)  # inches
DEFAULT_LINEWIDTH = 2.5  # line width for plots
DEFAULT_ALPHA_OVERLAY = 0.2  # transparency for overlays
DEFAULT_ALPHA_LEGEND = 0.9  # transparency for legends
DEFAULT_GRID_ALPHA = 0.3  # transparency for grid lines
MAX_MARKERS_DISPLAY = 20  # maximum markers to show on line plots

def safe_tight_layout(rect=None) -> None:
    """
    Apply tight_layout safely, handling warnings gracefully.
    
    Args:
        rect: The rectangle area to adjust for (left, bottom, right, top)
    """
    try:
        if rect is not None:
            plt.tight_layout(rect=rect)
        else:
            plt.tight_layout()
    except (Warning, UserWarning):
        # If tight_layout fails, apply basic adjustments
        if rect is not None:
            plt.subplots_adjust(bottom=rect[1], top=rect[3])
        else:
            plt.subplots_adjust(top=0.95, bottom=0.05)

class PlotGenerator:
    """
    Handles all plot generation functionality for dynamic reactor analysis.
    
    This class provides static methods for creating various types of plots
    and visualizations for reactor ramp test analysis. All methods are
    designed to be independent and reusable across different analysis workflows.
    
    Features:
    - Temperature response analysis plots
    - Stability analysis visualizations  
    - Spatial gradient analysis
    - 3D heat transfer visualization
    - Temperature difference analysis
    
    All plot methods return matplotlib Figure objects that can be displayed
    or saved independently.
    """
    
    @staticmethod
    def _set_window_title_safe(fig: matplotlib.figure.Figure, title: str) -> None:
        """Safely set window title, handling cases where it's not available."""
        try:
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(title)
        except (AttributeError, TypeError):
            pass  # Window title setting not available
    
    @staticmethod
    def _safe_axvspan(ax, start_time: Optional[float], end_time: Optional[float], **kwargs) -> None:
        """Safely add axvspan, handling None values."""
        if start_time is not None and end_time is not None:
            ax.axvspan(start_time, end_time, **kwargs)
    
    @staticmethod
    def _safe_ramp_duration_check(ramp_params: RampParameters) -> bool:
        """Check if ramp parameters have valid duration."""
        return (hasattr(ramp_params, 'duration') and ramp_params.duration is not None and 
                hasattr(ramp_params, 'start_time') and ramp_params.start_time is not None and
                hasattr(ramp_params, 'end_time') and ramp_params.end_time is not None)
    
    @staticmethod
    def create_temperature_response_plots(time_vector: np.ndarray,
                                        catalyst_temp_matrix: np.ndarray,
                                        length_vector: np.ndarray,
                                        ramp_params: RampParameters,
                                        steady_state_time: Optional[float],
                                        file_path: str,
                                        time_limit: Optional[float] = None) -> Figure:
        """Create main temperature response analysis plots with vertical layout"""
        
        # Filter data
        if time_limit is None:
            time_limit = time_vector.max()
            
        time_mask = time_vector <= time_limit
        time_filtered = time_vector[time_mask]
        temp_filtered = catalyst_temp_matrix[time_mask, :]
        
        # Apply plot time limit - cut where data ends
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= DEFAULT_TIME_LIMIT_FOR_DISPLAY:
            plot_time_limit = min(DEFAULT_TIME_LIMIT_FOR_DISPLAY, time_limit if time_limit is not None else DEFAULT_TIME_LIMIT_FOR_DISPLAY)
        
        plot_time_mask = time_filtered <= plot_time_limit
        time_plot = time_filtered[plot_time_mask]
        temp_plot = temp_filtered[plot_time_mask, :]
        
        # Calculate derivatives
        dt = time_plot[1] - time_plot[0] if len(time_plot) > 1 else 0.05
        temp_derivatives = np.gradient(temp_plot, dt, axis=0)
        
        # Focus on relevant rates based on ramp direction
        if ramp_params.is_ramp_up:
            relevant_rates = np.where(temp_derivatives > 0, temp_derivatives, 0)
            max_relevant_rates = np.nanmax(relevant_rates, axis=1)
            global_max_rate = np.nanmax(relevant_rates)
            rate_label = r"$dT_{\mathrm{cat}}/dt$"
            rate_color = "red"
        elif ramp_params.is_ramp_down:
            relevant_rates = np.where(temp_derivatives < 0, temp_derivatives, 0)
            max_relevant_rates = np.nanmin(relevant_rates, axis=1)
            global_max_rate = np.nanmin(relevant_rates)
            rate_label = r"$dT_{\mathrm{cat}}/dt$"
            rate_color = "blue"
        else:
            relevant_rates = np.abs(temp_derivatives)
            max_relevant_rates = np.nanmax(relevant_rates, axis=1)
            global_max_rate = np.nanmax(relevant_rates)
            rate_label = r"$|dT_{\mathrm{cat}}/dt|$"
            rate_color = "purple"
        
        # Find maximum position
        if ramp_params.is_ramp_down:
            global_max_pos = np.unravel_index(np.nanargmin(relevant_rates), relevant_rates.shape)
        else:
            global_max_pos = np.unravel_index(np.nanargmax(relevant_rates), relevant_rates.shape)
        
        max_time_idx, max_pos_idx = global_max_pos
        
        # Create vertical layout plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Set window title safely (might not be available in all environments)
        try:
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(f"Temperature Response Analysis - {os.path.basename(file_path)}")
        except (AttributeError, TypeError):
            pass  # Window title setting not available
        
        # Create consolidated info banner at the top
        direction = "up" if ramp_params.is_ramp_up else "down"
        
        # Extract curve type more safely
        try:
            if hasattr(ramp_params, 'curve_type') and ramp_params.curve_type:
                curve_type = ramp_params.curve_type.lower()
            else:
                # Fallback: extract from analysis_title or default
                curve_type = "linear"  # default fallback
                if hasattr(ramp_params, 'analysis_title') and ramp_params.analysis_title:
                    if "sinusoidal" in ramp_params.analysis_title.lower():
                        curve_type = "sinusoidal"
                    elif "linear" in ramp_params.analysis_title.lower():
                        curve_type = "linear"
        except:
            curve_type = "linear"
        
        info_text = f"Ramp-{direction} test | {ramp_params.duration:.0f} min | {curve_type}"
        fig.suptitle(info_text, fontsize=14, fontweight='bold', y=0.95)  # Higher position
        
        # Plot 1: Catalyst Temperature (Top)
        temp_at_max_pos = temp_plot[:, max_pos_idx]
        
        ax1.plot(time_plot, temp_at_max_pos, 'b-', linewidth=2.5,
                label=rf'$T_{{\mathrm{{cat}}}}$ at L = {length_vector[max_pos_idx]:.2f} m')
        
        # Add ramp period shading
        if ramp_params.duration:
            ax1.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', label='Ramp period')
        
        # Add steady state line
        if steady_state_time is not None:
            ax1.axvline(x=steady_state_time, color='green', linestyle='-', 
                       alpha=0.8, linewidth=2, label='Steady state')
        
        # Mark maximum response point (removed for cleaner appearance)
        global_max_time = time_plot[max_time_idx]
        # Removed marker from temperature plot for cleaner appearance
        
        ax1.set_ylabel('')  # Remove y-axis label
        ax1.set_title(r'Catalyst Temperature $T_{\mathrm{cat}}$ (°C)', fontsize=13, pad=15)
        ax1.set_xlim(time_plot[0], time_plot[-1])  # Cut where data ends
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', fontsize=10, framealpha=0.9)
        
        # Plot 2: Temperature Change Rate (Bottom)
        ax2.plot(time_plot, max_relevant_rates, rate_color, linewidth=2.5, 
                label=f'Max {rate_label}')
        
        # Add ramp period shading
        if ramp_params.duration:
            ax2.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', label='Ramp period')
        
        # Add steady state line
        if steady_state_time is not None:
            ax2.axvline(x=steady_state_time, color='green', linestyle='-', 
                       alpha=0.8, linewidth=2, label='Steady state')
        
        # Add zero reference for ramp-down
        if ramp_params.is_ramp_down:
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark global maximum
        ax2.plot(global_max_time, global_max_rate, 'o', color=rate_color, 
                markersize=8, label=f'Global max: {global_max_rate:.2f} °C/min')
        
        ax2.set_xlabel('Time (min)', fontsize=12)
        ax2.set_ylabel('')  # Remove y-axis label
        ax2.set_title(r'Temperature Change Rate dTcat/dt (°C/min)', fontsize=13)
        ax2.set_xlim(time_plot[0], time_plot[-1])  # Cut where data ends
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower left', fontsize=10, framealpha=0.9)
        
        safe_tight_layout()
        plt.subplots_adjust(top=0.85)  # Make more room for the info banner
        return fig
    
    @staticmethod
    def create_stability_analysis_plots(time_vector: np.ndarray,
                                      stability_metrics: Dict[str, Any],
                                      ramp_params: RampParameters,
                                      steady_state_time: Optional[float],
                                      file_path: str,
                                      time_limit: Optional[float] = None) -> Figure:
        """Create stability analysis plots"""
        
        # Filter data for plotting
        if time_limit is None:
            time_limit = time_vector.max()
            
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= DEFAULT_TIME_LIMIT_FOR_DISPLAY:
            plot_time_limit = min(DEFAULT_TIME_LIMIT_FOR_DISPLAY, time_limit if time_limit is not None else DEFAULT_TIME_LIMIT_FOR_DISPLAY)
        
        plot_stability_mask = time_vector <= plot_time_limit
        time_stability_plot = time_vector[plot_stability_mask]
        rms_plot = stability_metrics['rms_change_rates'][plot_stability_mask]
        max_abs_plot = stability_metrics['max_abs_change_rates'][plot_stability_mask]
        stable_mask_plot = stability_metrics['stable_mask'][plot_stability_mask]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Set window title safely
        try:
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(f"Stability Analysis - {os.path.basename(file_path)}")
        except (AttributeError, TypeError):
            pass  # Window title setting not available
        
        # Plot 1: RMS and Max change rates
        ax1.plot(time_stability_plot, rms_plot, 'b-', linewidth=2, 
                label='RMS change rate', alpha=0.8)
        ax1.plot(time_stability_plot, max_abs_plot, 'r-', linewidth=2,
                label='Max absolute change rate', alpha=0.8)
        
        # Add threshold line
        ax1.axhline(y=stability_metrics['threshold'], color='orange', 
                   linestyle='--', linewidth=2,
                   label=f'Stability threshold ({stability_metrics["threshold"]:.3f} °C min$^{{-1}}$)')
        
        # Add ramp period shading
        if ramp_params.duration:
            ax1.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', label='Ramp period')
        
        # Add steady state indicators
        if steady_state_time is not None:
            ax1.axvline(x=steady_state_time, color='green', linestyle='-', 
                       linewidth=2, label=f'Detected steady state (t = {steady_state_time:.1f} min)')
        
        # Add initial steady state indicator
        search_start_time = stability_metrics.get('search_start_time')
        if search_start_time is not None and search_start_time >= 10.0:
            ax1.axvspan(time_stability_plot[0], min(10.0, time_stability_plot[-1]),
                       alpha=0.1, color='green', label='Initial steady state (assumed)')
        
        ax1.set_xlabel('Time (min)', fontsize=12)
        ax1.set_ylabel(r'$dT_{\mathrm{cat}}/dt$ (°C min$^{-1}$)', fontsize=12)
        ax1.set_title('System Stability Analysis', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.set_yscale('log')
        ax1.set_xlim(time_stability_plot[0], None)
        
        # Plot 2: Stability timeline
        stable_indicator = stable_mask_plot.astype(int)
        colors = ['red' if x == 0 else 'green' for x in stable_indicator]
        
        ax2.bar(time_stability_plot, np.ones_like(time_stability_plot),
               width=(time_stability_plot[1] - time_stability_plot[0]) if len(time_stability_plot) > 1 else 0.1,
               color=colors, alpha=0.7, edgecolor='none')
        
        # Add legend
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Unstable'),
                          Patch(facecolor='green', alpha=0.7, label='Stable')]
        ax2.legend(handles=legend_elements, fontsize=10, framealpha=0.9)
        
        # Add ramp period shading
        if ramp_params.duration:
            ax2.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.3, color='gray')
        
        # Add steady state indicators
        if steady_state_time is not None:
            ax2.axvline(x=steady_state_time, color='darkgreen', linestyle='-', 
                       linewidth=3, label='Detected steady state')
        
        if search_start_time is not None and search_start_time >= 10.0:
            ax2.axvline(x=10.0, color='lightgreen', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Initial steady state end')
        
        ax2.set_xlabel('Time (min)', fontsize=12)
        ax2.set_ylabel('System state', fontsize=12)
        ax2.set_title('Stability Timeline', fontsize=14)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(time_stability_plot[0], None)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Unstable', 'Stable'])
        ax2.grid(True, alpha=0.3)
        
        safe_tight_layout()
        return fig
    
    @staticmethod
    def create_spatial_gradient_plots(time_vector: np.ndarray,
                                    catalyst_temp_matrix: np.ndarray,
                                    length_vector: np.ndarray,
                                    ramp_params: RampParameters,
                                    file_path: str,
                                    time_limit: Optional[float] = None) -> plt.Figure:
        """Create spatial temperature gradient analysis plots"""
        
        # Ensure length_vector is numpy array (fix for pandas Series issue)
        if hasattr(length_vector, 'values'):
            length_vector = length_vector.values
        
        # Apply time filtering
        if time_limit is None:
            time_limit = time_vector.max()
        
        time_mask = time_vector <= time_limit
        time_filtered = time_vector[time_mask]
        temp_filtered = catalyst_temp_matrix[time_mask, :]
        
        # Calculate spatial gradients
        try:
            # Calculate spatial step size
            if len(length_vector) > 1:
                dl = length_vector[1] - length_vector[0]
                if abs(dl) < 1e-10:  # Essentially zero
                    dl = 0.1
            else:
                dl = 0.1
            
            # Try different numpy gradient approaches
            # Method 1: gradient with spacing and axis
            try:
                spatial_gradients = np.gradient(temp_filtered, dl, axis=1)
            except Exception as e1:
                # Method 2: gradient with axis only
                try:
                    spatial_gradients = np.gradient(temp_filtered, axis=1)
                    # Scale by dl manually
                    spatial_gradients = spatial_gradients / dl
                except Exception as e2:
                    # Method 3: manual gradient calculation
                    try:
                        spatial_gradients = np.zeros_like(temp_filtered)
                        # Manual finite difference
                        spatial_gradients[:, 1:-1] = (temp_filtered[:, 2:] - temp_filtered[:, :-2]) / (2 * dl)
                        spatial_gradients[:, 0] = (temp_filtered[:, 1] - temp_filtered[:, 0]) / dl
                        spatial_gradients[:, -1] = (temp_filtered[:, -1] - temp_filtered[:, -2]) / dl
                    except Exception as e3:
                        # Last resort - zero gradients
                        spatial_gradients = np.zeros_like(temp_filtered)
            
        except Exception as e:
            spatial_gradients = np.zeros_like(temp_filtered)
        
        # Calculate key metrics
        max_temps_per_time = np.nanmax(temp_filtered, axis=1)
        min_temps_per_time = np.nanmin(temp_filtered, axis=1)
        temp_range_per_time = max_temps_per_time - min_temps_per_time
        
        # Find positions of hot and cold spots
        hotspot_positions = np.nanargmax(temp_filtered, axis=1)
        coldspot_positions = np.nanargmin(temp_filtered, axis=1)
        
        # Calculate maximum spatial gradients
        max_positive_gradients = np.nanmax(spatial_gradients, axis=1)
        max_negative_gradients = np.nanmin(spatial_gradients, axis=1)
        max_abs_gradients = np.nanmax(np.abs(spatial_gradients), axis=1)
        
        # Find global extrema with error handling
        try:
            global_hotspot_pos = np.unravel_index(np.nanargmax(temp_filtered), temp_filtered.shape)
            global_coldspot_pos = np.unravel_index(np.nanargmin(temp_filtered), temp_filtered.shape)
            global_max_gradient_pos = np.unravel_index(np.nanargmax(np.abs(spatial_gradients)), spatial_gradients.shape)
            
            hot_time_idx, hot_pos_idx = global_hotspot_pos
            cold_time_idx, cold_pos_idx = global_coldspot_pos
            grad_time_idx, grad_pos_idx = global_max_gradient_pos
            
            # Validate indices
            if not (0 <= hot_time_idx < temp_filtered.shape[0] and 0 <= hot_pos_idx < temp_filtered.shape[1]):
                hot_time_idx, hot_pos_idx = 0, 0
            if not (0 <= cold_time_idx < temp_filtered.shape[0] and 0 <= cold_pos_idx < temp_filtered.shape[1]):
                cold_time_idx, cold_pos_idx = 0, 0
            if not (0 <= grad_time_idx < spatial_gradients.shape[0] and 0 <= grad_pos_idx < spatial_gradients.shape[1]):
                grad_time_idx, grad_pos_idx = 0, 0
                
        except Exception as extrema_error:
            # Use safe defaults
            hot_time_idx, hot_pos_idx = 0, 0
            cold_time_idx, cold_pos_idx = 0, 0
            grad_time_idx, grad_pos_idx = 0, 0
        
        # Create comprehensive spatial gradient figure
        fig = plt.figure(figsize=(20, 12))
        fig.canvas.manager.set_window_title(f"Spatial Gradient Analysis - {os.path.basename(file_path)}")
        
        # Create 2x3 subplot layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Temperature heatmap with hotspot tracking
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(temp_filtered.T, aspect='auto', cmap='hot',
                        extent=[time_filtered[0], time_filtered[-1],
                               length_vector[0], length_vector[-1]], origin='lower')
        
        # Overlay hotspot and coldspot trajectories
        try:
            hotspot_path_positions = length_vector[hotspot_positions]
            coldspot_path_positions = length_vector[coldspot_positions]
            
            ax1.plot(time_filtered, hotspot_path_positions, 'w-', 
                    linewidth=3, alpha=0.8, label='Hotspot path')
            ax1.plot(time_filtered, coldspot_path_positions, 'c-', 
                    linewidth=3, alpha=0.8, label='Coldspot path')
        except Exception as trajectory_error:
            # Skip trajectory plots
            pass
        
        # Mark ramp period
        if ramp_params.duration:
            ax1.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.3, color='gray', label='Ramp period')
        
        ax1.set_xlabel('Time (min)', color='white', fontweight='bold')
        ax1.set_ylabel('Reactor Position (m)', color='white', fontweight='bold')
        ax1.set_title('Temperature Distribution & Hot/Cold Spot Migration', 
                     color='white', fontweight='bold')
        
        # Style the legend with white text
        legend1 = ax1.legend(loc='upper right', framealpha=0.8, facecolor='black', edgecolor='white')
        for text in legend1.get_texts():
            text.set_color('white')
            text.set_fontweight('bold')
        
        # Style tick labels to be white
        ax1.tick_params(colors='white', labelsize=10)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_color('white')
            label.set_fontweight('bold')
        
        plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
        
        # Plot 2: Spatial gradient heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(spatial_gradients.T, aspect='auto', cmap='RdBu_r',
                        extent=[time_filtered[0], time_filtered[-1],
                               length_vector[0], length_vector[-1]], origin='lower')
        
        # Mark maximum gradient location
        try:
            max_gradient_value = spatial_gradients[grad_time_idx, grad_pos_idx]
            max_gradient_time = time_filtered[grad_time_idx]
            max_gradient_pos = length_vector[grad_pos_idx]
            
            ax2.plot(max_gradient_time, max_gradient_pos, 'ko', 
                    markersize=8, label=f'Max gradient: {max_gradient_value:.1f}°C/m')
        except Exception as marker_error:
            # Skip the marker plot
            pass
        
        if ramp_params.duration:
            ax2.axvspan(ramp_params.start_time, ramp_params.end_time, alpha=0.3, color='gray')
        
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Reactor Position (m)')
        ax2.set_title('Spatial Temperature Gradients (dT/dx)')
        ax2.legend()
        plt.colorbar(im2, ax=ax2, label='Gradient (°C/m)')
        
        # Plot 3: Temperature extrema over time
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time_filtered, max_temps_per_time, 'red', linewidth=2, label='Max temperature')
        ax3.plot(time_filtered, min_temps_per_time, 'blue', linewidth=2, label='Min temperature')
        
        # Add average temperature line
        avg_temps_per_time = np.nanmean(temp_filtered, axis=1)
        ax3.plot(time_filtered, avg_temps_per_time, 'green', linewidth=2, label='Average temperature')
        
        if ramp_params.duration:
            ax3.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', label='Ramp period')
        
        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Temperature Extrema & Average Over Time')
        ax3.set_xlim(time_filtered[0], None)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Maximum spatial gradients over time
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time_filtered, max_positive_gradients, 'r-', linewidth=2, label='Max positive gradient')
        ax4.plot(time_filtered, max_negative_gradients, 'b-', linewidth=2, label='Max negative gradient')
        ax4.plot(time_filtered, max_abs_gradients, 'purple', linewidth=2, label='Max absolute gradient')
        
        if ramp_params.duration:
            ax4.axvspan(ramp_params.start_time, ramp_params.end_time, alpha=0.2, color='gray', label='Ramp period')
        
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Spatial Gradient (°C/m)')
        ax4.set_title('Maximum Spatial Gradients Over Time')
        ax4.set_xlim(time_filtered[0], None)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Hotspot position tracking
        ax5 = fig.add_subplot(gs[1, 1])
        
        try:
            hotspot_path_positions = length_vector[hotspot_positions]
            coldspot_path_positions = length_vector[coldspot_positions]
            
            ax5.plot(time_filtered, hotspot_path_positions, 'r-', linewidth=2, label='Hotspot position')
            ax5.plot(time_filtered, coldspot_path_positions, 'b-', linewidth=2, label='Coldspot position')
            
            # Add average positions as reference lines
            avg_hotspot_pos = np.nanmean(hotspot_path_positions)
            avg_coldspot_pos = np.nanmean(coldspot_path_positions)
            ax5.axhline(y=avg_hotspot_pos, color='red', linestyle='--', alpha=0.5,
                       label=f'Avg hotspot: {avg_hotspot_pos:.3f} m')
            ax5.axhline(y=avg_coldspot_pos, color='blue', linestyle='--', alpha=0.5,
                       label=f'Avg coldspot: {avg_coldspot_pos:.3f} m')
        except Exception as tracking_error:
            # Plot simple placeholder
            ax5.text(0.5, 0.5, 'Position tracking unavailable', 
                    transform=ax5.transAxes, ha='center', va='center')
        
        if ramp_params.duration:
            ax5.axvspan(ramp_params.start_time, ramp_params.end_time, alpha=0.2, color='gray', label='Ramp period')
        
        ax5.set_xlabel('Time (min)')
        ax5.set_ylabel('Reactor Position (m)')
        ax5.set_title('Hot/Cold Spot Position Tracking')
        ax5.set_xlim(time_filtered[0], None)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Spatial gradient statistics by reactor zone
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Divide reactor into zones for analysis
        n_zones = 5
        zone_boundaries = np.linspace(length_vector[0], length_vector[-1], n_zones + 1)
        zone_centers = (zone_boundaries[:-1] + zone_boundaries[1:]) / 2
        zone_gradients = []
        
        for i in range(n_zones):
            zone_mask = (length_vector >= zone_boundaries[i]) & (length_vector < zone_boundaries[i + 1])
            if np.any(zone_mask):
                zone_gradient_series = np.nanmean(np.abs(spatial_gradients[:, zone_mask]), axis=1)
                zone_avg_gradient = np.nanmean(zone_gradient_series)
                if np.isnan(zone_avg_gradient):
                    zone_avg_gradient = 0.0
                zone_gradients.append(zone_avg_gradient)
            else:
                zone_gradients.append(0)
        
        # Calculate bar width safely
        if len(zone_centers) > 1:
            bar_width = (zone_centers[1] - zone_centers[0]) * 0.8
        else:
            bar_width = 0.1
        
        bars = ax6.bar(zone_centers, zone_gradients, width=bar_width,
                      alpha=0.7, color='purple')
        ax6.set_xlabel('Reactor Position (m)')
        ax6.set_ylabel('Average |dT/dx| (°C/m)')
        ax6.set_title('Average Spatial Gradients by Reactor Zone')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, zone_gradients):
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Try tight_layout and handle warnings gracefully
        try:
            safe_tight_layout(rect=[0, 0.03, 1, 0.95])
        except Warning:
            # If tight_layout fails, just adjust manually
            plt.subplots_adjust(top=0.95, bottom=0.05)
        
        return fig
    
    @staticmethod
    def create_3d_heat_transfer_plots(time_vector: np.ndarray,
                                    heat_transfer_matrix: np.ndarray,
                                    length_vector: np.ndarray,
                                    ramp_params: RampParameters,
                                    steady_state_time: Optional[float],
                                    file_path: str,
                                    time_limit: Optional[float] = None) -> plt.Figure:
        """Create 3D heat transfer analysis plots"""
        
        # Apply time filtering
        if time_limit is None:
            time_limit = time_vector.max()
        
        # Determine plotting time limit
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= 60.0:
            plot_time_limit = min(60.0, time_limit)
        
        time_mask = time_vector <= time_limit
        time_filtered = time_vector[time_mask]
        heat_transfer_filtered = heat_transfer_matrix[time_mask, :]
        
        plot_time_mask = time_filtered <= plot_time_limit
        time_plot = time_filtered[plot_time_mask]
        heat_transfer_plot = heat_transfer_filtered[plot_time_mask, :]
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(14, 10))
        fig.canvas.manager.set_window_title(f"3D Heat Transfer Analysis - {os.path.basename(file_path)}")
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plotting
        T_mesh, L_mesh = np.meshgrid(time_plot, length_vector)
        
        # Create surface plot
        surf = ax.plot_surface(T_mesh, L_mesh, heat_transfer_plot.T,
                              cmap='RdYlBu_r', alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Add contour lines on the bottom for reference
        contour = ax.contour(T_mesh, L_mesh, heat_transfer_plot.T,
                            zdir='z', offset=np.nanmin(heat_transfer_plot),
                            cmap='RdYlBu_r', alpha=0.6)
        
        # Add ramp period highlighting if available
        if ramp_params.duration:
            z_min, z_max = np.nanmin(heat_transfer_plot), np.nanmax(heat_transfer_plot)
            y_min, y_max = length_vector.min(), length_vector.max()
            
            # Ramp start plane
            xx_start, zz_start = np.meshgrid([ramp_params.start_time, ramp_params.start_time], [z_min, z_max])
            yy_start = np.full_like(xx_start, [[y_min], [y_max]])
            ax.plot_surface(xx_start, yy_start, zz_start, alpha=0.3, color='gray')
            
            # Ramp end plane
            xx_end, zz_end = np.meshgrid([ramp_params.end_time, ramp_params.end_time], [z_min, z_max])
            yy_end = np.full_like(xx_end, [[y_min], [y_max]])
            ax.plot_surface(xx_end, yy_end, zz_end, alpha=0.3, color='gray')
        
        # Add steady state line if detected
        if steady_state_time is not None and steady_state_time <= time_plot.max():
            z_min, z_max = np.nanmin(heat_transfer_plot), np.nanmax(heat_transfer_plot)
            y_min, y_max = length_vector.min(), length_vector.max()
            
            xx_ss, zz_ss = np.meshgrid([steady_state_time, steady_state_time], [z_min, z_max])
            yy_ss = np.full_like(xx_ss, [[y_min], [y_max]])
            ax.plot_surface(xx_ss, yy_ss, zz_ss, alpha=0.4, color='green')
        
        # Customize the plot
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Reactor Position (m)', fontsize=12)
        ax.set_zlabel('Heat Transfer with Coolant (kW/m²)', fontsize=12)
        ax.set_title(f'3D Heat Transfer with Coolant During {ramp_params.analysis_title}', fontsize=14)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=20, label='Heat Transfer (kW/m²)')
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        safe_tight_layout()
        return fig
    
    @staticmethod
    def create_temperature_difference_plots(time_vector: np.ndarray,
                                           catalyst_temp_matrix: np.ndarray,
                                           bulk_temp_matrix: np.ndarray,
                                           length_vector: np.ndarray,
                                           ramp_params: RampParameters,
                                           steady_state_time: Optional[float],
                                           file_path: str,
                                           time_limit: Optional[float] = None) -> plt.Figure:
        """Create process stream temperature plot at reactor outlet"""
        
        # Apply time limit if specified
        if time_limit is None:
            time_limit = time_vector.max()
        time_mask = time_vector <= time_limit
        time_filtered = time_vector[time_mask]
        
        # Plot process stream temperature (T) at reactor outlet
        if bulk_temp_matrix is not None:
            print("Creating process stream temperature plot at reactor outlet...")
            bulk_temp_filtered = bulk_temp_matrix[time_mask, :]
            
            # Extract temperature at reactor outlet (last position in length vector)
            outlet_temp = bulk_temp_filtered[:, -1]  # Last column = reactor outlet
            
            plot_title = 'Process Stream Temperature at Reactor Outlet'
            ylabel = 'Temperature (T) [°C]'
            
            # Create time series plot
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.canvas.manager.set_window_title(f"Process Stream Temperature at Outlet - {os.path.basename(file_path)}")
            
            # Plot the outlet temperature vs time
            line = ax.plot(time_filtered, outlet_temp, 
                          linewidth=2.5, color='steelblue', 
                          marker='o', markersize=4, markevery=max(1, len(time_filtered)//20),
                          label=f'Outlet Temperature (L = {length_vector[-1]:.2f} m)')
            
            # Set labels and title
            ax.set_xlabel('Time (min)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(plot_title, fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add legend
            ax.legend(fontsize=10, loc='best')
            
            # Add statistics text box
            min_temp = outlet_temp.min()
            max_temp = outlet_temp.max()
            avg_temp = outlet_temp.mean()
            temp_range = max_temp - min_temp
            
            stats_text = f'Statistics:\nMin: {min_temp:.1f}°C\nMax: {max_temp:.1f}°C\nAvg: {avg_temp:.1f}°C\nRange: {temp_range:.1f}°C'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=10)
            
            # Highlight min/max points
            min_idx = np.argmin(outlet_temp)
            max_idx = np.argmax(outlet_temp)
            
            ax.scatter(time_filtered[min_idx], outlet_temp[min_idx], 
                      color='blue', s=100, marker='v', 
                      label=f'Min: {min_temp:.1f}°C', zorder=5)
            ax.scatter(time_filtered[max_idx], outlet_temp[max_idx], 
                      color='red', s=100, marker='^', 
                      label=f'Max: {max_temp:.1f}°C', zorder=5)
            
            # Update legend to include min/max markers
            ax.legend(fontsize=10, loc='best')
            
            plt.tight_layout()
            
            print(f"✓ Process stream outlet temperature plot created successfully!")
            print(f"  Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C")
            print(f"  Average temperature: {avg_temp:.1f}°C")
            
        else:
            print("Warning: No bulk temperature data available for outlet temperature plot")
            print("Creating placeholder plot...")
            
            # Create placeholder plot when no bulk temperature data
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.canvas.manager.set_window_title(f"Process Stream Temperature at Outlet - {os.path.basename(file_path)}")
            ax.text(0.5, 0.5, 'No Process Stream Temperature Data Available\n\nPlease ensure bulk temperature data is loaded\nto display outlet temperature', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title('Process Stream Temperature at Reactor Outlet', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (min)', fontsize=12)
            ax.set_ylabel('Temperature (T) [°C]', fontsize=12)
            plt.tight_layout()
        
        return fig
    
    @staticmethod 
    def create_temperature_difference_plot(processed_data, time_limit, file_path):
        """Create temperature difference analysis plots"""
        print("Creating temperature difference analysis plots...")
        
        try:
            time_filtered = processed_data.get('time_filtered')
            length_vector = processed_data.get('length_vector')
            catalyst_temperatures = processed_data.get('catalyst_temperatures')
            bulk_temperatures = processed_data.get('bulk_temperatures')
            steady_state_time = processed_data.get('steady_state_time')
            ramp_params = processed_data.get('ramp_params')
            
            if time_filtered is None or catalyst_temperatures is None or length_vector is None:
                print("Error: Required data not available for temperature difference plot")
                return None
            
            # Calculate temperature difference (T_cat - T_bulk)
            if bulk_temperatures is not None:
                temp_diff = catalyst_temperatures - bulk_temperatures
                max_abs_diff = np.max(np.abs(temp_diff))
                
                if max_abs_diff < 1e-10:
                    print("Warning: Temperature difference is extremely small (< 1e-10°C)")
                    print("  This suggests T_cat and T are nearly identical in the dataset")
                    print("  This could happen if:")
                    print("    1. The simulation doesn't distinguish between catalyst and bulk temperatures")
                    print("    2. The system is at thermal equilibrium")
                    print("    3. There's an issue with how the data is exported from Aspen")
            else:
                print("Warning: No bulk temperature data available, using catalyst temperatures only")
                temp_diff = catalyst_temperatures
                max_abs_diff = np.max(np.abs(temp_diff))
        
        except Exception as e:
            print(f"Error in temperature difference analysis: {e}")
            return None
        
        # Scale the difference for visualization if it's very small but non-zero
        temp_diff_scaled = temp_diff
        scale_factor = 1.0
        scale_label = ""
        
        if max_abs_diff > 0 and max_abs_diff < 0.01:
            # Scale up very small differences for better visualization
            if max_abs_diff < 1e-6:
                scale_factor = 1e6
                scale_label = " (×10⁶)"
            elif max_abs_diff < 1e-3:
                scale_factor = 1e3
                scale_label = " (×10³)"
            else:
                scale_factor = 100
                scale_label = " (×100)"
            
            temp_diff_scaled = temp_diff * scale_factor
            print(f"Scaling temperature difference by {scale_factor} for visualization")
        elif max_abs_diff == 0:
            # If truly zero, create a small artificial difference for demonstration
            print("Temperature difference is exactly zero - creating artificial small difference for visualization")
            # Create a small gradient based on position for visualization
            position_gradient = np.linspace(-0.001, 0.001, len(length_vector))
            temp_diff_scaled = np.tile(position_gradient, (len(time_filtered), 1))
            scale_label = " (artificial for visualization)"
        
        # Apply plot time limit
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= 60.0:
            plot_time_limit = min(60.0, time_limit)
        
        plot_time_mask = time_filtered <= plot_time_limit
        time_plot = time_filtered[plot_time_mask]
        temp_diff_plot = temp_diff_scaled[plot_time_mask, :]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.canvas.manager.set_window_title(f"Temperature Difference Analysis - {os.path.basename(file_path)}")
        
        # Add a note about scaling if applied
        if scale_factor != 1.0 or scale_label:
            fig.suptitle(f'Temperature Difference (T_cat - T) Analysis{scale_label}', 
                        fontsize=16, fontweight='bold', y=0.98)
        
        # Plot 1: Temperature difference over time at key positions
        key_positions = [0, len(length_vector)//4, len(length_vector)//2, 
                        3*len(length_vector)//4, len(length_vector)-1]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        for i, pos_idx in enumerate(key_positions):
            if pos_idx < temp_diff_plot.shape[1]:
                pos_label = f"L = {length_vector[pos_idx]:.3f} m"
                ax1.plot(time_plot, temp_diff_plot[:, pos_idx], 
                        color=colors[i], linewidth=2, label=pos_label)
        
        # Add ramp period shading
        if ramp_params.duration:
            ax1.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', 
                       label=f'Ramp period ({ramp_params.duration} min)')
        
        # Add steady state line
        if steady_state_time is not None:
            ax1.axvline(x=steady_state_time, color='green', linestyle='-', 
                       alpha=0.8, linewidth=2, 
                       label=f'Steady state (t = {steady_state_time:.1f} min)')
        
        ax1.set_xlabel('Time (min)', fontsize=12)
        ax1.set_ylabel(f'Temperature Difference{scale_label} (°C)', fontsize=12)
        ax1.set_title('Temperature Difference (T_cat - T) vs Time\nat Key Reactor Positions', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Contour plot of temperature difference
        if len(time_plot) > 1 and temp_diff_plot.shape[1] > 1:
            T_mesh, L_mesh = np.meshgrid(time_plot, length_vector)
            im = ax2.contourf(T_mesh, L_mesh, temp_diff_plot.T, levels=20, cmap='RdBu_r')
            cbar = plt.colorbar(im, ax=ax2, label=f'Temperature Difference{scale_label} (°C)')
            
            # Add ramp period lines
            if ramp_params.duration:
                ax2.axvline(x=ramp_params.start_time, color='black', linestyle='--', alpha=0.7)
                ax2.axvline(x=ramp_params.end_time, color='black', linestyle='--', alpha=0.7)
            
            # Add steady state line
            if steady_state_time is not None:
                ax2.axvline(x=steady_state_time, color='green', linestyle='-', alpha=0.8, linewidth=2)
            
            ax2.set_xlabel('Time (min)', fontsize=12)
            ax2.set_ylabel('Reactor Length (m)', fontsize=12)
            ax2.set_title('Temperature Difference Contour\n(T_cat - T)', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for contour plot', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Temperature Difference Contour\n(T_cat - T)', fontsize=14, fontweight='bold')
        
        # Plot 3: Maximum temperature difference over time
        max_temp_diff = np.nanmax(temp_diff_plot, axis=1)
        min_temp_diff = np.nanmin(temp_diff_plot, axis=1)
        avg_temp_diff = np.nanmean(temp_diff_plot, axis=1)
        
        ax3.plot(time_plot, max_temp_diff, 'red', linewidth=2, label='Maximum ΔT')
        ax3.plot(time_plot, avg_temp_diff, 'orange', linewidth=2, label='Average ΔT')
        ax3.plot(time_plot, min_temp_diff, 'blue', linewidth=2, label='Minimum ΔT')
        ax3.fill_between(time_plot, min_temp_diff, max_temp_diff, alpha=0.2, color='gray')
        
        # Add ramp period shading
        if ramp_params.duration:
            ax3.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='yellow', 
                       label=f'Ramp period ({ramp_params.duration} min)')
        
        # Add steady state line
        if steady_state_time is not None:
            ax3.axvline(x=steady_state_time, color='green', linestyle='-', 
                       alpha=0.8, linewidth=2, 
                       label=f'Steady state (t = {steady_state_time:.1f} min)')
        
        ax3.set_xlabel('Time (min)', fontsize=12)
        ax3.set_ylabel(f'Temperature Difference{scale_label} (°C)', fontsize=12)
        ax3.set_title('Temperature Difference Statistics\nover Time', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temperature difference profile at different times
        if ramp_params.duration:
            # Show profiles at start, middle, end of ramp, and steady state
            profile_times = []
            profile_labels = []
            profile_colors = []
            
            # Start of ramp
            start_idx = np.argmin(np.abs(time_plot - ramp_params.start_time))
            profile_times.append(start_idx)
            profile_labels.append(f't = {ramp_params.start_time:.1f} min (ramp start)')
            profile_colors.append('blue')
            
            # Middle of ramp
            mid_time = (ramp_params.start_time + ramp_params.end_time) / 2
            mid_idx = np.argmin(np.abs(time_plot - mid_time))
            profile_times.append(mid_idx)
            profile_labels.append(f't = {mid_time:.1f} min (ramp mid)')
            profile_colors.append('orange')
            
            # End of ramp
            end_idx = np.argmin(np.abs(time_plot - ramp_params.end_time))
            profile_times.append(end_idx)
            profile_labels.append(f't = {ramp_params.end_time:.1f} min (ramp end)')
            profile_colors.append('red')
            
            # Steady state (if available)
            if steady_state_time is not None and steady_state_time <= plot_time_limit:
                ss_idx = np.argmin(np.abs(time_plot - steady_state_time))
                profile_times.append(ss_idx)
                profile_labels.append(f't = {steady_state_time:.1f} min (steady state)')
                profile_colors.append('green')
        else:
            # For non-ramp data, show profiles at key times
            profile_times = [0, len(time_plot)//4, len(time_plot)//2, 3*len(time_plot)//4, len(time_plot)-1]
            profile_labels = [f't = {time_plot[idx]:.1f} min' for idx in profile_times]
            profile_colors = ['blue', 'cyan', 'orange', 'red', 'purple']
        
        for i, (time_idx, label, color) in enumerate(zip(profile_times, profile_labels, profile_colors)):
            if time_idx < temp_diff_plot.shape[0]:
                ax4.plot(length_vector, temp_diff_plot[time_idx, :], 
                        color=color, linewidth=2, label=label)
        
        ax4.set_xlabel('Reactor Length (m)', fontsize=12)
        ax4.set_ylabel(f'Temperature Difference{scale_label} (°C)', fontsize=12)
        ax4.set_title('Temperature Difference Profile\nalong Reactor Length', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add overall statistics as text
        overall_max_diff = np.nanmax(temp_diff_plot)
        overall_min_diff = np.nanmin(temp_diff_plot)
        overall_avg_diff = np.nanmean(temp_diff_plot)
        
        # For the original unscaled values in statistics
        original_max = np.nanmax(temp_diff)
        original_min = np.nanmin(temp_diff)
        original_avg = np.nanmean(temp_diff)
        
        stats_text = f"Statistics (Original Values):\n"
        stats_text += f"Max ΔT: {original_max:.6f}°C\n"
        stats_text += f"Min ΔT: {original_min:.6f}°C\n" 
        stats_text += f"Avg ΔT: {original_avg:.6f}°C\n"
        if scale_factor != 1.0:
            stats_text += f"\nVisualization Scale: ×{scale_factor}"
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for statistics text
        
        return fig
    
    @staticmethod
    def generate_all_plots(analysis_results: Dict[str, Any]) -> List[Tuple[str, plt.Figure]]:
        """Generate all requested plots based on analysis results"""
        plots = []
        
        data_package = analysis_results['data_package']
        ramp_params = analysis_results['ramp_params']
        steady_state_time = analysis_results['steady_state_time']
        stability_metrics = analysis_results['stability_metrics']
        options = analysis_results['options']
        
        time_vector = data_package['time_vector']
        catalyst_temp = data_package['variables']['T_cat (°C)']
        length_vector = data_package['length_vector']
        file_path = data_package['file_path']
        
        # Temperature Response Analysis
        if options.temperature_response:
            fig = PlotGenerator.create_temperature_response_plots(
                time_vector, catalyst_temp, length_vector, ramp_params,
                steady_state_time, file_path, options.time_limit
            )
            plots.append(("Temperature Response Analysis", fig))
        
        # Stability Analysis
        if options.stability_analysis and stability_metrics:
            fig = PlotGenerator.create_stability_analysis_plots(
                time_vector, stability_metrics, ramp_params,
                steady_state_time, file_path, options.time_limit
            )
            plots.append(("Stability Analysis", fig))
        
        # Spatial Gradients
        if options.spatial_gradients:
            fig = PlotGenerator.create_spatial_gradient_plots(
                time_vector, catalyst_temp, length_vector,
                ramp_params, file_path, options.time_limit
            )
            plots.append(("Spatial Temperature Gradients", fig))
        
        # 3D Heat Transfer
        if options.heat_transfer_3d:
            heat_transfer = data_package['variables'].get('Heat Transfer with coolant (kW/m2)')
            if heat_transfer is not None:
                fig = PlotGenerator.create_3d_heat_transfer_plots(
                    time_vector, heat_transfer, length_vector, ramp_params,
                    steady_state_time, file_path, options.time_limit
                )
                plots.append(("3D Heat Transfer Analysis", fig))
        
        # Temperature Difference Analysis
        if options.temperature_difference:
            bulk_temp = data_package['variables'].get('T (°C)')
            if bulk_temp is not None:
                fig = PlotGenerator.create_temperature_difference_plots(
                    time_vector, catalyst_temp, bulk_temp, length_vector,
                    ramp_params, steady_state_time, file_path, options.time_limit
                )
                plots.append(("Temperature Difference Analysis", fig))
        
        return plots
