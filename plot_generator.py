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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import Patch
from typing import Optional, Dict, Any, List, Tuple
from analysis_engine import RampParameters

class PlotGenerator:
    """Handles all plot generation functionality"""
    
    @staticmethod
    def create_temperature_response_plots(time_vector: np.ndarray,
                                        catalyst_temp_matrix: np.ndarray,
                                        length_vector: np.ndarray,
                                        ramp_params: RampParameters,
                                        steady_state_time: Optional[float],
                                        file_path: str,
                                        time_limit: Optional[float] = None) -> plt.Figure:
        """Create main temperature response analysis plots"""
        
        # Filter data
        if time_limit is None:
            time_limit = time_vector.max()
            
        time_mask = time_vector <= time_limit
        time_filtered = time_vector[time_mask]
        temp_filtered = catalyst_temp_matrix[time_mask, :]
        
        # Apply plot time limit
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= 60.0:
            plot_time_limit = min(60.0, time_limit)
        
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
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.canvas.manager.set_window_title(f"Temperature Response Analysis - {os.path.basename(file_path)}")
        
        # Plot 1: Maximum relevant change rates
        ax1.plot(time_plot, max_relevant_rates, rate_color, linewidth=2.5, 
                label=f'Max {rate_label}')
        
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
        
        # Add zero reference for ramp-down
        if ramp_params.is_ramp_down:
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark global maximum
        global_max_time = time_plot[max_time_idx]
        ax1.plot(global_max_time, global_max_rate, 'o', color=rate_color, 
                markersize=10, label=f'Global max: {global_max_rate:.2f} °C min$^{{-1}}$')
        
        ax1.set_xlabel('Time (min)', fontsize=12)
        ax1.set_ylabel(f'{rate_label} (°C min$^{{-1}}$)', fontsize=12)
        ax1.set_title(f'Catalyst Temperature Response During {ramp_params.analysis_title}', fontsize=14)
        ax1.set_xlim(time_plot[0], None)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, framealpha=0.9)
        
        # Plot 2: Temperature at maximum change position
        temp_at_max_pos = temp_plot[:, max_pos_idx]
        
        ax2.plot(time_plot, temp_at_max_pos, 'b-', linewidth=2.5,
                label=rf'$T_{{\mathrm{{cat}}}}$ at L = {length_vector[max_pos_idx]:.2f} m')
        
        # Add ramp period shading
        if ramp_params.duration:
            ax2.axvspan(ramp_params.start_time, ramp_params.end_time, 
                       alpha=0.2, color='gray', label='Ramp period')
        
        # Add steady state line
        if steady_state_time is not None:
            ax2.axvline(x=steady_state_time, color='green', linestyle='-', 
                       alpha=0.8, linewidth=2, label='Steady state')
        
        # Mark maximum response point
        ax2.plot(global_max_time, temp_at_max_pos[max_time_idx], 'o', 
                color=rate_color, markersize=10,
                label=f'Max response point: {temp_at_max_pos[max_time_idx]:.1f} °C')
        
        ax2.set_xlabel('Time (min)', fontsize=12)
        ax2.set_ylabel(r'$T_{\mathrm{cat}}$ (°C)', fontsize=12)
        ax2.set_title(rf'Catalyst Temperature at L = {length_vector[max_pos_idx]:.2f} m', fontsize=14)
        ax2.set_xlim(time_plot[0], None)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_stability_analysis_plots(time_vector: np.ndarray,
                                      stability_metrics: Dict[str, Any],
                                      ramp_params: RampParameters,
                                      steady_state_time: Optional[float],
                                      file_path: str,
                                      time_limit: Optional[float] = None) -> plt.Figure:
        """Create stability analysis plots"""
        
        # Filter data for plotting
        if time_limit is None:
            time_limit = time_vector.max()
            
        plot_time_limit = time_limit
        if steady_state_time is not None and steady_state_time <= 60.0:
            plot_time_limit = min(60.0, time_limit)
        
        plot_stability_mask = time_vector <= plot_time_limit
        time_stability_plot = time_vector[plot_stability_mask]
        rms_plot = stability_metrics['rms_change_rates'][plot_stability_mask]
        max_abs_plot = stability_metrics['max_abs_change_rates'][plot_stability_mask]
        stable_mask_plot = stability_metrics['stable_mask'][plot_stability_mask]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.canvas.manager.set_window_title(f"Stability Analysis - {os.path.basename(file_path)}")
        
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
        
        plt.tight_layout()
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
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        
        plt.tight_layout()
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
        
        return plots
