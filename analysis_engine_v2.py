"""
Dynamic Reactor Ramp Analysis Engine v2
======================================

Comprehensive analysis engine for dynamic reactor ramp experiments.
Provides modular analysis pipeline with configurable parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import os

from data_loader import DataLoaderManager


@dataclass
class AnalysisConfig:
    """Configuration for dynamic ramp analysis."""
    time_limit: Optional[float] = None
    ramp_start: Optional[float] = 10.0
    ramp_duration: Optional[float] = None
    steady_threshold: float = 0.05
    steady_min_duration: float = 1.0


def _detect_ramp_direction(file_path: str) -> str:
    """Extract ramp direction from filename."""
    fname = os.path.basename(file_path)
    if '-' in fname:
        after_hyphen = fname.split('-', 1)[1].lower()
        if any(keyword in after_hyphen for keyword in ['up']):
            return 'up'
        elif any(keyword in after_hyphen for keyword in ['down']):
            return 'down'
    return 'unknown'


def load_data(file_path: str) -> Dict[str, Any]:
    """Load data using DataLoaderManager and convert to analysis format."""
    loader = DataLoaderManager()
    data_pkg = loader.load_data(file_path)
    if data_pkg is None:
        raise RuntimeError(f"Failed to load data from {file_path}")

    # Convert metadata using vars() for proper dict conversion
    meta = vars(data_pkg.metadata) if hasattr(data_pkg.metadata, '__dict__') else {}
    meta['file_path'] = file_path

    return {
        'time_vector': data_pkg.time_vector,
        'length_vector': data_pkg.length_vector,
        'variables': data_pkg.variables,
        'metadata': meta
    }


def process_core(raw: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    """Trim time by cfg.time_limit and standardize variable names."""
    tv = raw['time_vector']
    vars_dict = raw['variables']

    if cfg.time_limit is not None:
        mask = tv <= cfg.time_limit
        tv = tv[mask]
        new_vars = {}
        for name, arr in vars_dict.items():
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2:
                    new_vars[name] = arr[mask, :]
                elif arr.ndim == 1:
                    new_vars[name] = arr[mask]
                else:
                    new_vars[name] = arr
            else:
                new_vars[name] = arr
        vars_dict = new_vars

    # Standardize catalyst temperature name
    if 'T_cat' in vars_dict:
        vars_dict['Tcat'] = vars_dict.pop('T_cat')

    return {
        'time_vector': tv,
        'length_vector': raw['length_vector'],
        'variables': vars_dict,
        'metadata': raw['metadata']
    }


def _compute_sliding_window_rates(tv: np.ndarray, Tcat: np.ndarray, window: float, 
                                 ramp_direction: str) -> Tuple[np.ndarray, float, int]:
    """Compute sliding window temperature rates."""
    if Tcat.ndim == 2:
        n_pos = Tcat.shape[1]
        avg_dTcat_dt = []
        for pos in range(n_pos):
            rates = []
            for i in range(len(tv)):
                t0, t1 = tv[i], tv[i] + window
                mask = (tv >= t0) & (tv <= t1)
                if np.sum(mask) < 2:
                    rates.append(np.nan)
                    continue
                delta_T = Tcat[mask, pos][-1] - Tcat[mask, pos][0]
                delta_t = tv[mask][-1] - tv[mask][0]
                rates.append(delta_T / delta_t if delta_t > 0 else np.nan)
            avg_dTcat_dt.append(np.array(rates))
        
        avg_dTcat_dt = np.stack(avg_dTcat_dt, axis=1)
        extreme_func = np.nanmin if ramp_direction == 'down' else np.nanmax
        extreme_func_arg = np.nanargmin if ramp_direction == 'down' else np.nanargmax
        
        extreme_rate_per_pos = extreme_func(avg_dTcat_dt, axis=0)
        extreme_rate = float(extreme_func(extreme_rate_per_pos))
        extreme_idx = int(extreme_func_arg(extreme_rate_per_pos))
        
        return extreme_rate_per_pos, extreme_rate, extreme_idx
    else:
        rates = []
        for i in range(len(tv)):
            t0, t1 = tv[i], tv[i] + window
            mask = (tv >= t0) & (tv <= t1)
            if np.sum(mask) < 2:
                rates.append(np.nan)
                continue
            delta_T = Tcat[mask][-1] - Tcat[mask][0]
            delta_t = tv[mask][-1] - tv[mask][0]
            rates.append(delta_T / delta_t if delta_t > 0 else np.nan)
        
        avg_dTcat_dt = np.array(rates)
        extreme_func = np.nanmin if ramp_direction == 'down' else np.nanmax
        extreme_func_arg = np.nanargmin if ramp_direction == 'down' else np.nanargmax
        
        extreme_rate = float(extreme_func(avg_dTcat_dt))
        extreme_idx = int(extreme_func_arg(avg_dTcat_dt))
        
        return avg_dTcat_dt, extreme_rate, extreme_idx


def _compute_interval_rates(tv: np.ndarray, Tcat: np.ndarray, t0: float, t1: float,
                           ramp_direction: str) -> Tuple[Union[np.ndarray, float], float, int]:
    """Compute temperature rates for a specific time interval."""
    mask = (tv >= t0) & (tv <= t1)
    
    if Tcat.ndim == 2:
        n_pos = Tcat.shape[1]
        rates = []
        for pos in range(n_pos):
            arr = Tcat[mask, pos]
            tvec = tv[mask]
            if len(arr) < 2:
                rates.append(np.nan)
                continue
            delta_T = arr[-1] - arr[0]
            delta_t = tvec[-1] - tvec[0]
            rates.append(delta_T / delta_t if delta_t > 0 else np.nan)
        
        rates_array = np.array(rates)
        extreme_func = np.nanmin if ramp_direction == 'down' else np.nanmax
        extreme_func_arg = np.nanargmin if ramp_direction == 'down' else np.nanargmax
        
        extreme_rate = float(extreme_func(rates_array))
        extreme_idx = int(extreme_func_arg(rates_array))
        
        return rates_array, extreme_rate, extreme_idx
    else:
        arr = Tcat[mask]
        tvec = tv[mask]
        if len(arr) < 2:
            rate = np.nan
        else:
            delta_T = arr[-1] - arr[0]
            delta_t = tvec[-1] - tvec[0]
            rate = delta_T / delta_t if delta_t > 0 else np.nan
        
        extreme_rate = float(rate) if not np.isnan(rate) else np.nan
        return rate, extreme_rate, 0


def compute_derived(core: Dict[str, Any], steady_time: Optional[float] = None) -> Dict[str, Any]:
    """Compute derived quantities: Tcat extremes and time derivatives."""
    tv = core['time_vector']
    vars_dict = core['variables']
    
    if 'Tcat' not in vars_dict:
        raise KeyError("Catalyst temperature 'Tcat' not found in variables")
    
    Tcat = vars_dict['Tcat']
    ramp_direction = _detect_ramp_direction(core['metadata'].get('file_path', ''))
    
    derived = {
        'Tcat_max': float(np.max(Tcat)),
        'Tcat_min': float(np.min(Tcat)),
        'dTcat_dt': np.gradient(Tcat, tv, axis=0),
        'ramp_direction': ramp_direction
    }

    # Define analysis windows
    windows = [1.0, 5.0, 10.0]
    interval_windows = []
    
    # Get timing parameters
    ramp_params = core['metadata'].get('ramp_parameters', {})
    ramp_start = ramp_params.get('start_time')
    ramp_duration = ramp_params.get('duration')
    
    if ramp_start is not None and ramp_duration is not None:
        ramp_end = ramp_start + ramp_duration
        interval_windows.append(('ramptime', ramp_start, ramp_end))
        
        if steady_time is not None:
            interval_windows.append(('stabtime', ramp_start, steady_time))

    # Compute rates for sliding windows
    for window in windows:
        label = f"{int(window)}min"
        rates, extreme_rate, extreme_idx = _compute_sliding_window_rates(
            tv, Tcat, window, ramp_direction)
        
        derived[f'dTcat_dt_extreme_{label}'] = rates
        derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
        derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx

    # Compute rates for interval windows
    for label, t0, t1 in interval_windows:
        rates, extreme_rate, extreme_idx = _compute_interval_rates(
            tv, Tcat, t0, t1, ramp_direction)
        
        derived[f'dTcat_dt_extreme_{label}'] = rates
        derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
        derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx

    return derived


def _calculate_steady_state(tv: np.ndarray, dTdt: np.ndarray, ramp_end: float,
                           cfg: AnalysisConfig) -> Optional[float]:
    """Calculate steady-state time using RMS threshold method."""
    idx0 = int(np.searchsorted(tv, ramp_end))
    segment = dTdt[idx0:]
    
    if segment.ndim == 2:
        rms = np.sqrt(np.mean(segment**2, axis=1))
    else:
        rms = np.sqrt(segment**2)

    dt = np.diff(tv, prepend=tv[0])
    run_time = 0.0
    
    for i, val in enumerate(rms, start=idx0):
        if val <= cfg.steady_threshold:
            run_time += dt[i]
            if run_time >= cfg.steady_min_duration:
                return tv[i]
        else:
            run_time = 0.0
    return None


def calculate_metrics(core: Dict[str, Any], derived: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    """Calculate steady-state and average ramp slope metrics."""
    tv = core['time_vector']
    dTdt = derived['dTcat_dt']
    ramp_direction = derived['ramp_direction']

    rp = core['metadata'].get('ramp_parameters', {})
    ramp_start = rp.get('start_time')
    duration = rp.get('duration')
    
    if ramp_start is None or duration is None:
        raise KeyError("Ramp parameters 'start_time' and 'duration' must be in metadata.ramp_parameters")
    
    ramp_end = ramp_start + duration
    steady_time = _calculate_steady_state(tv, dTdt, ramp_end, cfg)
    
    if steady_time is None:
        raise RuntimeError("Steady state not detected")

    # Calculate average ramp slope
    mask = (tv >= ramp_start) & (tv <= ramp_end)
    avg_slope = float(np.mean(dTdt[mask]))

    # Select extreme dTcat/dt based on ramp direction
    extreme_1min = derived.get('dTcat_dt_extreme_1min_extreme', 0.0)
    
    return {
        'Tcat_max': float(f"{derived['Tcat_max']:.5g}"),
        'Tcat_min': float(f"{derived['Tcat_min']:.5g}"),
        'steady_time': float(f"{steady_time:.5g}"),
        'avg_dTdt_ramp': float(f"{avg_slope:.5g}"),
        'dTcat_dt_extreme': float(f"{extreme_1min:.5g}"),
        'ramp_direction': ramp_direction
    }


def generate_metadata(core: Dict[str, Any], metrics: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    """Generate analysis metadata."""
    file_path = core['metadata'].get('file_path', '')
    filename = os.path.basename(file_path)
    
    return {
        'source_file': filename,
        'cfg': vars(cfg),
        'metrics': metrics,
        'ramp_direction': metrics['ramp_direction']
    }


def run_analysis(file_path: str, cfg: AnalysisConfig) -> Dict[str, Any]:
    """Run complete analysis pipeline."""
    # Load and process data
    raw = load_data(file_path)
    core = process_core(raw, cfg)

    # Calculate steady time first for derived calculations
    tv = core['time_vector']
    if 'Tcat' not in core['variables']:
        raise KeyError("Catalyst temperature 'Tcat' not found in variables")
    
    Tcat = core['variables']['Tcat']
    dTdt = np.gradient(Tcat, tv, axis=0)
    
    # Get ramp parameters
    rp = core['metadata'].get('ramp_parameters', {})
    ramp_start = rp.get('start_time')
    duration = rp.get('duration')
    
    steady_time = None
    if ramp_start is not None and duration is not None:
        ramp_end = ramp_start + duration
        steady_time = _calculate_steady_state(tv, dTdt, ramp_end, cfg)

    # Compute derived quantities and metrics
    derived = compute_derived(core, steady_time=steady_time)
    metrics = calculate_metrics(core, derived, cfg)
    metadata = generate_metadata(core, metrics, cfg)

    # Merge results
    result = {
        'time_vector': core['time_vector'],
        'length_vector': core['length_vector'],
        'variables': core['variables'],
        'metadata': metadata
    }
    
    # Add derived and metrics
    result.update(derived)
    result.update(metrics)
    
    return result

    return {
        'data': result,
        'metadata': metadata
    }
