from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import os

from data_loader import DataLoaderManager

@dataclass
class AnalysisConfig:
    """
    Configuration for a dynamic ramp analysis.
    """
    time_limit: Optional[float] = None        # minutes
    ramp_start: Optional[float] = 10        # ramp start time in minutes
    ramp_duration: Optional[float] = None     # ramp duration in minutes
    steady_threshold: float = 0.05            # threshold °C/min for steady-state
    steady_min_duration: float = 1.0          # consecutive minutes below threshold to call steady

@dataclass
class AnalysisPackage:
    """
    Container for analysis outputs.
    """
    time_vector: np.ndarray
    length_vector: Optional[np.ndarray]
    variables: Dict[str, np.ndarray]
    derived: Dict[str, Any]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


def load_data(file_path: str) -> Dict[str, Any]:
    loader = DataLoaderManager()
    data_pkg = loader.load_data(file_path)
    if data_pkg is None:
        raise RuntimeError(f"Failed to load data from {file_path}")

    # Convert metadata to dict
    if hasattr(data_pkg.metadata, '__dict__'):
        meta = vars(data_pkg.metadata)
    else:
        meta = dict(data_pkg.metadata) # type: ignore
    meta['file_path'] = file_path # type: ignore

    return {
        'time_vector': data_pkg.time_vector,
        'length_vector': data_pkg.length_vector,
        'variables': data_pkg.variables,
        'metadata': meta
    }


def process_core(raw: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    """
    Trim time by cfg.time_limit and standardize variable names.
    """
    tv = raw['time_vector']
    vars_dict = raw['variables']

    if cfg.time_limit is not None:
        mask = tv <= cfg.time_limit
        tv = tv[mask]
        new_vars = {}
        for name, arr in vars_dict.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                new_vars[name] = arr[mask, :]
            elif isinstance(arr, np.ndarray) and arr.ndim == 1:
                new_vars[name] = arr[mask]
            else:
                new_vars[name] = arr
        vars_dict = new_vars

    # Rename catalyst temperature
    if 'T_cat' in vars_dict:
        vars_dict['Tcat'] = vars_dict.pop('T_cat')

    return {
        'time_vector': tv,
        'length_vector': raw['length_vector'],
        'variables': vars_dict,
        'metadata': raw['metadata']
    }


from typing import Optional
def compute_derived(core: Dict[str, Any], steady_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute derived quantities: Tcat extremes and time derivative.
    """
    tv = core['time_vector']
    vars_dict = core['variables']
    derived: Dict[str, Any] = {}

    if 'Tcat' not in vars_dict:
        raise KeyError("Catalyst temperature 'Tcat' not found in variables")
    Tcat = vars_dict['Tcat']

    derived['Tcat_max'] = float(np.max(Tcat))
    derived['Tcat_min'] = float(np.min(Tcat))
    derived['dTcat_dt'] = np.gradient(Tcat, tv, axis=0)
    # Add ramp direction to derived for later use
    ramp_direction = 'unknown'
    fname = os.path.basename(core['metadata'].get('file_path', ''))
    if '-' in fname:
        after_hyphen = fname.split('-', 1)[1].lower()
        if after_hyphen.startswith('up') or '-up' in after_hyphen or 'up' in after_hyphen:
            ramp_direction = 'up'
        elif after_hyphen.startswith('down') or '-down' in after_hyphen or 'down' in after_hyphen:
            ramp_direction = 'down'
    derived['ramp_direction'] = ramp_direction
    
    # Compute sliding average dTcat/dt for 1, 5, 10 min windows, ramp time, and stabilization time
    tv = np.asarray(tv)
    Tcat = np.asarray(Tcat)
    from typing import Union
    windows: list[Union[float, tuple[float, float]]] = [1.0, 5.0, 10.0]
    window_labels = [f"{int(w)}min" for w in windows]

    # Get ramp and stabilization times from metadata if available
    ramp_start = None
    ramp_end = None
    ramp_params = core['metadata'].get('ramp_parameters', {})
    if 'start_time' in ramp_params and 'duration' in ramp_params:
        ramp_start = ramp_params['start_time']
        ramp_end = ramp_start + ramp_params['duration']
    # Use steady_time if provided, else try to get from metadata
    if steady_time is None and 'steady_time' in core['metadata']:
        steady_time = core['metadata']['steady_time']

    # Add ramp and stabilization windows if possible
    if ramp_start is not None and ramp_end is not None:
        windows.append((ramp_start, ramp_end))
        window_labels.append('ramptime')
    if ramp_start is not None and steady_time is not None:
        windows.append((ramp_start, steady_time))
        window_labels.append('stabtime')

    for window, label in zip(windows, window_labels):
        if isinstance(window, (float, int)):
            # Sliding window as before
            w = float(window)
            if Tcat.ndim == 2:
                n_pos = Tcat.shape[1]
                avg_dTcat_dt = []
                for pos in range(n_pos):
                    rates = []
                    for i in range(len(tv)):
                        t0 = tv[i]
                        t1 = t0 + w
                        mask = (tv >= t0) & (tv <= t1)
                        if np.sum(mask) < 2:
                            rates.append(np.nan)
                            continue
                        delta_T = Tcat[mask, pos][-1] - Tcat[mask, pos][0]
                        delta_t = tv[mask][-1] - tv[mask][0]
                        if delta_t == 0:
                            rates.append(np.nan)
                        else:
                            rates.append(delta_T / delta_t)
                    avg_dTcat_dt.append(np.array(rates))
                avg_dTcat_dt = np.stack(avg_dTcat_dt, axis=1)
                if ramp_direction == 'down':
                    extreme_rate_per_pos = np.nanmin(avg_dTcat_dt, axis=0)
                    extreme_rate = float(np.nanmin(extreme_rate_per_pos))
                    extreme_idx = int(np.nanargmin(extreme_rate_per_pos))
                else:
                    extreme_rate_per_pos = np.nanmax(avg_dTcat_dt, axis=0)
                    extreme_rate = float(np.nanmax(extreme_rate_per_pos))
                    extreme_idx = int(np.nanargmax(extreme_rate_per_pos))
                derived[f'dTcat_dt_extreme_{label}'] = extreme_rate_per_pos
                derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
                derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx
            else:
                rates = []
                for i in range(len(tv)):
                    t0 = tv[i]
                    t1 = t0 + w
                    mask = (tv >= t0) & (tv <= t1)
                    if np.sum(mask) < 2:
                        rates.append(np.nan)
                        continue
                    delta_T = Tcat[mask][-1] - Tcat[mask][0]
                    delta_t = tv[mask][-1] - tv[mask][0]
                    if delta_t == 0:
                        rates.append(np.nan)
                    else:
                        rates.append(delta_T / delta_t)
                avg_dTcat_dt = np.array(rates)
                if ramp_direction == 'down':
                    extreme_rate = float(np.nanmin(avg_dTcat_dt))
                    extreme_idx = int(np.nanargmin(avg_dTcat_dt))
                else:
                    extreme_rate = float(np.nanmax(avg_dTcat_dt))
                    extreme_idx = int(np.nanargmax(avg_dTcat_dt))
                derived[f'dTcat_dt_extreme_{label}'] = avg_dTcat_dt
                derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
                derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx
        elif isinstance(window, tuple) and len(window) == 2:
            # Use the full window from start to end (not sliding)
            t0, t1 = window
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
                    if delta_t == 0:
                        rates.append(np.nan)
                    else:
                        rates.append(delta_T / delta_t)
                rates = np.array(rates)
                if ramp_direction == 'down':
                    extreme_rate = float(np.nanmin(rates))
                    extreme_idx = int(np.nanargmin(rates))
                else:
                    extreme_rate = float(np.nanmax(rates))
                    extreme_idx = int(np.nanargmax(rates))
                derived[f'dTcat_dt_extreme_{label}'] = rates
                derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
                derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx
            else:
                arr = Tcat[mask]
                tvec = tv[mask]
                if len(arr) < 2:
                    rate = np.nan
                else:
                    delta_T = arr[-1] - arr[0]
                    delta_t = tvec[-1] - tvec[0]
                    if delta_t == 0:
                        rate = np.nan
                    else:
                        rate = delta_T / delta_t
                if ramp_direction == 'down':
                    extreme_rate = float(rate) if not np.isnan(rate) else np.nan
                    extreme_idx = 0
                else:
                    extreme_rate = float(rate) if not np.isnan(rate) else np.nan
                    extreme_idx = 0
                derived[f'dTcat_dt_extreme_{label}'] = rate
                derived[f'dTcat_dt_extreme_{label}_extreme'] = extreme_rate
                derived[f'dTcat_dt_extreme_{label}_extreme_idx'] = extreme_idx
    return derived


def calculate_metrics(core: Dict[str, Any], derived: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    """
    Calculate steady-state and average ramp slope metrics with numeric formatting.
    """
    tv = core['time_vector']
    dTdt = derived['dTcat_dt']

    rp = core['metadata'].get('ramp_parameters', {})
    ramp_start = rp.get('start_time')
    duration = rp.get('duration')
    if ramp_start is None or duration is None:
        raise KeyError("Ramp parameters 'start_time' and 'duration' must be in metadata.ramp_parameters")
    ramp_end = ramp_start + duration

    idx0 = int(np.searchsorted(tv, ramp_end))
    segment = dTdt[idx0:]
    if segment.ndim == 2:
        rms = np.sqrt(np.mean(segment**2, axis=1))
    else:
        rms = np.sqrt(segment**2)

    dt = np.diff(tv, prepend=tv[0])
    run_time = 0.0
    steady_time = None
    for i, val in enumerate(rms, start=idx0):
        if val <= cfg.steady_threshold:
            run_time += dt[i]
            if run_time >= cfg.steady_min_duration:
                steady_time = tv[i]
                break
        else:
            run_time = 0.0
    if steady_time is None:
        raise RuntimeError("Steady state not detected")

    mask = (tv >= ramp_start) & (tv <= ramp_end)
    if dTdt.ndim == 2:
        avg_slope = float(np.mean(dTdt[mask, :]))
    else:
        avg_slope = float(np.mean(dTdt[mask]))

    # Use ramp direction to select which extreme to report for dTcat/dt
    ramp_direction = derived.get('ramp_direction', 'unknown')
    if ramp_direction == 'down':
        dTcat_dt_extreme = float(f"{np.nanmin(derived['dTcat_dt_extreme_1min']):.5g}") if isinstance(derived['dTcat_dt_extreme_1min'], np.ndarray) else float(f"{derived['dTcat_dt_extreme_1min']:.5g}")
    else:
        dTcat_dt_extreme = float(f"{np.nanmax(derived['dTcat_dt_extreme_1min']):.5g}") if isinstance(derived['dTcat_dt_extreme_1min'], np.ndarray) else float(f"{derived['dTcat_dt_extreme_1min']:.5g}")

    metrics = {
        'Tcat_max': float(f"{derived['Tcat_max']:.5g}"),
        'Tcat_min': float(f"{derived['Tcat_min']:.5g}"),
        'steady_time': float(f"{steady_time:.5g}"),
        'avg_dTdt_ramp': float(f"{avg_slope:.5g}"),
        'dTcat_dt_extreme': dTcat_dt_extreme,
        'ramp_direction': ramp_direction
    }
    return metrics


def generate_metadata(core: Dict[str, Any], metrics: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    fp = core['metadata'].get('file_path', '')
    fname = os.path.basename(fp)
    ramp_direction = 'unknown'
    # Extract ramp direction from filename after first hyphen
    if '-' in fname:
        after_hyphen = fname.split('-', 1)[1].lower()
        if after_hyphen.startswith('up') or '-up' in after_hyphen:
            ramp_direction = 'up'
        elif after_hyphen.startswith('down') or '-down' in after_hyphen:
            ramp_direction = 'down'
        else:
            # Try to find 'up' or 'down' as a word after hyphen
            if 'up' in after_hyphen:
                ramp_direction = 'up'
            elif 'down' in after_hyphen:
                ramp_direction = 'down'
    return {
        'source_file': fname,
        'cfg': vars(cfg),
        'metrics': metrics,
        'ramp_direction': ramp_direction
    }


def run_analysis(file_path: str, cfg: AnalysisConfig):
    raw = load_data(file_path)
    core = process_core(raw, cfg)

    # --- Calculate steady_time before compute_derived ---
    tv = core['time_vector']
    dTcat_dt = None
    try:
        if 'Tcat' not in core['variables']:
            raise KeyError("Catalyst temperature 'Tcat' not found in variables")
        Tcat = core['variables']['Tcat']
        dTcat_dt = np.gradient(Tcat, tv, axis=0)
    except Exception:
        dTcat_dt = None
    steady_time = None
    rp = core['metadata'].get('ramp_parameters', {})
    ramp_start = rp.get('start_time')
    duration = rp.get('duration')
    ramp_end = ramp_start + duration if ramp_start is not None and duration is not None else None
    if dTcat_dt is not None and ramp_end is not None:
        idx0 = int(np.searchsorted(tv, ramp_end))
        segment = dTcat_dt[idx0:]
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
                    steady_time = tv[i]
                    break
            else:
                run_time = 0.0

    # --- Pass steady_time to compute_derived ---
    derived = compute_derived(core, steady_time=steady_time)
    metrics = calculate_metrics(core, derived, cfg)
    metadata = generate_metadata(core, metrics, cfg)

    # Merge all variables except metadata into a single dict
    result = {}
    # Add core variables
    result.update({
        'time_vector': core['time_vector'],
        'length_vector': core['length_vector'],
        'variables': core['variables'],
    })
    # Add derived and metrics (flattened)
    for k, v in derived.items():
        result[k] = v
    for k, v in metrics.items():
        result[k] = v

    return {
        'data': result,
        'metadata': metadata
    }
