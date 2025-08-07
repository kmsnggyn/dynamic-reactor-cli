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
    ramp_start: Optional[float] = None        # ramp start time in minutes
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
        meta = dict(data_pkg.metadata)
    meta['file_path'] = file_path

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


def compute_derived(core: Dict[str, Any]) -> Dict[str, Any]:
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

    # Format numbers with up to 5 significant figures
    metrics = {
        'Tcat_max': float(f"{derived['Tcat_max']:.5g}"),
        'Tcat_min': float(f"{derived['Tcat_min']:.5g}"),
        'steady_time': float(f"{steady_time:.5g}"),
        'avg_dT_ramp': float(f"{avg_slope:.5g}")
    }
    return metrics


def generate_metadata(core: Dict[str, Any], metrics: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    fp = core['metadata'].get('file_path', '')
    return {
        'source_file': os.path.basename(fp),
        'cfg': vars(cfg),
        'metrics': metrics
    }


def run_analysis(file_path: str, cfg: AnalysisConfig) -> AnalysisPackage:
    raw = load_data(file_path)
    core = process_core(raw, cfg)
    derived = compute_derived(core)
    metrics = calculate_metrics(core, derived, cfg)
    metadata = generate_metadata(core, metrics, cfg)
    return AnalysisPackage(
        time_vector=core['time_vector'],
        length_vector=core['length_vector'],
        variables=core['variables'],
        derived=derived,
        metrics=metrics,
        metadata=metadata
    )
