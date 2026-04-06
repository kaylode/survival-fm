import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

def get_time_bins(durations: np.ndarray, events: np.ndarray, n_bins: int, scheme: str = 'quantiles') -> np.ndarray:
    """
    Generate time bin boundaries based on observed event times.
    
    Parameters
    ----------
    durations : np.ndarray
        Survival times.
    events : np.ndarray
        Event indicators (1 for event, 0 for censored).
    n_bins : int
        Number of bins.
    scheme : str
        'quantiles' or 'equidistant'.
        
    Returns
    -------
    np.ndarray
        Sorted bin boundaries.
    """
    event_times = durations[events > 0]
    if len(event_times) == 0:
        return np.linspace(durations.min(), durations.max(), n_bins)
    
    if scheme == 'quantiles':
        # Exclude 0 and 100 to get n_bins internal boundaries if possible, 
        # but usually we want n_bins intervals.
        # pycox LabTransDiscreteTime uses n_bins intervals.
        # For zero-shot, we want n_bins discrete time points to evaluate at.
        probs = np.linspace(0, 100, n_bins + 1)[1:] # pycox style captures the right edge of bins
        bins = np.unique(np.percentile(event_times, probs))
    elif scheme == 'equidistant':
        bins = np.linspace(event_times.min(), event_times.max(), n_bins)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    return bins

def discretize_times(durations: np.ndarray, events: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous times into bin indices.
    """
    bins = get_time_bins(durations, events, n_bins)
    # findnearest bin index
    # We want index k such that bin[k-1] < t <= bin[k]
    # np.digitize(t, bins) returns i such that bins[i-1] <= t < bins[i]
    # For survival, we usually want t_idx such that t >= bins[t_idx] is survived? 
    # Let's align with pycox LabTransDiscreteTime.
    # Actually, let's just use pycox's LabTransDiscreteTime when possible, 
    # but provide this for non-pycox models or zero-shot.
    return bins, np.digitize(durations, bins, right=True)
