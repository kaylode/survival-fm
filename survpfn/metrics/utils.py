"""Shared utilities for survival evaluation metrics."""

import numpy as np

def _get_survival_array(y, duration_col, event_col):
    """Convert pandas dataframe/series into structured array for sksurv."""
    return np.array(
        list(zip(y[event_col].astype(bool), y[duration_col])),
        dtype=[('event', '?'), ('time', '<f8')]
    )


def _safe_time_grid(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_points: int = 50,
) -> np.ndarray:
    # For IPCW (used in IBS and AUC), the censoring survival function G(t)
    # cannot be zero at any evaluation time. The KM estimator for G(t) drops
    # to zero at the largest observed time if that observation is censored.
    # To be unconditionally safe, we cap `hi` slightly below the maximum time 
    # of the training set, but more importantly, below the max observed event
    # time if that's safer, though sksurv primarily checks the global max.
    
    train_max = float(y_train["time"].max())
    test_max = float(y_test["time"].max())
    
    # We find the max event time in train data to be extra safe
    train_events = y_train["time"][y_train["event"]]
    train_event_max = float(train_events.max()) if len(train_events) > 0 else train_max
    
    # The absolute ceiling is the smallest of:
    # 1. train_max
    # 2. test_max
    # We apply a small margin to this ceiling.
    lo = max(float(y_train["time"].min()), float(y_test["time"].min()))
    hi = min(train_max, test_max)
    
    if lo >= hi:
        return np.array([])
        
    margin = (hi - lo) * 0.05  # 5% margin to aggressively avoid boundary issues
    hi_safe = min(hi - margin, train_event_max - 1e-5)
    
    if lo + margin >= hi_safe:
        # Fallback if margin is too large
        margin = (hi - lo) * 0.001
        hi_safe = min(hi - margin, train_event_max - 1e-5)
        if lo + margin >= hi_safe:
            return np.array([])
            
    grid = np.linspace(lo + margin, hi_safe, n_points)
    
    # Check if grid points violate 0 < G(t) and < test_max
    grid = grid[grid < test_max - 1e-4]
    
    return grid

def _filter_ipcw_test(y_train: np.ndarray, y_test: np.ndarray, *arrays):
    """
    Filter y_test and associated predictions to strictly remain within the 
    estimable censoring support of y_train. 
    Required for sksurv IPCW calculations which crash if any test observation 
    exceeds the max time in the training data, or falls into G(t)=0 area.
    """
    if len(y_train) == 0 or len(y_test) == 0:
        return (y_test,) + arrays

    # Max allowable time is the max time in y_train
    t_max = y_train["time"].max()
    
    # Check if KM censoring estimate drops to 0 at the upper boundary
    extends_mask = y_train["time"] == t_max
    # If the absolute max time patient is right-censored (~event), G(T_max) = 0 exactly.
    # Therefore test times cannot be >= t_max.
    if np.any(~y_train["event"][extends_mask]):
        t_max -= 1e-7

    mask = y_test["time"] <= t_max

    if mask.all():
        return (y_test,) + arrays
    elif not mask.any():
        return (y_test[mask],) + tuple(a[mask] for a in arrays)
        
    return (y_test[mask],) + tuple(a[mask] for a in arrays)
