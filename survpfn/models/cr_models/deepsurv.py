"""survpfn.models.cr_models.deepsurv — Cause-Specific DeepSurv for Competing Risks."""

import numpy as np
import pandas as pd
from typing import Optional
from survpfn.models.sr_models.deepsurv import train_deepsurv

def train_deepsurv_cr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str = "Follow Up Data",
    event_col: str = "Total mortality",
    random_state: int = 42,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    study_id: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> tuple:
    """Cause-specific DeepSurv model for competing risks.
    
    Fits one DeepSurv model per cause, treating competing events as censored.
    Returns the primary model, integrated risk for cause 1, CIFs for all causes,
    and the time grid.
    """
    all_events = np.unique(df_train[event_col].values)
    causes = sorted([c for c in all_events if c > 0])
    if len(causes) == 0:
        causes = [1]
        
    models = []
    cif_per_cause = []
    grid = None
    
    for cause in causes:
        if verbose:
            print(f"  [DeepSurv-CR] Training Cause-Specific Model for Event {cause}", flush=True)
            
        df_train_c = df_train.copy()
        df_train_c[event_col] = (df_train_c[event_col] == cause).astype(int)
        
        df_test_c = df_test.copy()
        df_test_c[event_col] = (df_test_c[event_col] == cause).astype(int)
        
        study_id_cause = f"{study_id}_cause{cause}" if study_id else f"cause{cause}"
        
        # Train DeepSurv for this cause
        model, risk_scores, surv_probs, surv_times = train_deepsurv(
            df_train_c, df_test_c,
            duration_col=duration_col, event_col=event_col,
            random_state=random_state, tune=tune, n_trials=n_trials,
            out_dir=out_dir, study_id=study_id_cause,
            verbose=verbose,
            **kwargs
        )
        
        models.append(model)
        if grid is None:
            grid = surv_times
            cif_per_cause.append(1.0 - surv_probs)
        else:
            from scipy.interpolate import interp1d
            f = interp1d(surv_times, 1.0 - surv_probs, kind='linear', axis=1, fill_value="extrapolate")
            cif_interp = np.clip(f(grid), 0.0, 1.0)
            cif_per_cause.append(cif_interp)
            
    # Calculate integrated risk for cause 1
    span = grid[-1] - grid[0] + 1e-8
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk_cause1 = _trapz(cif_per_cause[0], grid, axis=1) / span
    
    return models[0], risk_cause1, cif_per_cause, grid
