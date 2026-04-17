"""survpfn.models.cr_models.classical — Classical Competing Risks models."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, AalenJohansenFitter

def run_competing_risks_cox(df_train, df_test, duration_col, event_col, penalizer=0.1, **kwargs):
    # Dynamically find event values (excluding 0 for censored)
    all_events = np.unique(df_train[event_col].values)
    causes = [c for c in all_events if c > 0]
    
    if len(causes) == 0:
        # Fallback if no events
        causes = [1] 
    
    # Simple strategy: fit a Cause-Specific Cox model for the first cause
    event1_val = causes[0]
    
    # Model for event 1
    df_temp1_train = df_train.copy()
    df_temp1_train[event_col] = (df_temp1_train[event_col] == event1_val).astype(int)
    
    df_temp1_test = df_test.copy()
    df_temp1_test[event_col] = (df_temp1_test[event_col] == event1_val).astype(int)

    # Drop constant columns (common source of Matrix is singular)
    non_const = [c for c in df_temp1_train.columns 
                 if c in {duration_col, event_col} or df_temp1_train[c].nunique() > 1]
    df_temp1_train = df_temp1_train[non_const]
    df_temp1_test = df_temp1_test[[c for c in df_temp1_test.columns if c in non_const]]
    
    cph1 = CoxPHFitter(penalizer=penalizer)
    try:
        cph1.fit(df_temp1_train, duration_col=duration_col, event_col=event_col)
    except Exception:
        # Retry with a larger penalizer if it fails
        cph1 = CoxPHFitter(penalizer=max(penalizer * 10, 1.0))
        cph1.fit(df_temp1_train, duration_col=duration_col, event_col=event_col)

    c_index1 = cph1.score(df_temp1_test, scoring_method="concordance_index")
    
    # For compatibility with DeepHit output format: (model, risk_p1, cif_per_cause, grid)
    grid = np.linspace(df_train[duration_col].min() + 1e-6, df_train[duration_col].max() - 1e-6, 50)
    # Simple CIF estimate for cause 1: 1 - S(t)
    feats = [c for c in df_test.columns if c in non_const and c not in {duration_col, event_col}]
    surv1 = cph1.predict_survival_function(df_test[feats], times=grid)
    cif_p1 = 1.0 - surv1.values.T  # (n_samples, n_times)
    cif_per_cause = [cif_p1]
    
    # Return lead CPH, its summary, and the structured metrics
    return cph1, cif_p1, cif_per_cause, grid

def run_aalen_johansen(df_train, df_test, duration_col, event_col, **kwargs):
    """
    Aalen-Johansen non-parametric estimator for the Cumulative Incidence Function (CIF).
    Returns the same baseline CIF for all test samples since it doesn't use features.
    """
    all_events = np.unique(df_train[event_col].values)
    causes = sorted([c for c in all_events if c > 0])
    if len(causes) == 0:
        causes = [1]
        
    # Generate prediction grid
    grid = np.linspace(df_train[duration_col].min() + 1e-6, df_train[duration_col].max() - 1e-6, 50)
    n_samples = len(df_test)
    cif_per_cause = []

    # In lifelines 0.30.0, AJFitter calculates one cause's CIF at a time
    for cause in causes:
        aj_c = AalenJohansenFitter(calculate_variance=False)
        aj_c.fit(df_train[duration_col], df_train[event_col], event_of_interest=cause)
        
        # Reindex the cumulative_density_ to our comparison grid
        # lifelines uses 'cumulative_density_' attribute in 0.30.0
        cif_at_grid = aj_c.cumulative_density_.reindex(grid, method='ffill').fillna(0).values.flatten()
        
        # Broadcast to all samples
        cif_per_cause.append(np.tile(cif_at_grid, (n_samples, 1)))
        
    aj = None # We used multiple fitters
        
    # Calculate integrated risk for cause 1
    span = grid[-1] - grid[0] + 1e-8
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk_cause1 = _trapz(cif_per_cause[0], grid, axis=1) / span
    
    return aj, risk_cause1, cif_per_cause, grid

def run_fine_gray(df_train, df_test, duration_col, event_col, penalizer=0.1, **kwargs):
    """
    Fine-Gray subdistribution hazard model approximation.
    For this implementation, we utilize Cause-Specific Cox models as a proxy for 
    relative risks and CIF estimation in the presence of competing events.
    """
    all_events = np.unique(df_train[event_col].values)
    causes = sorted([c for c in all_events if c > 0])
    if len(causes) == 0:
        causes = [1]
        
    grid = np.linspace(df_train[duration_col].min() + 1e-6, df_train[duration_col].max() - 1e-6, 50)
    models = []
    cif_per_cause = []
    
    # We fit a separate Cause-Specific Cox for each cause
    for cause in causes:
        df_c = df_train.copy()
        # Mark other causes as censored for Cause-Specific Hazard
        df_c[event_col] = (df_c[event_col] == cause).astype(int)
        
        # Filter constant features
        non_const = [c for c in df_c.columns if c in {duration_col, event_col} or df_c[c].nunique() > 1]
        df_c_train = df_c[non_const]
        
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(df_c_train, duration_col=duration_col, event_col=event_col)
        except Exception:
            cph = CoxPHFitter(penalizer=max(penalizer * 10, 1.0))
            cph.fit(df_c_train, duration_col=duration_col, event_col=event_col)
            
        feats = [c for c in df_test.columns if c in non_const and c not in {duration_col, event_col}]
        # Risk estimation: 1 - Survival
        surv = cph.predict_survival_function(df_test[feats], times=grid)
        cif_per_cause.append(1.0 - surv.values.T)
        models.append(cph)
        
    # Calculate integrated risk for cause 1
    span = grid[-1] - grid[0] + 1e-8
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk_cause1 = _trapz(cif_per_cause[0], grid, axis=1) / span
    
    # Return cause 1 model as primary
    return models[0], risk_cause1, cif_per_cause, grid


def run_survival_boost_cr(df_train, df_test, duration_col, event_col,
                          n_estimators=100, learning_rate=0.1, max_depth=3,
                          **kwargs):
    """Cause-specific GradientBoosting competing risks (via sksurv).

    Fits one ``GradientBoostingSurvivalAnalysis`` model per cause (competing
    events treated as censored for each cause's model), then converts
    cause-specific survival functions into CIFs via 1 - S_c(t).

    Falls back to ``run_fine_gray`` if sksurv is unavailable.
    """
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    except ImportError:
        import warnings
        warnings.warn(
            "sksurv not available for survival_boost_cr; falling back to fine_gray_cr.",
            RuntimeWarning,
        )
        return run_fine_gray(df_train, df_test, duration_col, event_col, **kwargs)

    all_events = np.unique(df_train[event_col].values)
    causes = sorted([c for c in all_events if c > 0])
    if len(causes) == 0:
        causes = [1]

    grid = np.linspace(
        df_train[duration_col].min() + 1e-6,
        df_train[duration_col].max() - 1e-6,
        50,
    )
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train = df_train[feat_cols].values.astype(np.float32)
    X_test  = df_test[feat_cols].values.astype(np.float32)
    T_train = df_train[duration_col].values.astype(float)

    models = []
    cif_per_cause = []

    for cause in causes:
        # Mark other causes as censored for this cause's model
        E_c = (df_train[event_col].values == cause).astype(bool)
        y_c = np.array(
            [(bool(e), float(t)) for e, t in zip(E_c, T_train)],
            dtype=[("event", bool), ("time", float)],
        )

        gbm = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        gbm.fit(X_train, y_c)
        models.append(gbm)

        # Predict survival at each grid point then convert to CIF
        surv_fns = gbm.predict_survival_function(X_test)
        cif_c = np.stack(
            [1.0 - fn(grid) for fn in surv_fns],
            axis=0,
        )  # (n_test, n_grid)
        cif_per_cause.append(cif_c)

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    span = grid[-1] - grid[0] + 1e-8
    risk_cause1 = _trapz(cif_per_cause[0], grid, axis=1) / span

    return models[0], risk_cause1, cif_per_cause, grid

