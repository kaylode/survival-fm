"""survpfn.models.cr_models.classical — Classical Competing Risks models."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def run_competing_risks_cox(df_train, df_test, duration_col, event_col, penalizer=0.1):
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
    
    # Summary of coefficients
    importance = cph1.summary.reset_index()
    importance.rename(columns={'index': 'feature'}, inplace=True)
    
    # For now return the lead CPH and its summary
    return cph1, importance, importance, c_index1
