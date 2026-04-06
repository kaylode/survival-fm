"""survpfn.models — survival model implementations."""

from typing import Callable, Union, List, Optional
import numpy as np
import pandas as pd
import warnings

# --- shared ---
from .shared.heads import train_fm_embedding_surv
from .shared.zeroshot_surv import train_zeroshot_surv

# --- sr_models ---
from .sr_models.classical import (
    run_univariate_cox,
    run_multivariate_cox,
    run_kaplan_meier,
)
from .sr_models.tree import train_rsf, train_gbsa
from .sr_models.deep_surv import train_deepsurv
from .sr_models.deep_hit import (
    train_mtlr,
    train_pchazard,
    train_deephit_single,
)
from .sr_models.custom import train_custom_torchsurv
from .sr_models.soden import train_soden
from .sr_models.beta_surv import train_beta_surv
from .sr_models.survtrace import train_survtrace

# --- cr_models ---
from .cr_models.classical import run_competing_risks_cox
from .cr_models.deephit import train_deephit_cr

# --- foundation models ---
from .tabpfn import train_tabpfn_embedding_surv, TabPFNSurvPH
from .tabdpt import train_tabdpt_embedding_surv, TabDPTSurvPH
from .tabicl import train_tabicl_embedding_surv, TabICLSurvPH


# ---------------------------------------------------------------------------
# Helpers for ALL_MODELS registry
# ---------------------------------------------------------------------------

def _make_fm_wrapper(fm_name: str, head_type: str, task_type: str = "sr", fixed_cr_loss: str = None, fixed_use_adapter: Optional[bool] = None) -> Callable:
    def _fn(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **kw):
        # Extract params, popping to avoid double-passing.
        cr_loss_type = fixed_cr_loss or kw.pop("cr_loss_type", "deephit")
        use_adapter = fixed_use_adapter if fixed_use_adapter is not None else kw.pop("use_adapter", False)
        
        # Ensure we don't have duplicates if they were still in kw
        kw.pop("cr_loss_type", None)
        kw.pop("use_adapter", None)

        if fm_name == "tabpfn":
            return train_tabpfn_embedding_surv(df_train, df_test, dur_col, ev_col, 
                                             head_type=head_type, tune=tune, n_trials=n_trials, 
                                             save_dir=out_dir, task_type=task_type, 
                                             cr_loss_type=cr_loss_type, use_adapter=use_adapter, **kw)
        if fm_name == "tabdpt":
            return train_tabdpt_embedding_surv(df_train, df_test, dur_col, ev_col, 
                                             head_type=head_type, tune=tune, n_trials=n_trials, 
                                             save_dir=out_dir, task_type=task_type, 
                                             cr_loss_type=cr_loss_type, use_adapter=use_adapter, **kw)
        if fm_name == "tabicl":
            return train_tabicl_embedding_surv(df_train, df_test, dur_col, ev_col, 
                                             head_type=head_type, tune=tune, n_trials=n_trials, 
                                             save_dir=out_dir, task_type=task_type, 
                                             cr_loss_type=cr_loss_type, use_adapter=use_adapter, **kw)
    _fn.__name__ = f"train_{fm_name}_{head_type}_{task_type}"
    return _fn

def _make_zeroshot_wrapper(backbone: str, method: str = "single_context", cr_method: str = "multinomial") -> Callable:
    def _fn(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **kw):
        num_events = int(df_train[ev_col].max())
        return train_zeroshot_surv(
            df_train, df_test, dur_col, ev_col,
            backbone=backbone,
            method=method,
            cr_method=cr_method if num_events > 1 else "multinomial", # binary fallback
            **kw
        )
    return _fn

def _make_joint_fm_wrapper(fm_name: str, head_type: str, task_type: str = "sr", fixed_cr_loss: str = None, fixed_use_adapter: Optional[bool] = None) -> Callable:
    def _fn(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **kw):
        num_events = int(df_train[ev_col].max())
        # Extract training params from kw with defaults, popping them to avoid double-passing.
        epochs = kw.pop("epochs", 100)
        batch_size = kw.pop("batch_size", 128)
        device = kw.pop("device", "cuda:0")
        cr_loss_type = fixed_cr_loss or kw.pop("cr_loss_type", "deephit")
        use_adapter = fixed_use_adapter if fixed_use_adapter is not None else kw.pop("use_adapter", False)

        # Cleanup duplicates
        kw.pop("cr_loss_type", None)
        kw.pop("use_adapter", None)

        # Feature dimension for adapter
        feat_cols = [c for c in df_train.columns if c not in {dur_col, ev_col}]
        input_dim = len(feat_cols)

        if fm_name == "tabpfn":
            from .tabpfn import TabPFNSurvPH
            wrapper = TabPFNSurvPH(
                head_type=head_type, 
                task_type=task_type, 
                num_events=num_events, 
                cr_loss_type=cr_loss_type,
                use_adapter=use_adapter,
                input_dim=input_dim,
                device=device,
                **kw
            )
        elif fm_name == "tabdpt":
            from .tabdpt import TabDPTSurvPH
            wrapper = TabDPTSurvPH(
                head_type=head_type, 
                task_type=task_type, 
                num_events=num_events, 
                cr_loss_type=cr_loss_type,
                use_adapter=use_adapter,
                input_dim=input_dim,
                device=device,
                **kw
            )
        elif fm_name == "tabicl":
            from .tabicl import TabICLSurvPH
            wrapper = TabICLSurvPH(
                head_type=head_type, 
                task_type=task_type, 
                num_events=num_events, 
                cr_loss_type=cr_loss_type,
                use_adapter=use_adapter,
                input_dim=input_dim,
                device=device,
                **kw
            )
        
        # Match common API: fit(x, dur, ev)
        x_train = df_train[feat_cols].values.astype(np.float32)
        durations = df_train[dur_col].values
        events = df_train[ev_col].values
        
        training_kwargs = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=kw.get("verbose", True),
        )

        wrapper.fit(x_train, durations, events, **training_kwargs)
        
        # predict_survival_df returns (times, samples)
        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_df = wrapper.predict_survival_df(x_test)
        
        if task_type == "cr":
            # For CR, surv_df is a List[pd.DataFrame]
            cifs = surv_df
            risk = 1.0 - cifs[0].iloc[-1].values # Risk of cause 1
            return wrapper, risk, [c.values.T for c in cifs], cifs[0].index.values
        else:
            risk = 1.0 - surv_df.iloc[-1].values
            return wrapper, risk, surv_df.values.T, surv_df.index.values
    return _fn

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, Callable] = {
    # ── Single Risk (SR) ────────────────────────────────────────────────────
    "cox": lambda df_tr, df_ts, dur, ev, **kw: (
        lambda m, s, p, c: (m, m.predict_partial_hazard(df_ts.drop(columns=[dur, ev])).values, 
                           m.predict_survival_function(df_ts.drop(columns=[dur, ev])).values.T, 
                           m.predict_survival_function(df_ts.drop(columns=[dur, ev])).index.values)
    )(*run_multivariate_cox(df_tr, df_ts, dur, ev)),
    "km": lambda df_tr, df_ts, dur, ev, **kw: (
        lambda t, p, df: (None, np.zeros(len(df_ts)), np.tile(p, (len(df_ts), 1)), t)
    )(*run_kaplan_meier(df_ts, dur, ev)),
    "rsf": lambda df_tr, df_ts, dur, ev, **kw: train_rsf(df_tr, df_ts, dur, ev, **kw),
    "gbsa": lambda df_tr, df_ts, dur, ev, **kw: train_gbsa(df_tr, df_ts, dur, ev, **kw),
    "deepsurv": lambda df_tr, df_ts, dur, ev, **kw: train_deepsurv(df_tr, df_ts, dur, ev, **kw),
    "mtlr": lambda df_tr, df_ts, dur, ev, **kw: train_mtlr(df_tr, df_ts, dur, ev, **kw),
    "pchazard": lambda df_tr, df_ts, dur, ev, **kw: train_pchazard(df_tr, df_ts, dur, ev, **kw),
    "deephit_single": lambda df_tr, df_ts, dur, ev, **kw: train_deephit_single(df_tr, df_ts, dur, ev, **kw),
    "survtrace": lambda df_tr, df_ts, dur, ev, **kw: train_survtrace(df_tr, df_ts, dur, ev, **kw),
    "soden": lambda df_tr, df_ts, dur, ev, **kw: train_soden(df_tr, df_ts, dur, ev, **kw),
    "beta_surv": lambda df_tr, df_ts, dur, ev, **kw: train_beta_surv(df_tr, df_ts, dur, ev, **kw),
    
    # Strategy 1: Embedding (SR)
    "tabpfn_embedding_cox":             _make_fm_wrapper("tabpfn", "cox"),
    "tabpfn_embedding_cox_adapter":      _make_fm_wrapper("tabpfn", "cox", fixed_use_adapter=True),
    "tabpfn_embedding_deephit":         _make_fm_wrapper("tabpfn", "deephit"),
    "tabpfn_embedding_deephit_adapter":   _make_fm_wrapper("tabpfn", "deephit", fixed_use_adapter=True),
    "tabpfn_embedding_pchazard":        _make_fm_wrapper("tabpfn", "pchazard"),
    "tabpfn_embedding_pchazard_adapter":  _make_fm_wrapper("tabpfn", "pchazard", fixed_use_adapter=True),
    "tabpfn_embedding_mtlr":            _make_fm_wrapper("tabpfn", "mtlr"),
    "tabpfn_embedding_mtlr_adapter":      _make_fm_wrapper("tabpfn", "mtlr", fixed_use_adapter=True),
    
    "tabdpt_embedding_cox":             _make_fm_wrapper("tabdpt", "cox"),
    "tabdpt_embedding_cox_adapter":      _make_fm_wrapper("tabdpt", "cox", fixed_use_adapter=True),
    "tabdpt_embedding_deephit":         _make_fm_wrapper("tabdpt", "deephit"),
    "tabdpt_embedding_deephit_adapter":   _make_fm_wrapper("tabdpt", "deephit", fixed_use_adapter=True),
    "tabdpt_embedding_pchazard":        _make_fm_wrapper("tabdpt", "pchazard"),
    "tabdpt_embedding_pchazard_adapter":  _make_fm_wrapper("tabdpt", "pchazard", fixed_use_adapter=True),
    "tabdpt_embedding_mtlr":            _make_fm_wrapper("tabdpt", "mtlr"),
    "tabdpt_embedding_mtlr_adapter":      _make_fm_wrapper("tabdpt", "mtlr", fixed_use_adapter=True),

    "tabicl_embedding_cox":             _make_fm_wrapper("tabicl", "cox"),
    "tabicl_embedding_cox_adapter":      _make_fm_wrapper("tabicl", "cox", fixed_use_adapter=True),
    "tabicl_embedding_deephit":         _make_fm_wrapper("tabicl", "deephit"),
    "tabicl_embedding_deephit_adapter":   _make_fm_wrapper("tabicl", "deephit", fixed_use_adapter=True),
    "tabicl_embedding_pchazard":        _make_fm_wrapper("tabicl", "pchazard"),
    "tabicl_embedding_pchazard_adapter":  _make_fm_wrapper("tabicl", "pchazard", fixed_use_adapter=True),
    "tabicl_embedding_mtlr":            _make_fm_wrapper("tabicl", "mtlr"),
    "tabicl_embedding_mtlr_adapter":      _make_fm_wrapper("tabicl", "mtlr", fixed_use_adapter=True),
    
    # Strategy 2: Joint Training (SR)
    "tabpfn_joint_cox":             _make_joint_fm_wrapper("tabpfn", "cox"),
    "tabpfn_joint_cox_adapter":      _make_joint_fm_wrapper("tabpfn", "cox", fixed_use_adapter=True),
    "tabpfn_joint_deephit":         _make_joint_fm_wrapper("tabpfn", "deephit"),
    "tabpfn_joint_deephit_adapter":   _make_joint_fm_wrapper("tabpfn", "deephit", fixed_use_adapter=True),
    "tabpfn_joint_pchazard":        _make_joint_fm_wrapper("tabpfn", "pchazard"),
    "tabpfn_joint_pchazard_adapter":  _make_joint_fm_wrapper("tabpfn", "pchazard", fixed_use_adapter=True),
    "tabpfn_joint_mtlr":            _make_joint_fm_wrapper("tabpfn", "mtlr"),
    "tabpfn_joint_mtlr_adapter":      _make_joint_fm_wrapper("tabpfn", "mtlr", fixed_use_adapter=True),
 
    "tabdpt_joint_cox":             _make_joint_fm_wrapper("tabdpt", "cox"),
    "tabdpt_joint_cox_adapter":      _make_joint_fm_wrapper("tabdpt", "cox", fixed_use_adapter=True),
    "tabdpt_joint_deephit":         _make_joint_fm_wrapper("tabdpt", "deephit"),
    "tabdpt_joint_deephit_adapter":   _make_joint_fm_wrapper("tabdpt", "deephit", fixed_use_adapter=True),
    "tabdpt_joint_pchazard":        _make_joint_fm_wrapper("tabdpt", "pchazard"),
    "tabdpt_joint_pchazard_adapter":  _make_joint_fm_wrapper("tabdpt", "pchazard", fixed_use_adapter=True),
    "tabdpt_joint_mtlr":            _make_joint_fm_wrapper("tabdpt", "mtlr"),
    "tabdpt_joint_mtlr_adapter":      _make_joint_fm_wrapper("tabdpt", "mtlr", fixed_use_adapter=True),
 
    "tabicl_joint_cox":             _make_joint_fm_wrapper("tabicl", "cox"),
    "tabicl_joint_cox_adapter":      _make_joint_fm_wrapper("tabicl", "cox", fixed_use_adapter=True),
    "tabicl_joint_deephit":         _make_joint_fm_wrapper("tabicl", "deephit"),
    "tabicl_joint_deephit_adapter":   _make_joint_fm_wrapper("tabicl", "deephit", fixed_use_adapter=True),
    "tabicl_joint_pchazard":        _make_joint_fm_wrapper("tabicl", "pchazard"),
    "tabicl_joint_pchazard_adapter":  _make_joint_fm_wrapper("tabicl", "pchazard", fixed_use_adapter=True),
    "tabicl_joint_mtlr":            _make_joint_fm_wrapper("tabicl", "mtlr"),
    "tabicl_joint_mtlr_adapter":      _make_joint_fm_wrapper("tabicl", "mtlr", fixed_use_adapter=True),
    
    # Strategy 3: Zero-shot (SR)
    "tabpfn_zeroshot": _make_zeroshot_wrapper("tabpfn"),
    "tabdpt_zeroshot": _make_zeroshot_wrapper("tabdpt"),
    "tabicl_zeroshot": _make_zeroshot_wrapper("tabicl"),

    # ── Competing Risks (CR) ────────────────────────────────────────────────
    "cox_cr": lambda df_tr, df_ts, dur, ev, **kw: (
        lambda m, s, p, c: (m, m.predict_partial_hazard(df_ts.drop(columns=[dur, ev])).values, 
                           [m.predict_survival_function(df_ts.drop(columns=[dur, ev])).values.T], 
                           m.predict_survival_function(df_ts.drop(columns=[dur, ev])).index.values)
    )(*run_competing_risks_cox(df_tr, df_ts, dur, ev)),
    "deephit_cr": lambda df_tr, df_ts, dur, ev, **kw: train_deephit_cr(df_tr, df_ts, dur, ev, **kw),
    
    # Strategy 1: Embedding (CR)
    "tabpfn_embedding_deephit_cr":          _make_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabpfn_embedding_deephit_v2_cr":       _make_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabpfn_embedding_deephit_v2_cr_adapter": _make_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabpfn_embedding_cox_cr":              _make_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabpfn_embedding_cox_cr_adapter":        _make_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),
    
    "tabdpt_embedding_deephit_cr":          _make_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabdpt_embedding_deephit_v2_cr":       _make_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabdpt_embedding_deephit_v2_cr_adapter": _make_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabdpt_embedding_cox_cr":              _make_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabdpt_embedding_cox_cr_adapter":        _make_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),

    "tabicl_embedding_deephit_cr":          _make_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabicl_embedding_deephit_v2_cr":       _make_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabicl_embedding_deephit_v2_cr_adapter": _make_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabicl_embedding_cox_cr":              _make_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabicl_embedding_cox_cr_adapter":        _make_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),
    
    # Strategy 2: Joint Training (CR)
    "tabpfn_joint_deephit_cr":              _make_joint_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabpfn_joint_deephit_v2_cr":           _make_joint_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabpfn_joint_deephit_v2_cr_adapter":    _make_joint_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabpfn_joint_cox_cr":                 _make_joint_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabpfn_joint_cox_cr_adapter":           _make_joint_fm_wrapper("tabpfn", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),
    
    "tabdpt_joint_deephit_cr":              _make_joint_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabdpt_joint_deephit_v2_cr":           _make_joint_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabdpt_joint_deephit_v2_cr_adapter":    _make_joint_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabdpt_joint_cox_cr":                 _make_joint_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabdpt_joint_cox_cr_adapter":           _make_joint_fm_wrapper("tabdpt", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),
    
    "tabicl_joint_deephit_cr":              _make_joint_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit"),
    "tabicl_joint_deephit_v2_cr":           _make_joint_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2"),
    "tabicl_joint_deephit_v2_cr_adapter":    _make_joint_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="deephit_v2", fixed_use_adapter=True),
    "tabicl_joint_cox_cr":                 _make_joint_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="cox"),
    "tabicl_joint_cox_cr_adapter":           _make_joint_fm_wrapper("tabicl", "deephit_cr", task_type="cr", fixed_cr_loss="cox", fixed_use_adapter=True),
    
    # Strategy 3: Zero-shot (CR)
    "tabpfn_zeroshot_cr_multinomial": _make_zeroshot_wrapper("tabpfn", cr_method="multinomial"),
    "tabpfn_zeroshot_cr_perevent": _make_zeroshot_wrapper("tabpfn", cr_method="per_event"),
}

__all__ = [
    "ALL_MODELS",
]
