"""survpfn.models — survival model registry.

Refactored to use a unified backbone-head framework for all foundation models.
Strategy 1 (Frozen) and Strategy 2 (Joint) are now handled by the same classes 
using a `freeze_backbone` flag.

Strategies
----------
{fm}_joint_{head}     : Frozen Backbone = False. Backbone + Head trained together.
{fm}_embedding_{head} : Frozen Backbone = True. Faster training, focuses on head.
{fm}_finetune         : Temporal expansion approach (binary classification).
{fm}_zeroshot         : Direct ICL prediction (no head training).
"""

from __future__ import annotations

from typing import Callable, Optional, List
import numpy as np
import pandas as pd
import torch

from survpfn.utils.config import FMConfig

# ── shared helpers ────────────────────────────────────────────────────────────
from .shared.zeroshot import train_zeroshot_surv
from .shared.binning import resolve_num_durations

# ── classical / deep SR models ────────────────────────────────────────────────
from .sr_models.classical import run_multivariate_cox, run_kaplan_meier
from .sr_models.tree import train_rsf, train_gbsa
from .sr_models.deep_surv import train_deepsurv
from .sr_models.deep_hit import train_mtlr, train_pchazard, train_deephit_single
from .sr_models.soden import train_soden
from .sr_models.survtrace import train_survtrace

# ── CR models ─────────────────────────────────────────────────────────────────
from .cr_models.classical import run_competing_risks_cox, run_aalen_johansen, run_fine_gray, run_survival_boost_cr
from .cr_models.deephit import train_deephit_cr

# ── FM entry points ───────────────────────────────────────────────────────────
from .tabpfn import TabPFNSurvPH, TabPFNSurvPHFinetune
from .tabdpt import TabDPTSurvPH, TabDPTSurvPHFinetune
from .tabicl import TabICLSurvPH, TabICLSurvPHFinetune

# ---------------------------------------------------------------------------
# Internal wrapper factories
# ---------------------------------------------------------------------------

def _fm_joint_wrapper(fm_name: str, head_type: str, freeze_backbone: bool, task_type: str = "sr") -> Callable:
    """Unified factory for both Joint and Frozen-Backbone survival models.
    
    Replaces both _embedding_wrapper and _joint_wrapper.
    """
    _SurvPH = {
        "tabpfn": TabPFNSurvPH, 
        "tabdpt": TabDPTSurvPH, 
        "tabicl": TabICLSurvPH, 
    }[fm_name]

    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        cfg = cfg or FMConfig.from_kwargs(**kw)
        num_events = int(df_train[ev_col].max())
        feat_cols = [c for c in df_train.columns if c not in {dur_col, ev_col}]

        num_durations = resolve_num_durations(
            df_train[dur_col].values,
            df_train[ev_col].values,
            -1,
        )

        wrapper = _SurvPH(
            head_type=head_type,
            task_type=task_type,
            num_events=num_events,
            num_durations=num_durations,
            head_num_nodes=cfg.head_hidden_dims,
            dropout=cfg.dropout,
            learning_rate=5e-5,#cfg.learning_rate,
            cr_loss_type=cfg.cr_loss_type,
            use_adapter=cfg.use_adapter,
            input_dim=len(feat_cols),
            context_size=cfg.context_size,
            device=cfg.device,
            freeze_backbone=freeze_backbone,
        )

        wrapper.fit(
            df_train[feat_cols].values.astype(np.float32),
            df_train[dur_col].values,
            df_train[ev_col].values,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=cfg.verbose,
        )

        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_out = wrapper.predict_survival_df(x_test, n_ensemble=cfg.n_ensemble)

        if task_type == "cr":
            cifs = surv_out  # List[pd.DataFrame]
            risk = 1.0 - cifs[0].iloc[-1].values
            return wrapper, risk, [c.values.T for c in cifs], cifs[0].index.values
        else:
            risk = 1.0 - surv_out.iloc[-1].values
            return wrapper, risk, surv_out.values.T, surv_out.index.values

    _fn.__name__ = f"train_{fm_name}_{'frozen' if freeze_backbone else 'joint'}_{head_type}_{task_type}"
    return _fn


def _finetune_wrapper(fm_name: str) -> Callable:
    """Factory for temporally expanded survival finetuning."""
    _SurvPH = {
        "tabpfn": TabPFNSurvPHFinetune,
        "tabdpt": TabDPTSurvPHFinetune,
        "tabicl": TabICLSurvPHFinetune,
    }[fm_name]

    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        cfg = cfg or FMConfig.from_kwargs(**kw)
        feat_cols = [c for c in df_train.columns if c not in {dur_col, ev_col}]

        num_durations = resolve_num_durations(
            df_train[dur_col].values,
            df_train[ev_col].values,
            -1,
        )

        wrapper = _SurvPH(
            num_durations=num_durations,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            device=cfg.device,
            context_size=cfg.context_size,
        )

        wrapper.fit(
            df_train[feat_cols].values.astype(np.float32),
            df_train[dur_col].values,
            df_train[ev_col].values,
            verbose=cfg.verbose,
        )

        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_out = wrapper.predict_survival_df(x_test, n_ensemble=cfg.n_ensemble)

        risk = 1.0 - surv_out.iloc[-1].values
        return wrapper, risk, surv_out.values.T, surv_out.index.values

    _fn.__name__ = f"train_{fm_name}_finetune"
    return _fn


def _zeroshot_wrapper(backbone: str, method: str = "single_context", use_time_bin_encoder: bool = False, n_estimators: int = 1) -> Callable:
    """Strategy 3 — zero-shot ICL, no head training."""
    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        cfg = cfg or FMConfig.from_kwargs(**kw)
        num_events = int(df_train[ev_col].max())
        return train_zeroshot_surv(
            df_train, df_test, dur_col, ev_col,
            backbone=backbone,
            method=method,
            n_bins=cfg.zeroshot_n_bins,
            context_size=cfg.context_size,
            device=cfg.device,
            cr_method=cfg.zeroshot_cr_method if num_events > 1 else "multinomial",
            max_context_size=cfg.context_size,
            use_time_bin_encoder=use_time_bin_encoder,
            n_estimators=n_estimators,
            output_dir=out_dir,
        )

    _fn.__name__ = f"train_{backbone}_zeroshot_{method}"
    return _fn


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, Callable] = {

    # ── Classical / Deep SR ──────────────────────────────────────────────────
    "cox": lambda df_tr, df_ts, dur, ev, **kw: (
        lambda m: (
            m,
            m.predict_partial_hazard(df_ts.drop(columns=[dur, ev])).values,
            m.predict_survival_function(df_ts.drop(columns=[dur, ev])).values.T,
            m.predict_survival_function(df_ts.drop(columns=[dur, ev])).index.values,
        )
    )(run_multivariate_cox(df_tr, df_ts, dur, ev)[0]),

    "km": lambda df_tr, df_ts, dur, ev, **kw: (
        lambda t, p, df: (None, np.zeros(len(df_ts)), np.tile(p, (len(df_ts), 1)), t)
    )(*run_kaplan_meier(df_ts, dur, ev)),

    "rsf":           lambda df_tr, df_ts, dur, ev, **kw: train_rsf(df_tr, df_ts, dur, ev, **kw),
    "gbsa":          lambda df_tr, df_ts, dur, ev, **kw: train_gbsa(df_tr, df_ts, dur, ev, **kw),
    "deepsurv":      lambda df_tr, df_ts, dur, ev, **kw: train_deepsurv(df_tr, df_ts, dur, ev, **kw),
    "mtlr":          lambda df_tr, df_ts, dur, ev, **kw: train_mtlr(df_tr, df_ts, dur, ev, **kw),
    "pchazard":      lambda df_tr, df_ts, dur, ev, **kw: train_pchazard(df_tr, df_ts, dur, ev, **kw),
    "deephit_single":lambda df_tr, df_ts, dur, ev, **kw: train_deephit_single(df_tr, df_ts, dur, ev, **kw),
    "survtrace":     lambda df_tr, df_ts, dur, ev, **kw: train_survtrace(df_tr, df_ts, dur, ev, **kw),
    "soden":         lambda df_tr, df_ts, dur, ev, **kw: train_soden(df_tr, df_ts, dur, ev, **kw),

    # ── FM Frozen Embedding Models (Backward compatibility names) ───────────
    "tabpfn_embedding_cox":      _fm_joint_wrapper("tabpfn", "cox", freeze_backbone=True),
    "tabpfn_embedding_deepsurv": _fm_joint_wrapper("tabpfn", "deepsurv", freeze_backbone=True),
    "tabpfn_embedding_deephit":  _fm_joint_wrapper("tabpfn", "deephit", freeze_backbone=True),
    "tabpfn_embedding_pchazard": _fm_joint_wrapper("tabpfn", "pchazard", freeze_backbone=True),
    "tabpfn_embedding_mtlr":     _fm_joint_wrapper("tabpfn", "mtlr", freeze_backbone=True),

    "tabdpt_embedding_cox":      _fm_joint_wrapper("tabdpt", "cox", freeze_backbone=True),
    "tabdpt_embedding_deepsurv": _fm_joint_wrapper("tabdpt", "deepsurv", freeze_backbone=True),
    "tabdpt_embedding_deephit":  _fm_joint_wrapper("tabdpt", "deephit", freeze_backbone=True),
    "tabdpt_embedding_pchazard": _fm_joint_wrapper("tabdpt", "pchazard", freeze_backbone=True),
    "tabdpt_embedding_mtlr":     _fm_joint_wrapper("tabdpt", "mtlr", freeze_backbone=True),

    "tabicl_embedding_cox":      _fm_joint_wrapper("tabicl", "cox", freeze_backbone=True),
    "tabicl_embedding_deepsurv": _fm_joint_wrapper("tabicl", "deepsurv", freeze_backbone=True),
    "tabicl_embedding_deephit":  _fm_joint_wrapper("tabicl", "deephit", freeze_backbone=True),
    "tabicl_embedding_pchazard": _fm_joint_wrapper("tabicl", "pchazard", freeze_backbone=True),
    "tabicl_embedding_mtlr":     _fm_joint_wrapper("tabicl", "mtlr", freeze_backbone=True),

    # ── FM Temporal Expansion Finetuning ────────────────────────────────────
    "tabpfn_finetune":       _finetune_wrapper("tabpfn"),
    "tabdpt_finetune":       _finetune_wrapper("tabdpt"),
    "tabicl_finetune":       _finetune_wrapper("tabicl"),

    # ── Zero-shot ICL (SR) ──────────────────────────────────────────────────
    "tabpfn_zeroshot":         _zeroshot_wrapper("tabpfn"),
    "tabpfn_zeroshot_perbin":  _zeroshot_wrapper("tabpfn", method="per_bin"),
    "tabpfn_zeroshot_perbin_time":  _zeroshot_wrapper("tabpfn", method="per_bin", use_time_bin_encoder=True),
    "tabpfn_zeroshot_perbin_time_ens":  _zeroshot_wrapper("tabpfn", method="per_bin", use_time_bin_encoder=True, n_estimators=5),
    "tabdpt_zeroshot":         _zeroshot_wrapper("tabdpt"),
    "tabdpt_zeroshot_perbin":  _zeroshot_wrapper("tabdpt", method="per_bin"),
    "tabdpt_zeroshot_perbin_time":  _zeroshot_wrapper("tabdpt", method="per_bin", use_time_bin_encoder=True),
    "tabdpt_zeroshot_perbin_time_ens":  _zeroshot_wrapper("tabdpt", method="per_bin", use_time_bin_encoder=True, n_estimators=5),
    "tabicl_zeroshot":         _zeroshot_wrapper("tabicl"),
    "tabicl_zeroshot_perbin":  _zeroshot_wrapper("tabicl", method="per_bin"),
    "tabicl_zeroshot_perbin_time":  _zeroshot_wrapper("tabicl", method="per_bin", use_time_bin_encoder=True),
    "tabicl_zeroshot_perbin_time_ens":  _zeroshot_wrapper("tabicl", method="per_bin", use_time_bin_encoder=True, n_estimators=5),

    # ── Competing Risks (CR) ────────────────────────────────────────────────
    "cox_cr":             lambda df_tr, df_ts, dur, ev, **kw: run_competing_risks_cox(df_tr, df_ts, dur, ev, **kw),
    "aj_cr":              lambda df_tr, df_ts, dur, ev, **kw: run_aalen_johansen(df_tr, df_ts, dur, ev),
    "fine_gray_cr":       lambda df_tr, df_ts, dur, ev, **kw: run_fine_gray(df_tr, df_ts, dur, ev, **kw),
    "survival_boost_cr":  lambda df_tr, df_ts, dur, ev, **kw: run_survival_boost_cr(df_tr, df_ts, dur, ev, **kw),
    "deephit_cr":         lambda df_tr, df_ts, dur, ev, **kw: train_deephit_cr(df_tr, df_ts, dur, ev, **kw),

    # ── FM Fixed Embedding Models (CR) ──────────────────────────────────────
    "tabpfn_embedding_deephit_cr": _fm_joint_wrapper("tabpfn", "deephit_cr", freeze_backbone=True, task_type="cr"),
    "tabdpt_embedding_deephit_cr": _fm_joint_wrapper("tabdpt", "deephit_cr", freeze_backbone=True, task_type="cr"),
    "tabicl_embedding_deephit_cr": _fm_joint_wrapper("tabicl", "deephit_cr", freeze_backbone=True, task_type="cr"),

    # ── FM Joint Adaptation Models (CR) ─────────────────────────────────────
    "tabpfn_joint_deephit_cr": _fm_joint_wrapper("tabpfn", "deephit_cr", freeze_backbone=False, task_type="cr"),
    "tabdpt_joint_deephit_cr": _fm_joint_wrapper("tabdpt", "deephit_cr", freeze_backbone=False, task_type="cr"),
    "tabicl_joint_deephit_cr": _fm_joint_wrapper("tabicl", "deephit_cr", freeze_backbone=False, task_type="cr"),

    # ── FM Zero-shot ICL (CR) ────────────────────────────────────────────────
    "tabpfn_zeroshot_cr":  _zeroshot_wrapper("tabpfn"),
    "tabdpt_zeroshot_cr":  _zeroshot_wrapper("tabdpt"),
    "tabicl_zeroshot_cr":  _zeroshot_wrapper("tabicl"),
}

__all__ = ["ALL_MODELS"]
