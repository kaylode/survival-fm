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

# ── Valid Binning Schemes ─────────────────────────────────────────────────────
AVAILABLE_BINNING_SCHEMES = ["quantile", "uniform", "hybrid"]

# ── shared helpers ────────────────────────────────────────────────────────────
from .shared.zeroshot import train_zeroshot_surv
from .shared.binning import resolve_num_durations

# ── classical / deep SR models ────────────────────────────────────────────────
from .sr_models.classical import run_multivariate_cox, run_kaplan_meier
from .sr_models.tree import train_rsf, train_gbsa
from .sr_models.deepsurv import train_deepsurv
from .sr_models.discrete import train_mtlr, train_pchazard, train_deephit_single, _surv_df_to_arrays
from .sr_models.soden import train_soden
from .sr_models.survtrace import train_survtrace
from .sr_models.dysurv import train_dysurv

# ── CR models ─────────────────────────────────────────────────────────────────
from .cr_models.classical import run_competing_risks_cox, run_aalen_johansen, run_fine_gray, run_survival_boost_cr
from .cr_models.deep_cr import train_deephit_cr
from .cr_models.dysurv.survival import train_dysurv_cr
from .cr_models.survtrace.survival import train_survtrace_cr
from .cr_models.deepsurv import train_deepsurv_cr

# ── FM entry points ───────────────────────────────────────────────────────────
from .tabpfn import TabPFNSurvPH, TabPFNSurvPHFinetune
from .tabdpt import TabDPTSurvPH, TabDPTSurvPHFinetune
from .tabicl import TabICLSurvPH, TabICLSurvPHFinetune

# ── TabTune (tabtune library) wrappers ───────────────────────────────────────
from .shared.tabtune import TabPFNTune, TabDPTTune, TabICLTune

# ── CR fine-tuning wrappers ──────────────────────────────────────────────────
from .shared.cr import TabPFNCRPH, TabDPTCRPH, TabICLCRPH, CRZeroShotPredictor
from .shared.cr.zeroshot import train_cr_zeroshot

# ---------------------------------------------------------------------------
# Internal wrapper factories
# ---------------------------------------------------------------------------

def _fm_joint_wrapper(fm_name: str, head_type: str, freeze_backbone: bool, task_type: str = "sr", use_isotonic_calibration: bool = False) -> Callable:
    """Unified factory for both Joint and Frozen-Backbone survival models.

    For ``task_type="cr"`` the factory uses the ``BaseCRJointFinetune``
    subclasses (``TabPFNCRPH``, etc.) which carry fixed loss functions and
    correct output orientation.  SR models use the original ``SurvPH`` classes.
    """
    if task_type == "cr":
        _SurvPH = {
            "tabpfn": TabPFNCRPH,
            "tabdpt": TabDPTCRPH,
            "tabicl": TabICLCRPH,
        }[fm_name]
    else:
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

        if kw.get("num_durations") is not None and kw.get("num_durations") > 0:
            num_durations = kw["num_durations"]
        else:
            num_durations = resolve_num_durations(
                df_train[dur_col].values,
                df_train[ev_col].values,
                -1,
            )

        if head_type == "pchazard":
            num_durations = 100
            learning_rate = 1e-4
        else:
            learning_rate = 1e-5

        wrapper = _SurvPH(
            head_type=head_type,
            task_type=task_type,
            num_events=num_events,
            num_durations=num_durations,
            head_num_nodes=[64],#cfg.head_hidden_dims,
            dropout=cfg.dropout,
            patience=cfg.patience,
            random_state=cfg.random_state,
            learning_rate=learning_rate,
            use_adapter=cfg.use_adapter,
            input_dim=len(feat_cols),
            context_size=cfg.context_size,
            device=cfg.device,
            freeze_backbone=freeze_backbone,
            early_event_quantile=cfg.early_event_quantile,
            early_bin_frac=cfg.early_bin_frac,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            sampling_ratio=cfg.sampling_ratio,
            use_isotonic_calibration=use_isotonic_calibration,
        )

        wrapper.fit(
            df_train[feat_cols].values.astype(np.float32),
            df_train[dur_col].values,
            df_train[ev_col].values,
            verbose=cfg.verbose,
        )

        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_out = wrapper.predict_survival_df(x_test, n_ensemble=cfg.n_ensemble)

        if isinstance(surv_out, list):
            cifs = surv_out  # List[pd.DataFrame] — rows=times, cols=patients
            # Return list of risk scores (e.g. integrated CIF) for all causes
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            risks = []
            for cif in cifs:
                if _trapz is not None:
                    risks.append(_trapz(cif.values.T, cif.index.values, axis=1) / (cif.index.values[-1] - cif.index.values[0] + 1e-8))
                else:
                    risks.append(cif.iloc[-1].values)
            return wrapper, risks, [c.values.T for c in cifs], cifs[0].index.values
        else:
            if head_type != 'cox':
                # Use integrated survival (AUC) as an robust risk score instead of S(T_max)
                # Higher AUC = longer life expectancy = lower risk.
                # risk = -AUC
                _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
                risk = -_trapz(surv_out.values.T, surv_out.index.values, axis=1)
                return wrapper, risk, surv_out.values.T, surv_out.index.values
            else:
                risk_scores, surv_probs, surv_times = _surv_df_to_arrays(surv_out)
                return wrapper, risk_scores, surv_probs, surv_times

    _fn.__name__ = f"train_{fm_name}_{'frozen' if freeze_backbone else 'joint'}_{head_type}_{task_type}"
    return _fn


def _cause_specific_fm_wrapper(fm_name: str, head_type: str, freeze_backbone: bool, use_isotonic_calibration: bool = False) -> Callable:
    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        all_events = np.unique(df_train[ev_col].values)
        causes = sorted([c for c in all_events if c > 0])
        if len(causes) == 0: causes = [1]
        
        models = []
        cif_per_cause = []
        grid = None
        
        for cause in causes:
            df_train_c = df_train.copy()
            df_train_c[ev_col] = (df_train_c[ev_col] == cause).astype(int)
            df_test_c = df_test.copy()
            df_test_c[ev_col] = (df_test_c[ev_col] == cause).astype(int)
            
            sr_wrapper = _fm_joint_wrapper(fm_name, head_type, freeze_backbone, task_type="sr", use_isotonic_calibration=use_isotonic_calibration)
            
            model, risk_scores, surv_probs, surv_times = sr_wrapper(
                df_train_c, df_test_c, dur_col, ev_col, tune=tune, n_trials=n_trials, out_dir=out_dir, cfg=cfg, **kw
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
                
        span = grid[-1] - grid[0] + 1e-8
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        risks = []
        for cif in cif_per_cause:
            risks.append(_trapz(cif, grid, axis=1) / span)
        
        return models[0], risks, cif_per_cause, grid
        
    _fn.__name__ = f"train_{fm_name}_{'frozen' if freeze_backbone else 'joint'}_{head_type}_cr_cs"
    return _fn

def _finetune_wrapper(fm_name: str, binning_scheme: str = "quantile", use_isotonic_calibration: bool = False, task_type: str = "sr") -> Callable:
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

        if kw.get("num_durations") is not None and kw.get("num_durations") > 0:
            num_durations = kw["num_durations"]
        else:
            num_durations = resolve_num_durations(
                df_train[dur_col].values,
                df_train[ev_col].values,
                -1,
            )

        wrapper = _SurvPH(
            num_durations=num_durations,
            learning_rate=1e-5, #cfg.learning_rate,
            epochs=5,#cfg.epochs,
            batch_size=cfg.batch_size,
            device=cfg.device,
            context_size=cfg.context_size,
            binning_scheme=binning_scheme,
            early_event_quantile=cfg.early_event_quantile,
            early_bin_frac=cfg.early_bin_frac,
            patience=cfg.patience,
            sampling_ratio=cfg.sampling_ratio,
            random_state=cfg.random_state,
            use_isotonic_calibration=use_isotonic_calibration,
            task_type=task_type,
        )

        wrapper.fit(
            df_train[feat_cols].values.astype(np.float32),
            df_train[dur_col].values,
            df_train[ev_col].values,
            verbose=cfg.verbose,
        )

        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_out = wrapper.predict_survival_df(x_test, n_ensemble=cfg.n_ensemble)

        if isinstance(surv_out, list):
            cifs = surv_out
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            risks = []
            for cif in cifs:
                if _trapz is not None:
                    risks.append(_trapz(cif.values.T, cif.index.values, axis=1) / (cif.index.values[-1] - cif.index.values[0] + 1e-8))
                else:
                    risks.append(cif.iloc[-1].values)
            return wrapper, risks, [c.values.T for c in cifs], cifs[0].index.values
        else:
            # Use integrated survival (AUC) as an robust risk score instead of S(T_max)
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            risk = -_trapz(surv_out.values.T, surv_out.index.values, axis=1)
            return wrapper, risk, surv_out.values.T, surv_out.index.values

    _fn.__name__ = f"train_{fm_name}_finetune"
    return _fn


def _zeroshot_wrapper(backbone: str, method: str = "single_context", use_time_bin_encoder: bool = False, n_estimators: int = 1, binning_scheme: str = "quantile") -> Callable:
    """Strategy 3 — zero-shot ICL, no head training.

    For single-risk data (max event code == 1) delegates to
    ``train_zeroshot_surv`` (``ZeroShotSurvivalPredictor``).
    For competing-risks data (max event code > 1) delegates to
    ``train_cr_zeroshot`` (``CRZeroShotPredictor``).
    """
    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        cfg = cfg or FMConfig.from_kwargs(**kw)
        num_events = int(df_train[ev_col].max())

        if kw.get("num_durations") is not None and kw.get("num_durations") > 0:
            num_durations = kw["num_durations"]
        else:
            num_durations = resolve_num_durations(
                df_train[dur_col].values,
                df_train[ev_col].values,
                -1,
            )

        common_kw = dict(
            backbone=backbone,
            method=method,
            n_bins=num_durations,
            context_size=cfg.context_size,
            device=cfg.device,
            max_context_size=cfg.context_size,
            use_time_bin_encoder=use_time_bin_encoder,
            n_estimators=n_estimators,
            binning_scheme=binning_scheme,
            early_event_quantile=cfg.early_event_quantile,
            early_bin_frac=cfg.early_bin_frac,
        )

        if num_events > 1:
            # Competing risks: use CRZeroShotPredictor
            return train_cr_zeroshot(
                df_train, df_test, dur_col, ev_col,
                cr_method=getattr(cfg, "zeroshot_cr_method", "per_event"),
                **common_kw,
            )
        else:
            return train_zeroshot_surv(
                df_train, df_test, dur_col, ev_col,
                **common_kw,
            )

    _fn.__name__ = f"train_{backbone}_zeroshot_{method}"
    return _fn


def _tabtune_wrapper(fm_name: str, binning_scheme: str = "quantile", task_type: str = "sr") -> Callable:
    """Factory for tabtune-based temporal expansion finetuning.

    Uses :class:`TabPFNTune`, :class:`TabDPTTune`, or :class:`TabICLTune`
    (each backed by a ``tabtune.TabularPipeline``) instead of the hand-written
    ``BaseSurvExpandedFinetune`` subclasses.
    """
    _TuneCls = {
        "tabpfn": TabPFNTune,
        "tabdpt": TabDPTTune,
        "tabicl": TabICLTune,
    }[fm_name]

    def _fn(df_train, df_test, dur_col, ev_col,
            tune=False, n_trials=10, out_dir="results",
            cfg: Optional[FMConfig] = None, **kw):
        cfg = cfg or FMConfig.from_kwargs(**kw)
        feat_cols = [c for c in df_train.columns if c not in {dur_col, ev_col}]

        if kw.get("num_durations") is not None and kw.get("num_durations") > 0:
            num_durations = kw["num_durations"]
        else:
            num_durations = resolve_num_durations(
                df_train[dur_col].values,
                df_train[ev_col].values,
                -1,
            )

        if fm_name == 'tabpfn':
            tuning_strategy="finetune"
            finetune_mode="native"
        else:
            tuning_strategy="peft"
            finetune_mode="sft"

        wrapper = _TuneCls(
            num_durations=num_durations,
            learning_rate=1e-5,
            epochs=5,
            batch_size=cfg.batch_size,
            device=cfg.device,
            binning_scheme=binning_scheme,
            early_event_quantile=cfg.early_event_quantile,
            early_bin_frac=cfg.early_bin_frac,
            patience=cfg.patience,
            random_state=cfg.random_state,
            task_type=task_type,
            finetune_mode=finetune_mode,
			tuning_strategy=tuning_strategy
        )

        wrapper.fit(
            df_train[feat_cols].values.astype(np.float32),
            df_train[dur_col].values,
            df_train[ev_col].values,
            verbose=cfg.verbose,
        )

        x_test = df_test[feat_cols].values.astype(np.float32)
        surv_out = wrapper.predict_survival_df(x_test, n_ensemble=0)

        if isinstance(surv_out, list):
            cifs = surv_out
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            risks = []
            for cif in cifs:
                if _trapz is not None:
                    risks.append(_trapz(cif.values.T, cif.index.values, axis=1) / (cif.index.values[-1] - cif.index.values[0] + 1e-8))
                else:
                    risks.append(cif.iloc[-1].values)
            return wrapper, risks, [c.values.T for c in cifs], cifs[0].index.values
        else:
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            risk = -_trapz(surv_out.values.T, surv_out.index.values, axis=1)
            return wrapper, risk, surv_out.values.T, surv_out.index.values

    _fn.__name__ = f"train_{fm_name}_tabtune"
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
    "dysurv":        lambda df_tr, df_ts, dur, ev, **kw: train_dysurv(df_tr, df_ts, dur, ev, **kw),

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

    # ── TabTune library finetuning (tabtune.TabularPipeline backend) ─────────
    "tabpfn_tabtune":        _tabtune_wrapper("tabpfn"),
    "tabdpt_tabtune":        _tabtune_wrapper("tabdpt"),
    "tabicl_tabtune":        _tabtune_wrapper("tabicl"),

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

    # ---- Binning Schemes -----
    "tabpfn_finetune_hybrid":       _finetune_wrapper("tabpfn", binning_scheme="hybrid"),
    "tabdpt_finetune_hybrid":       _finetune_wrapper("tabdpt", binning_scheme="hybrid"),
    "tabicl_finetune_hybrid":       _finetune_wrapper("tabicl", binning_scheme="hybrid"),

    "tabpfn_zeroshot_hybrid":  _zeroshot_wrapper("tabpfn", method="per_bin", use_time_bin_encoder=True, n_estimators=5, binning_scheme="hybrid"),
    "tabdpt_zeroshot_hybrid":  _zeroshot_wrapper("tabdpt", method="per_bin", use_time_bin_encoder=True, n_estimators=5, binning_scheme="hybrid"),
    "tabicl_zeroshot_hybrid":  _zeroshot_wrapper("tabicl", method="per_bin", use_time_bin_encoder=True, n_estimators=5, binning_scheme="hybrid"),

    # ---- Calibration ----
    "tabpfn_embedding_mtlr_calibration": _fm_joint_wrapper("tabpfn", "mtlr", freeze_backbone=True, use_isotonic_calibration=False),
    "tabdpt_embedding_mtlr_calibration": _fm_joint_wrapper("tabdpt", "mtlr", freeze_backbone=True, use_isotonic_calibration=False),
    "tabicl_embedding_mtlr_calibration": _fm_joint_wrapper("tabicl", "mtlr", freeze_backbone=True, use_isotonic_calibration=False),


    # ── Competing Risks (CR) ────────────────────────────────────────────────
    "cox_cr":             lambda df_tr, df_ts, dur, ev, **kw: run_competing_risks_cox(df_tr, df_ts, dur, ev, **kw),
    "aj_cr":              lambda df_tr, df_ts, dur, ev, **kw: run_aalen_johansen(df_tr, df_ts, dur, ev),
    "fine_gray_cr":       lambda df_tr, df_ts, dur, ev, **kw: run_fine_gray(df_tr, df_ts, dur, ev, **kw),
    "survival_boost_cr":  lambda df_tr, df_ts, dur, ev, **kw: run_survival_boost_cr(df_tr, df_ts, dur, ev, **kw),
    "deephit_cr":         lambda df_tr, df_ts, dur, ev, **kw: train_deephit_cr(df_tr, df_ts, dur, ev, **kw),
    "dysurv_cr":          lambda df_tr, df_ts, dur, ev, **kw: train_dysurv_cr(df_tr, df_ts, dur, ev, **kw),
    "survtrace_cr":       lambda df_tr, df_ts, dur, ev, **kw: train_survtrace_cr(df_tr, df_ts, dur, ev, **kw),
    "deepsurv_cr":        lambda df_tr, df_ts, dur, ev, **kw: train_deepsurv_cr(df_tr, df_ts, dur, ev, **kw),

    # ── FM Fixed Embedding Models (CR) ──────────────────────────────────────
    "tabpfn_embedding_deephit_cr": _fm_joint_wrapper("tabpfn", "deephit", freeze_backbone=True, task_type="cr"),
    "tabdpt_embedding_deephit_cr": _fm_joint_wrapper("tabdpt", "deephit", freeze_backbone=True, task_type="cr"),
    "tabicl_embedding_deephit_cr": _fm_joint_wrapper("tabicl", "deephit", freeze_backbone=True, task_type="cr"),


    "tabpfn_embedding_cox_cr": _fm_joint_wrapper("tabpfn", "cox", freeze_backbone=True, task_type="cr"),
    "tabdpt_embedding_cox_cr": _fm_joint_wrapper("tabdpt", "cox", freeze_backbone=True, task_type="cr"),
    "tabicl_embedding_cox_cr": _fm_joint_wrapper("tabicl", "cox", freeze_backbone=True, task_type="cr"),

    "tabpfn_embedding_mtlr_cr": _cause_specific_fm_wrapper("tabpfn", "mtlr", freeze_backbone=True),
    "tabdpt_embedding_mtlr_cr": _cause_specific_fm_wrapper("tabdpt", "mtlr", freeze_backbone=True),
    "tabicl_embedding_mtlr_cr": _cause_specific_fm_wrapper("tabicl", "mtlr", freeze_backbone=True),

    "tabpfn_embedding_pchazard_cr": _cause_specific_fm_wrapper("tabpfn", "pchazard", freeze_backbone=True),
    "tabdpt_embedding_pchazard_cr": _cause_specific_fm_wrapper("tabdpt", "pchazard", freeze_backbone=True),
    "tabicl_embedding_pchazard_cr": _cause_specific_fm_wrapper("tabicl", "pchazard", freeze_backbone=True),

    "tabpfn_finetune_cr":       _finetune_wrapper("tabpfn", task_type="cr"),
    "tabdpt_finetune_cr":       _finetune_wrapper("tabdpt", task_type="cr"),
    "tabicl_finetune_cr":       _finetune_wrapper("tabicl", task_type="cr"),

    # ── FM Zero-shot ICL (CR) ────────────────────────────────────────────────
    # Use per_bin + time-bin encoder for CR: produces a separate FM fit per
    # time bin, which is more expressive for multi-cause CIF estimation.
    "tabpfn_zeroshot_cr":  _zeroshot_wrapper("tabpfn", method="per_bin", use_time_bin_encoder=True),
    "tabdpt_zeroshot_cr":  _zeroshot_wrapper("tabdpt", method="per_bin", use_time_bin_encoder=True),
    "tabicl_zeroshot_cr":  _zeroshot_wrapper("tabicl", method="per_bin", use_time_bin_encoder=True),
    "tabpfn_zeroshot_cr_ens":  _zeroshot_wrapper("tabpfn", method="per_bin", use_time_bin_encoder=True, n_estimators=5),
    "tabdpt_zeroshot_cr_ens":  _zeroshot_wrapper("tabdpt", method="per_bin", use_time_bin_encoder=True, n_estimators=5),
    "tabicl_zeroshot_cr_ens":  _zeroshot_wrapper("tabicl", method="per_bin", use_time_bin_encoder=True, n_estimators=5),
}

__all__ = ["ALL_MODELS"]
