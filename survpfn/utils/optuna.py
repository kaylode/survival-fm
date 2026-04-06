from typing import *
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.preprocessing import StandardScaler




# ---------------------------------------------------------------------------
# Fold-level data preparation
# ---------------------------------------------------------------------------

def _scale_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    duration_col: str,
    event_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split, scale features (fit on train only), binarize strictly."""
    feat_cols = [c for c in df.columns if c not in {duration_col, event_col}]

    df_train = df.iloc[train_idx].reset_index(drop=True).copy()
    df_test = df.iloc[test_idx].reset_index(drop=True).copy()

    scaler = StandardScaler()
    df_train[feat_cols] = scaler.fit_transform(df_train[feat_cols])
    df_test[feat_cols] = scaler.transform(df_test[feat_cols])

    return df_train, df_test


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def get_feature_importance(
    model_name: str,
    model,
    feature_names: list[str],
) -> dict | None:
    """Return a serialisable feature-importance dict or None if not available.

    Supported models
    ----------------
    cox  — coefficients + hazard ratios + p-values from ``CoxPHFitter.summary``
    rsf  — impurity-based feature importances from ``RandomSurvivalForest``
    gbsa — impurity-based feature importances from ``GradientBoostingSurvivalAnalysis``
    """
    if model is None:
        return None

    try:
        if model_name in ("rsf", "gbsa") and hasattr(model, "feature_importances_"):
            return {
                "type": "impurity",
                "features": dict(zip(feature_names, model.feature_importances_.tolist())),
            }

        if model_name == "cox" and hasattr(model, "summary"):
            s = model.summary.reset_index()
            feat_col = "covariate" if "covariate" in s.columns else s.columns[0]
            feats = s[feat_col].tolist()
            return {
                "type": "cox_coefficients",
                "coef":     dict(zip(feats, s["coef"].tolist())),
                "exp_coef": dict(zip(feats, np.exp(s["coef"]).tolist())),
                "p":        dict(zip(feats, s["p"].tolist())),
            }
    except (NotImplementedError, Exception):
        return None

    return None


# ---------------------------------------------------------------------------
# Optuna best-params extraction
# -----------------------------------------------------------------------# Legacy overrides for specific models if needed.
# Most models follow: log_name="optuna_{model_name}.log", study_name="{model_name}_tuning"
_LEGACY_STUDY_OVERRIDES: dict[str, tuple[str, str]] = {
    "rsf":            ("optuna_rsf.log",            "rsf_tuning"),
    "gbsa":           ("optuna_gbsa.log",            "gbsa_tuning"),
    "deepsurv":       ("optuna_deepsurv.log",        "deepsurv_tuning"),
    "mtlr":           ("optuna_mtlr.log",            "mtlr_tuning"),
    "pchazard":       ("optuna_pchazard.log",        "pchazard_tuning"),
    "deephit_single": ("optuna_deephit_single.log",  "deephit_single_tuning"),
    "survtrace":      ("optuna_survtrace.log",       "survtrace_tuning"),
    "deephit_cr":     ("optuna_deephit_cr.log",      "deephit_cr_tuning"),
    "soden":          ("optuna_soden.log",           "soden_tuning"),
    "beta_surv":      ("optuna_beta_surv.log",        "beta_surv_tuning"),
}


def get_best_params(model_name: str, out_dir: str) -> dict | None:
    """Read best hyperparameters + tuning summary from the Optuna journal in out_dir.

    Returns a dict with keys:
        best_params, best_value, n_completed_trials, best_trial_number
    Returns None if the model is not tunable or the journal does not exist yet.
    """
    if model_name in _LEGACY_STUDY_OVERRIDES:
        log_name, study_name = _LEGACY_STUDY_OVERRIDES[model_name]
    else:
        log_name = f"optuna_{model_name}.log"
        study_name = f"{model_name}_tuning"
    log_file = os.path.join(out_dir, log_name)
    if not os.path.exists(log_file):
        return None
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = JournalStorage(JournalFileBackend(log_file))
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        return {
            "best_params":        study.best_params,
            "best_value":         study.best_value,
            "n_completed_trials": len(completed),
            "best_trial_number":  study.best_trial.number,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Neural-network parameter counter
# ---------------------------------------------------------------------------

def count_params(model) -> int | None:
    """Return the number of trainable parameters for PyTorch / pycox models.

    Returns None for non-neural models (Cox, RSF, GBSA, KM).
    """
    try:
        # Direct nn.Module (e.g. TabPFNSurvPH)
        if isinstance(model, nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # pycox / torchtuples wrappers expose .net
        if hasattr(model, "net") and isinstance(model.net, nn.Module):
            return sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Optuna trial management
# ---------------------------------------------------------------------------

def get_n_trials_to_run(study: optuna.Study, n_total_trials: int) -> int:
    """Calculate the number of trials remaining to reach n_total_trials.
    
    If n_total_trials is 20 and 5 trials are already COMPLETE, returns 15.
    If 25 trials are already COMPLETE, returns 0.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_completed = len(completed)
    return max(0, n_total_trials - n_completed)

