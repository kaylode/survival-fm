"""
survpfn.models.deep_hit — DeepHit and discrete-time survival models.

Merged from: deep_hit.py + discrete_surv.py

Models provided
---------------
* DeepHit competing risks     — from original deep_hit.py
* MTLR                        — Multi-Task Logistic Regression for Survival
* PC-Hazard / Logistic-Hazard — piecewise-constant hazard
* DeepHit Single              — discrete-time single-risk model

All training functions follow the unified API:
    train_*(df_train, df_test, duration_col, event_col, ...) ->
        (model, risk_scores, surv_probs, surv_times)

risk_scores  : 1-D array, higher = worse prognosis
surv_probs   : 2-D array (n_test, n_times)
surv_times   : 1-D array of time points for surv_probs columns

Dependencies
------------
    uv add pycox torchtuples optuna
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

import torchtuples as tt
from pycox.evaluation import EvalSurv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.model_selection import train_test_split
from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run


# ---------------------------------------------------------------------------
# Competing-risks DeepHit (original deep_hit.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Internal helpers (from discrete_surv.py)
# ---------------------------------------------------------------------------

def _prep_discrete(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    num_durations: int,
    model_cls,
):
    """Shared labelling logic for pycox discrete-time models."""
    T_train = df_train[duration_col].values
    E_train = df_train[event_col].values.astype(float)
    T_test = df_test[duration_col].values
    E_test = df_test[event_col].values.astype(float)

    X_train = df_train.drop(columns=[duration_col, event_col]).values.astype(np.float32)
    X_test = df_test.drop(columns=[duration_col, event_col]).values.astype(np.float32)

    # NOTE: No StandardScaler here — benchmark._scale_fold() already applies
    # StandardScaler per fold (fit on train, transform on test) before any model
    # is called.  A second scaler would double-normalise the features, compressing
    # variance further and harming model quality.

    # Use quantile binning by default for better event distribution
    try:
        labtrans = model_cls.label_transform(num_durations, scheme='quantiles')
    except TypeError:
        labtrans = model_cls.label_transform(num_durations)
        
    y_train_t = labtrans.fit_transform(T_train, E_train)

    return X_train, X_test, y_train_t, T_test, E_test, X_train.shape[1], labtrans


def _build_net(in_features: int, num_nodes: list[int], out_features: int, dropout: float) -> nn.Module:
    return tt.practical.MLPVanilla(
        in_features, num_nodes, out_features,
        batch_norm=True, dropout=dropout, activation=nn.ReLU,
    )


def _surv_df_to_arrays(
    surv_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a pycox survival DataFrame to (risk_scores, surv_probs, surv_times).

    surv_df columns = test subjects, rows = time points.
    """
    surv_times = surv_df.index.values.astype(float)
    surv_probs = surv_df.values.T   # (n_test, n_times)
    
    # Use integrated survival (AUC) as an robust risk score instead of S(T_max)
    # Higher AUC = longer life expectancy = lower risk.
    # risk = -AUC
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise AttributeError("module 'numpy' has no attribute 'trapezoid' or 'trapz'")
    auc = _trapz(surv_df.values.T, surv_times)
    risk_scores = -auc
    return risk_scores, surv_probs, surv_times


# ---------------------------------------------------------------------------
# 1.  MTLR
# ---------------------------------------------------------------------------

def train_mtlr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str = "Follow Up Data",
    event_col: str = "Total mortality",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    random_state: int = 42,
    study_id: str = None,
    **kwargs
) -> tuple:
    """Train an MTLR (Multi-Task Logistic Regression) survival model.

    Parameters
    ----------
    df_train, df_test:
        Train/test DataFrames with features plus ``duration_col`` and
        ``event_col``.
    duration_col:
        Time-to-event column name.
    event_col:
        Event indicator column name (binary: 0 = censored, 1 = event).
    num_durations:
        Number of discrete time intervals.
    tune:
        If True, run Optuna hyperparameter search.
    n_trials:
        Number of Optuna trials (used only when ``tune=True``).
    out_dir:
        Directory for Optuna journal storage.
    random_state:
        Seed for train/val split during tuning.

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    try:
        from pycox.models import MTLR
    except ImportError as exc:
        raise ImportError("pycox is required. Install with: uv add pycox") from exc

    X_train, X_test, y_train_t, T_test, E_test, in_features, labtrans = _prep_discrete(
        df_train, df_test, duration_col, event_col, num_durations, MTLR
    )

    params = {"lr": 1e-3, "num_nodes": [64, 64], "dropout": 0.1, "batch_size": 256, "epochs": 100}
    out_features = labtrans.out_features

    if tune:
        # Split everything together for consistency
        E_all = df_train[event_col].values
        (X_tr, X_val, y_tr_0, y_val_0, y_tr_1, y_val_1, T_tr, T_val, E_tr, E_val) = train_test_split(
            X_train, y_train_t[0], y_train_t[1],
            df_train[duration_col].values, E_all,
            test_size=0.2, random_state=random_state, stratify=E_all
        )
        y_tr = (y_tr_0, y_tr_1)

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_mtlr_{study_id}.log" if study_id else "optuna_mtlr.log"
        storage = JournalStorage(JournalFileBackend(os.path.join(out_dir, log_name)))
        
        study_name = f"mtlr_tuning_{study_id}" if study_id else "mtlr_tuning"
        study = optuna.create_study(
            study_name=study_name, direction="maximize",
            storage=storage, load_if_exists=True,
        )

        config = load_model_config("mtlr")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
            m = MTLR(net, tt.optim.Adam(p["lr"]), duration_index=labtrans.cuts)
            success = False
            current_bs = params["batch_size"]
            while not success:
                try:
                    m.fit(X_tr, y_tr, current_bs, 30, verbose=False)
                    success = True
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and current_bs > 1:
                        torch.cuda.empty_cache()
                        current_bs = max(1, current_bs // 2)
                        print(f"      {C_YELLOW}→ CUDA OOM! Reducing batch_size to {current_bs}{C_RESET}")
                        net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
                        m = MTLR(net, tt.optim.Adam(p["lr"]), duration_index=labtrans.cuts)
                    else:
                        return 0.0
                except Exception:
                    return 0.0
            
            surv = m.interpolate(10).predict_surv_df(X_val)
            # Use continuous evaluation for accurate Antolini C-index
            ev = EvalSurv(surv, T_val, E_val, censor_surv="km")
            return ev.concordance_td()

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        bp = study.best_params
        params["lr"] = bp["lr"]
        params["dropout"] = bp["dropout"]
        params["num_nodes"] = [bp["nodes"]] * bp["layers"]

    net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
    model = MTLR(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
    
    success = False
    current_bs = params["batch_size"]
    while not success:
        try:
            model.fit(X_train, y_train_t, current_bs, params["epochs"], verbose=True)
            success = True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"      {C_YELLOW}→ CUDA OOM (Final)! Reducing batch_size to {current_bs}{C_RESET}")
                net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
                model = MTLR(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
            else:
                raise e
    surv_df = model.interpolate(10).predict_surv_df(X_test)

    risk_scores, surv_probs, surv_times = _surv_df_to_arrays(surv_df)
    return model, risk_scores, surv_probs, surv_times


# ---------------------------------------------------------------------------
# 2.  PC-Hazard
# ---------------------------------------------------------------------------



def train_pchazard(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str = "Follow Up Data",
    event_col: str = "Total mortality",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    random_state: int = 42,
    study_id: str = None,
    **kwargs
) -> tuple:
    """Train a PC-Hazard (piecewise-constant / logistic-hazard) survival model.

    Parameters
    ----------
    df_train, df_test:
        Train/test DataFrames.
    duration_col, event_col:
        Column names.
    num_durations:
        Number of discrete time intervals.
    tune:
        If True, run Optuna hyperparameter search.
    n_trials:
        Number of Optuna trials.
    save_dir:
        Directory for Optuna journal storage.
    random_state:
        Seed for train/val split during tuning.

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    try:
        from pycox.models import PCHazard
        from pycox.models.loss import NLLPCHazardLoss
    except ImportError as exc:
        raise ImportError("pycox is required. Install with: uv add pycox") from exc

    X_train, X_test, y_train_t, T_test, E_test, in_features, labtrans = _prep_discrete(
        df_train, df_test, duration_col, event_col, num_durations, PCHazard
    )

    params = {"lr": 1e-3, "num_nodes": [64, 64], "dropout": 0.1, "batch_size": 256, "epochs": 100}
    out_features = labtrans.out_features

    if tune:
        # Split everything together for consistency
        E_all = df_train[event_col].values
        (X_tr, X_val, 
         y_tr_0, y_val_0, y_tr_1, y_val_1, y_tr_2, y_val_2,
         T_tr, T_val, E_tr, E_val) = train_test_split(
            X_train, y_train_t[0], y_train_t[1], y_train_t[2],
            df_train[duration_col].values, E_all,
            test_size=0.2, random_state=random_state, stratify=E_all
        )
        y_tr = (y_tr_0, y_tr_1, y_tr_2)
        # y_val_t = (y_val_0, y_val_1, y_val_2) # for validation, though pycox fit doesn't use it directly

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_pchazard_{study_id}.log" if study_id else "optuna_pchazard.log"
        storage = JournalStorage(JournalFileBackend(os.path.join(out_dir, log_name)))
        
        study_name = f"pchazard_tuning_{study_id}" if study_id else "pchazard_tuning"
        study = optuna.create_study(
            study_name=study_name, direction="maximize",
            storage=storage, load_if_exists=True,
        )

        config = load_model_config("pchazard")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
            m = PCHazard(net, tt.optim.Adam(p["lr"]), duration_index=labtrans.cuts)
            
            success = False
            current_bs = params["batch_size"]
            while not success:
                try:
                    m.fit(X_tr, y_tr, current_bs, 30, verbose=False)
                    success = True
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and current_bs > 1:
                        torch.cuda.empty_cache()
                        current_bs = max(1, current_bs // 2)
                        print(f"      {C_YELLOW}→ CUDA OOM! Reducing batch_size to {current_bs}{C_RESET}")
                        net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
                        m = PCHazard(net, tt.optim.Adam(p["lr"]), duration_index=labtrans.cuts)
                    else:
                        return 0.0
                except Exception:
                    return 0.0

            surv = m.predict_surv_df(X_val)
            # Use continuous ground truth
            ev = EvalSurv(surv, T_val, E_val, censor_surv="km")
            return ev.concordance_td()

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        bp = study.best_params
        params["lr"] = bp["lr"]
        params["dropout"] = bp["dropout"]
        params["num_nodes"] = [bp["nodes"]] * bp["layers"]

    net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
    model = PCHazard(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
    
    success = False
    current_bs = params["batch_size"]
    while not success:
        try:
            model.fit(X_train, y_train_t, current_bs, params["epochs"], verbose=True)
            success = True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"      {C_YELLOW}→ CUDA OOM (Final)! Reducing batch_size to {current_bs}{C_RESET}")
                net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
                model = PCHazard(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
            else:
                raise e
      
    surv_df = model.predict_surv_df(X_test)

    risk_scores, surv_probs, surv_times = _surv_df_to_arrays(surv_df)
    return model, risk_scores, surv_probs, surv_times


# ---------------------------------------------------------------------------
# 3.  DeepHit Single
# ---------------------------------------------------------------------------

def train_deephit_single(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str = "Follow Up Data",
    event_col: str = "Total mortality",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    random_state: int = 42,
    study_id: str = None,
    **kwargs
) -> tuple:
    """Train a DeepHit Single (discrete-time, single risk) survival model.

    Parameters
    ----------
    df_train, df_test:
        Train/test DataFrames.
    duration_col, event_col:
        Column names.
    num_durations:
        Number of discrete time intervals.
    tune:
        If True, run Optuna hyperparameter search.
    n_trials:
        Number of Optuna trials.
    save_dir:
        Directory for Optuna journal storage.
    random_state:
        Seed for train/val split during tuning.

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    try:
        from pycox.models import DeepHitSingle
    except ImportError as exc:
        raise ImportError("pycox is required. Install with: uv add pycox") from exc

    X_train, X_test, y_train_t, T_test, E_test, in_features, labtrans = _prep_discrete(
        df_train, df_test, duration_col, event_col, num_durations, DeepHitSingle
    )

    params = {"lr": 1e-3, "num_nodes": [64, 64], "dropout": 0.1, "batch_size": 256, "epochs": 100,
              "alpha": 0.2, "sigma": 0.1}
    out_features = labtrans.out_features

    if tune:
        # Split everything together for consistency
        E_all = df_train[event_col].values
        (X_tr, X_val, y_tr_0, y_val_0, y_tr_1, y_val_1, T_tr, T_val, E_tr, E_val) = train_test_split(
            X_train, y_train_t[0], y_train_t[1],
            df_train[duration_col].values, E_all,
            test_size=0.2, random_state=random_state, stratify=E_all
        )
        y_tr = (y_tr_0, y_tr_1)

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_deephit_single_{study_id}.log" if study_id else "optuna_deephit_single.log"
        storage = JournalStorage(JournalFileBackend(os.path.join(out_dir, log_name)))
        
        study_name = f"deephit_single_tuning_{study_id}" if study_id else "deephit_single_tuning"
        study = optuna.create_study(
            study_name=study_name, direction="maximize",
            storage=storage, load_if_exists=True,
        )

        config = load_model_config("deephit_single")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
            m = DeepHitSingle(net, tt.optim.Adam(p["lr"]),
                               alpha=p["alpha"], sigma=p["sigma"],
                               duration_index=labtrans.cuts)
            success = False
            current_bs = params["batch_size"]
            while not success:
                try:
                    m.fit(X_tr, y_tr, current_bs, 30, verbose=False)
                    success = True
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and current_bs > 1:
                        torch.cuda.empty_cache()
                        current_bs = max(1, current_bs // 2)
                        print(f"      {C_YELLOW}→ CUDA OOM! Reducing batch_size to {current_bs}{C_RESET}")
                        net = _build_net(in_features, [p["nodes"]] * p["layers"], out_features, p["dropout"])
                        m = DeepHitSingle(net, tt.optim.Adam(p["lr"]),
                                           alpha=p["alpha"], sigma=p["sigma"],
                                           duration_index=labtrans.cuts)
                    else:
                        return 0.0
                except Exception:
                    return 0.0

            surv = m.interpolate(10).predict_surv_df(X_val)
            # Continuous evaluation for higher quality tuning
            ev = EvalSurv(surv, T_val, E_val, censor_surv="km")
            return ev.concordance_td()

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        bp = study.best_params
        params["lr"] = bp["lr"]
        params["dropout"] = bp["dropout"]
        params["num_nodes"] = [bp["nodes"]] * bp["layers"]
        params["alpha"] = bp["alpha"]
        params["sigma"] = bp["sigma"]

    net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
    model = DeepHitSingle(
        net, tt.optim.Adam(params["lr"]),
        alpha=params["alpha"], sigma=params["sigma"],
        duration_index=labtrans.cuts,
    )
    
    success = False
    current_bs = params["batch_size"]
    while not success:
        try:
            model.fit(X_train, y_train_t, current_bs, params["epochs"], verbose=True)
            success = True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"      {C_YELLOW}→ CUDA OOM (Final)! Reducing batch_size to {current_bs}{C_RESET}")
                net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
                model = DeepHitSingle(
                    net, tt.optim.Adam(params["lr"]),
                    alpha=params["alpha"], sigma=params["sigma"],
                    duration_index=labtrans.cuts,
                )
            else:
                raise e
    surv_df = model.interpolate(10).predict_surv_df(X_test)

    risk_scores, surv_probs, surv_times = _surv_df_to_arrays(surv_df)
    return model, risk_scores, surv_probs, surv_times
