"""
survpfn/discrete_surv.py — Discrete-time survival models.

Models provided
---------------
* MTLR        — Multi-Task Logistic Regression for Survival (pycox.models.MTLR)
* PC-Hazard   — Piecewise-constant hazard / logistic-hazard (pycox.models.PCHazard)
* DeepHit Single — Discrete-time single-risk model (pycox.models.DeepHitSingle)

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

import torchtuples as tt
from pycox.evaluation import EvalSurv
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_discrete(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    num_durations: int,
    model_cls,
):
    """Shared labelling logic for pycox discrete-time models.

    Parameters
    ----------
    df_train, df_test:
        DataFrames with features, duration and event columns.
    duration_col, event_col:
        Column names.
    num_durations:
        Number of discrete time intervals.
    model_cls:
        A pycox discrete-time model class (MTLR, PCHazard, DeepHitSingle) —
        used only to call ``model_cls.label_transform``.

    Returns
    -------
    x_train, x_test : np.ndarray float32
    y_train_t       : transformed labels for model.fit()
    durations_test  : np.ndarray
    events_test     : np.ndarray
    in_features     : int
    labtrans        : fitted label transformer
    """
    T_train = df_train[duration_col].values
    E_train = df_train[event_col].values.astype(float)
    T_test = df_test[duration_col].values
    E_test = df_test[event_col].values.astype(float)

    X_train = df_train.drop(columns=[duration_col, event_col]).values.astype(np.float32)
    X_test = df_test.drop(columns=[duration_col, event_col]).values.astype(np.float32)

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
    risk_scores = -surv_df.iloc[-1].values  # negative survival at last time = risk
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
    save_dir: str = "results",
    random_state: int = 42,
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
    save_dir:
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
        X_tr, X_val, y_tr_0, y_val_0, y_tr_1, y_val_1 = train_test_split(
            X_train, y_train_t[0], y_train_t[1],
            test_size=0.2, random_state=random_state,
        )
        y_tr = (y_tr_0, y_tr_1)

        os.makedirs(save_dir, exist_ok=True)
        storage = JournalStorage(JournalFileStorage(os.path.join(save_dir, "optuna_mtlr.log")))
        study = optuna.create_study(
            study_name="mtlr_tuning", direction="maximize",
            storage=storage, load_if_exists=True,
        )

        def objective(trial):
            lr_t = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            dropout_t = trial.suggest_float("dropout", 0.0, 0.5)
            layers = trial.suggest_int("layers", 1, 3)
            nodes = trial.suggest_categorical("nodes", [32, 64, 128])
            net = _build_net(in_features, [nodes] * layers, out_features, dropout_t)
            m = MTLR(net, tt.optim.Adam(lr_t), duration_index=labtrans.cuts)
            try:
                m.fit(X_tr, y_tr, params["batch_size"], 30, verbose=False)
                surv = m.interpolate(10).predict_surv_df(X_val)
                ev = EvalSurv(surv, y_val_0, y_val_1, censor_surv="km")
                return ev.concordance_td()
            except Exception:
                return 0.0

        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params
        params["lr"] = bp["lr"]
        params["dropout"] = bp["dropout"]
        params["num_nodes"] = [bp["nodes"]] * bp["layers"]

    net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
    model = MTLR(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
    model.fit(X_train, y_train_t, params["batch_size"], params["epochs"], verbose=True)
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
    save_dir: str = "results",
    random_state: int = 42,
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
    except ImportError as exc:
        raise ImportError("pycox is required. Install with: uv add pycox") from exc

    X_train, X_test, y_train_t, T_test, E_test, in_features, labtrans = _prep_discrete(
        df_train, df_test, duration_col, event_col, num_durations, PCHazard
    )

    params = {"lr": 1e-3, "num_nodes": [64, 64], "dropout": 0.1, "batch_size": 256, "epochs": 100}
    out_features = labtrans.out_features

    if tune:
        X_tr, X_val, y_tr_0, y_val_0, y_tr_1, y_val_1 = train_test_split(
            X_train, y_train_t[0], y_train_t[1],
            test_size=0.2, random_state=random_state,
        )
        y_tr = (y_tr_0, y_tr_1)

        os.makedirs(save_dir, exist_ok=True)
        storage = JournalStorage(JournalFileStorage(os.path.join(save_dir, "optuna_pchazard.log")))
        study = optuna.create_study(
            study_name="pchazard_tuning", direction="maximize",
            storage=storage, load_if_exists=True,
        )

        def objective(trial):
            lr_t = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            dropout_t = trial.suggest_float("dropout", 0.0, 0.5)
            layers = trial.suggest_int("layers", 1, 3)
            nodes = trial.suggest_categorical("nodes", [32, 64, 128])
            net = _build_net(in_features, [nodes] * layers, out_features, dropout_t)
            m = PCHazard(net, tt.optim.Adam(lr_t), duration_index=labtrans.cuts)
            try:
                m.fit(X_tr, y_tr, params["batch_size"], 30, verbose=False)
                surv = m.interpolate(10).predict_surv_df(X_val)
                ev = EvalSurv(surv, y_val_0, y_val_1, censor_surv="km")
                return ev.concordance_td()
            except Exception:
                return 0.0

        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params
        params["lr"] = bp["lr"]
        params["dropout"] = bp["dropout"]
        params["num_nodes"] = [bp["nodes"]] * bp["layers"]

    net = _build_net(in_features, params["num_nodes"], out_features, params["dropout"])
    model = PCHazard(net, tt.optim.Adam(params["lr"]), duration_index=labtrans.cuts)
    model.fit(X_train, y_train_t, params["batch_size"], params["epochs"], verbose=True)
    surv_df = model.interpolate(10).predict_surv_df(X_test)

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
    save_dir: str = "results",
    random_state: int = 42,
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
        X_tr, X_val, y_tr_0, y_val_0, y_tr_1, y_val_1 = train_test_split(
            X_train, y_train_t[0], y_train_t[1],
            test_size=0.2, random_state=random_state,
        )
        y_tr = (y_tr_0, y_tr_1)

        os.makedirs(save_dir, exist_ok=True)
        storage = JournalStorage(JournalFileStorage(os.path.join(save_dir, "optuna_deephit_single.log")))
        study = optuna.create_study(
            study_name="deephit_single_tuning", direction="maximize",
            storage=storage, load_if_exists=True,
        )

        def objective(trial):
            lr_t = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            dropout_t = trial.suggest_float("dropout", 0.0, 0.5)
            layers = trial.suggest_int("layers", 1, 3)
            nodes = trial.suggest_categorical("nodes", [32, 64, 128])
            alpha_t = trial.suggest_float("alpha", 0.0, 0.5)
            sigma_t = trial.suggest_float("sigma", 0.01, 1.0, log=True)
            net = _build_net(in_features, [nodes] * layers, out_features, dropout_t)
            m = DeepHitSingle(net, tt.optim.Adam(lr_t),
                              alpha=alpha_t, sigma=sigma_t,
                              duration_index=labtrans.cuts)
            try:
                m.fit(X_tr, y_tr, params["batch_size"], 30, verbose=False)
                surv = m.interpolate(10).predict_surv_df(X_val)
                ev = EvalSurv(surv, y_val_0, y_val_1, censor_surv="km")
                return ev.concordance_td()
            except Exception:
                return 0.0

        study.optimize(objective, n_trials=n_trials)
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
    model.fit(X_train, y_train_t, params["batch_size"], params["epochs"], verbose=True)
    surv_df = model.interpolate(10).predict_surv_df(X_test)

    risk_scores, surv_probs, surv_times = _surv_df_to_arrays(surv_df)
    return model, risk_scores, surv_probs, surv_times
