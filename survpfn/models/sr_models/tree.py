"""survpfn.models.tree — Tree-based survival models (RSF and GBSA)."""

import os
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import lightgbm as lgb
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run
from sksurv.functions import StepFunction

_MAX_SURV_TIMES  = 100      # max time-grid points for the returned surv matrix
_MAX_SAMPLES_FIT = 10_000   # max bootstrap/subsample samples for RSF/GBSA fit


def _build_surv_matrix(model, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (surv_probs, surv_times) with time grid capped at _MAX_SURV_TIMES."""
    # Handle both sksurv and LightGBM wrappers
    times = model.unique_times_
    if len(times) > _MAX_SURV_TIMES:
        idx = np.linspace(0, len(times) - 1, _MAX_SURV_TIMES, dtype=int)
        times = times[idx]
    else:
        idx = None
    
    # Vectorized extraction is much faster for large N (100k+)
    fns = model.predict_survival_function(X_test)
    
    try:
        # StepFunction objects in sksurv share the same x-axis (model.unique_times_)
        # We can stack the .y attributes directly instead of calling each function.
        if isinstance(fns, np.ndarray) and len(fns) > 0 and hasattr(fns[0], 'y'):
            y_matrix = np.array([fn.y for fn in fns], dtype=np.float32)
        elif isinstance(fns, np.ndarray):
            # Already a matrix (from my LightGBM wrapper)
            y_matrix = fns
        else:
            raise AttributeError
            
        if idx is not None:
            surv_probs = y_matrix[:, idx]
        else:
            surv_probs = y_matrix
    except (AttributeError, ValueError):
        # Fallback if fns elements are not standard StepFunctions or shapes differ
        surv_probs = np.empty((len(X_test), len(times)), dtype=np.float32)
        for i, fn in enumerate(fns):
            surv_probs[i] = fn(times)
            
    return surv_probs, times

def train_rsf(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, random_state=42, out_dir="results", study_id=None, **kwargs):
    T_train = df_train[duration_col]
    E_train = df_train[event_col].astype(bool)
    X_train = df_train.drop(columns=[duration_col, event_col])

    y_train_sksurv = pd.DataFrame({'event': E_train, 'time': T_train}).to_records(index=False)

    params = {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 15, 'n_jobs': -1, 'random_state': random_state}

    if tune:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_sksurv, test_size=0.2, random_state=random_state, stratify=E_train
        )

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_rsf_{study_id}.log" if study_id else "optuna_rsf.log"
        log_file = os.path.join(out_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"rsf_tuning_{study_id}" if study_id else "rsf_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        config = load_model_config("rsf")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            model = RandomSurvivalForest(
                **{**params, **p, "n_jobs": 1, "random_state": random_state, "verbose": 1, "max_samples": min(len(X_tr), _MAX_SAMPLES_FIT)}
            )
            model.fit(X_tr, y_tr)
            return model.score(X_val, y_val)

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        params.update(study.best_params)

    # Cap bootstrap samples for large datasets to avoid O(N²) blowup
    final_params = {**params}
    if len(X_train) > _MAX_SAMPLES_FIT:
        final_params["max_samples"] = _MAX_SAMPLES_FIT
    
    # Ensure n_jobs is set for the final fit
    final_params["n_jobs"] = 1

    model = RandomSurvivalForest(**final_params)
    model.fit(X_train, y_train_sksurv)

    X_test = df_test.drop(columns=[duration_col, event_col]).values
    risk_scores = model.predict(X_test)
    surv_probs, surv_times = _build_surv_matrix(model, X_test)

    return model, risk_scores, surv_probs, surv_times


def train_gbsa(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, random_state=42, out_dir="results", study_id=None, **kwargs):
    T_train = df_train[duration_col]
    E_train = df_train[event_col].astype(bool)
    X_train = df_train.drop(columns=[duration_col, event_col])

    y_train_sksurv = pd.DataFrame({'event': E_train, 'time': T_train}).to_records(index=False)

    params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': random_state}

    if tune:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_sksurv, test_size=0.2, random_state=random_state, stratify=E_train
        )

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_gbsa_{study_id}.log" if study_id else "optuna_gbsa.log"
        log_file = os.path.join(out_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"gbsa_tuning_{study_id}" if study_id else "gbsa_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        config = load_model_config("gbsa")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            # Cap subsample during tuning to keep trials fast
            subsample = 1.0
            if len(X_tr) > _MAX_SAMPLES_FIT:
                subsample = _MAX_SAMPLES_FIT / len(X_tr)
                
            model = GradientBoostingSurvivalAnalysis(
                **{**params, **p, "random_state": random_state, "verbose": 1, "subsample": subsample}
            )
            model.fit(X_tr, y_tr)
            return model.score(X_val, y_val)

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        params.update(study.best_params)

    # GBSA uses `subsample` (a fraction 0–1) for stochastic boosting; it does NOT
    # accept `max_samples` (an RSF-only bagging kwarg).  Strip any RSF-only keys
    # that may have crept in via a shared config or future edits (BUG-H5).
    _GBSA_INVALID_KEYS = {"max_samples", "n_jobs", "max_features", "oob_score",
                          "warm_start", "bootstrap"}
    final_params = {k: v for k, v in params.items() if k not in _GBSA_INVALID_KEYS}
    
    # Use more aggressive subsampling for very large datasets to keep training reasonable
    if len(X_train) > _MAX_SAMPLES_FIT:
        final_params["subsample"] = _MAX_SAMPLES_FIT / len(X_train)

    model = GradientBoostingSurvivalAnalysis(**final_params)
    model.fit(X_train, y_train_sksurv)

    X_test = df_test.drop(columns=[duration_col, event_col]).values
    risk_scores = model.predict(X_test)
    surv_probs, surv_times = _build_surv_matrix(model, X_test)

    return model, risk_scores, surv_probs, surv_times
