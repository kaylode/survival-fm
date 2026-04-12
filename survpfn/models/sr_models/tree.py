"""survpfn.models.tree — Tree-based survival models (RSF and GBSA)."""

import os
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run

# ---------------------------------------------------------------------------
# Scalability constants
# ---------------------------------------------------------------------------
# Large ICU datasets (eICU ≈148k, MIMIC ≈70k) have thousands of unique event
# times. Without capping, the survival matrix becomes (N_test × K_times) which
# can hit 200M+ floats in a Python loop, taking 30–60 min per fold.
_MAX_SURV_TIMES  = 100      # max time-grid points for the returned surv matrix
_MAX_SAMPLES_FIT = 50_000   # max bootstrap samples for RSF/GBSA final fit


def _build_surv_matrix(model, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (surv_probs, surv_times) with time grid capped at _MAX_SURV_TIMES."""
    times = model.unique_times_
    if len(times) > _MAX_SURV_TIMES:
        idx = np.linspace(0, len(times) - 1, _MAX_SURV_TIMES, dtype=int)
        times = times[idx]
    surv_probs = np.row_stack(
        [fn(times) for fn in model.predict_survival_function(X_test)]
    )
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
                **{**params, **p, "n_jobs": -1, "random_state": random_state, "verbose": 1, "max_samples": min(len(X_tr), _MAX_SAMPLES_FIT)}
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
            model = GradientBoostingSurvivalAnalysis(
                **{**params, **p, "random_state": random_state, "verbose": 1}
            )
            model.fit(X_tr, y_tr)
            return model.score(X_val, y_val)

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        params.update(study.best_params)

    # For large datasets cap row-subsampling fraction (GBSA uses `subsample`, not `max_samples`)
    final_params = {**params}
    if len(X_train) > _MAX_SAMPLES_FIT and "subsample" not in final_params:
        final_params["subsample"] = _MAX_SAMPLES_FIT / len(X_train)

    model = GradientBoostingSurvivalAnalysis(**final_params)
    model.fit(X_train, y_train_sksurv)

    X_test = df_test.drop(columns=[duration_col, event_col]).values
    risk_scores = model.predict(X_test)
    surv_probs, surv_times = _build_surv_matrix(model, X_test)

    return model, risk_scores, surv_probs, surv_times
