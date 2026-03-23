import os
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from optuna.storages import JournalStorage, JournalFileStorage

def train_rsf(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, random_state=42, save_dir="results"):
    T_train = df_train[duration_col]
    E_train = df_train[event_col].astype(bool)
    X_train = df_train.drop(columns=[duration_col, event_col])
    
    y_train_sksurv = pd.DataFrame({'event': E_train, 'time': T_train}).to_records(index=False)
    
    params = {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 15, 'n_jobs': -1, 'random_state': random_state}

    if tune:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_sksurv, test_size=0.2, random_state=random_state, stratify=E_train
        )

        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "optuna_rsf.log")
        storage = JournalStorage(JournalFileStorage(log_file))
        
        study = optuna.create_study(
            study_name="rsf_tuning",
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            n_est = trial.suggest_int('n_estimators', 50, 200)
            ms_split = trial.suggest_int('min_samples_split', 5, 20)
            ms_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
            
            model = RandomSurvivalForest(
                n_estimators=n_est, min_samples_split=ms_split, 
                min_samples_leaf=ms_leaf, n_jobs=-1, random_state=random_state
            )
            model.fit(X_tr, y_tr)
            return model.score(X_val, y_val)
            
        study.optimize(objective, n_trials=n_trials)
        params.update(study.best_params)

    model = RandomSurvivalForest(**params)
    model.fit(X_train, y_train_sksurv)
    
    X_test = df_test.drop(columns=[duration_col, event_col])
    risk_scores = model.predict(X_test)
    surv_funcs = model.predict_survival_function(X_test)
    surv_times = model.unique_times_
    surv_probs = pd.DataFrame([fn(surv_times) for fn in surv_funcs]).values

    return model, risk_scores, surv_probs, surv_times

def train_gbsa(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, random_state=42, save_dir="results"):
    T_train = df_train[duration_col]
    E_train = df_train[event_col].astype(bool)
    X_train = df_train.drop(columns=[duration_col, event_col])
    
    y_train_sksurv = pd.DataFrame({'event': E_train, 'time': T_train}).to_records(index=False)
    
    params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': random_state}
    
    if tune:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_sksurv, test_size=0.2, random_state=random_state, stratify=E_train
        )

        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "optuna_gbsa.log")
        storage = JournalStorage(JournalFileStorage(log_file))
        
        study = optuna.create_study(
            study_name="gbsa_tuning",
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            lr = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            n_est = trial.suggest_int('n_estimators', 50, 200)
            m_depth = trial.suggest_int('max_depth', 2, 5)
            
            model = GradientBoostingSurvivalAnalysis(
                learning_rate=lr, n_estimators=n_est, max_depth=m_depth,
                random_state=random_state
            )
            model.fit(X_tr, y_tr)
            return model.score(X_val, y_val)
            
        study.optimize(objective, n_trials=n_trials)
        params.update(study.best_params)

    model = GradientBoostingSurvivalAnalysis(**params)
    model.fit(X_train, y_train_sksurv)
    
    X_test = df_test.drop(columns=[duration_col, event_col])
    risk_scores = model.predict(X_test)
    surv_funcs = model.predict_survival_function(X_test)
    surv_times = model.unique_times_
    surv_probs = pd.DataFrame([fn(surv_times) for fn in surv_funcs]).values

    return model, risk_scores, surv_probs, surv_times

