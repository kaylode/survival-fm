import os
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from pycox.models import DeepHitSingle
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def train_deephit_competing_risks(df_train, df_test, duration_col="Follow Up Data", event_col="death", random_state=42, tune=False, n_trials=10, save_dir="results"):
    T_train = df_train[duration_col]
    E_train = df_train[event_col]
    X_train = df_train.drop(columns=[duration_col, event_col])

    T_test = df_test[duration_col]
    E_test = df_test[event_col]
    X_test = df_test.drop(columns=[duration_col, event_col])

    X_train_np = X_train.values.astype('float32')
    X_test_np = X_test.values.astype('float32')

    num_durations = 100
    labtrans = LabTransDiscreteTime(num_durations)
    y_train = labtrans.fit_transform(T_train.values, E_train.values)
    
    in_features = X_train.shape[1]
    out_features = labtrans.out_features
    
    params = {'lr': 0.01, 'num_nodes': [64, 32], 'dropout': 0.1, 'batch_size': 128}

    if tune:
        X_tr, X_val, T_tr, T_val, E_tr, E_val = train_test_split(
            X_train_np, T_train.values, E_train.values, test_size=0.2, random_state=random_state, stratify=E_train.values
        )
        y_tr = labtrans.transform(T_tr, E_tr)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "optuna_deephit.log")
        storage = JournalStorage(JournalFileStorage(log_file))
        
        study = optuna.create_study(
            study_name="deephit_tuning",
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            lr_trial = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            dropout_trial = trial.suggest_float('dropout', 0.0, 0.5)
            nodes = trial.suggest_categorical('nodes', [32, 64, 128])
            num_nodes_trial = [nodes, nodes//2]
            
            net = tt.practical.MLPVanilla(in_features, num_nodes_trial, out_features=out_features, batch_norm=True, dropout=dropout_trial, activation=nn.ReLU)
            optimizer = tt.optim.AdamWR(lr=lr_trial)
            model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
            
            try:
                model.fit(X_tr, y_tr, batch_size=params['batch_size'], epochs=20, verbose=False)
                surv = model.predict_surv_df(X_val)
                ev = EvalSurv(surv, T_val, E_val, censor_surv='km')
                return ev.concordance_td()
            except Exception:
                return 0.0
                
        study.optimize(objective, n_trials=n_trials)
        best_p = study.best_params
        params['lr'] = best_p['lr']
        params['dropout'] = best_p['dropout']
        params['num_nodes'] = [best_p['nodes'], best_p['nodes']//2]

    net = tt.practical.MLPVanilla(
        in_features, params['num_nodes'], out_features=out_features,
        batch_norm=True, dropout=params['dropout'], activation=nn.ReLU
    )
    optimizer = tt.optim.AdamWR(lr=params['lr'])
    model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
    model.fit(X_train_np, y_train, batch_size=params['batch_size'], epochs=50, verbose=True)
    surv = model.predict_surv_df(X_test_np)
    ev = EvalSurv(surv, T_test.values, E_test.values, censor_surv='km')
    return model, surv, ev

