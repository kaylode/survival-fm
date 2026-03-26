"""survpfn.models.deep_surv — DeepSurv (CoxPH neural network) model."""

import os
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def train_deepsurv(df_train, df_test, duration_col="Follow Up Data", event_col="Total mortality", random_state=42, tune=False, n_trials=10, save_dir="results", study_id=None):
    T_train = df_train[duration_col]
    E_train = df_train[event_col]
    X_train = df_train.drop(columns=[duration_col, event_col])

    T_test = df_test[duration_col]
    E_test = df_test[event_col]
    X_test = df_test.drop(columns=[duration_col, event_col])

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)
    E_train_tensor = torch.tensor(E_train.values, dtype=torch.float32)
    T_test_tensor = torch.tensor(T_test.values, dtype=torch.float32)
    E_test_tensor = torch.tensor(E_test.values, dtype=torch.float32)

    in_features = X_train.shape[1]

    params = {'lr': 0.01, 'num_nodes': [32, 32], 'dropout': 0.1, 'batch_size': 128}

    if tune:
        X_tr, X_val, T_tr, T_val, E_tr, E_val = train_test_split(
            X_train.values, T_train.values, E_train.values, test_size=0.2, random_state=random_state, stratify=E_train.values
        )
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        T_tr_t = torch.tensor(T_tr, dtype=torch.float32)
        E_tr_t = torch.tensor(E_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        os.makedirs(save_dir, exist_ok=True)
        log_name = f"optuna_deepsurv_{study_id}.log" if study_id else "optuna_deepsurv.log"
        log_file = os.path.join(save_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"deepsurv_tuning_{study_id}" if study_id else "deepsurv_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            lr_trial = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            dropout_trial = trial.suggest_float('dropout', 0.0, 0.5)
            layers = trial.suggest_categorical('layers', [1, 2, 3])
            nodes = trial.suggest_categorical('nodes', [16, 32, 64])
            num_nodes_trial = [nodes] * layers

            net = tt.practical.MLPVanilla(in_features, num_nodes_trial, 1, batch_norm=True, dropout=dropout_trial, activation=nn.ReLU)
            optimizer = tt.optim.AdamWR(lr=lr_trial)
            model = CoxPH(net, optimizer)

            try:
                model.fit(X_tr_t, (T_tr_t, E_tr_t), params['batch_size'], 20, verbose=False)
                model.compute_baseline_hazards()
                surv = model.predict_surv_df(X_val_t)
                ev = EvalSurv(surv, T_val, E_val, censor_surv='km')
                return ev.concordance_td()
            except Exception:
                return 0.0

        study.optimize(objective, n_trials=n_trials)
        best_p = study.best_params
        params['lr'] = best_p['lr']
        params['dropout'] = best_p['dropout']
        params['num_nodes'] = [best_p['nodes']] * best_p['layers']

    net = tt.practical.MLPVanilla(in_features, params['num_nodes'], 1, batch_norm=True, dropout=params['dropout'], activation=nn.ReLU)
    optimizer = tt.optim.AdamWR(lr=params['lr'])
    model = CoxPH(net, optimizer)
    model.fit(X_train_tensor, (T_train_tensor, E_train_tensor), params['batch_size'], 100, verbose=True)
    model.compute_baseline_hazards()
    surv_df = model.predict_surv_df(X_test_tensor)

    # Build standard return values matching the unified API:
    # (model, risk_scores, surv_probs, surv_times)
    # risk_scores: higher = worse prognosis, so use negative last-row survival
    surv_times = surv_df.index.values.astype(float)
    surv_probs = surv_df.values.T  # shape (n_test, n_times)
    risk_scores = -surv_df.iloc[-1].values  # negative survival at last time point

    return model, risk_scores, surv_probs, surv_times
