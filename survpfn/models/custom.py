"""survpfn.models.custom — Custom TorchSurv neural Cox model."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsurv.loss import cox
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


class SurvivalDataset(Dataset):
    def __init__(self, df, duration_col, event_col):
        self.x = torch.tensor(df.drop(columns=[duration_col, event_col]).values, dtype=torch.float32)
        self.event = torch.tensor(df[event_col].values, dtype=torch.bool)
        self.time = torch.tensor(df[duration_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], (self.event[idx], self.time[idx])


def get_torchsurv_model(in_features, nodes1=32, nodes2=64, dropout=0.2):
    model = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, nodes1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(nodes1, nodes2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(nodes2, 1),
    )
    return model


def train_custom_torchsurv(df_train, df_test, duration_col, event_col, epochs=100, lr=1e-2, tune=False, n_trials=10, save_dir="results"):
    in_features = df_train.drop(columns=[duration_col, event_col]).shape[1]

    params = {'lr': lr, 'nodes1': 32, 'nodes2': 64, 'dropout': 0.2, 'batch_size': 128}

    if tune:
        df_tr, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train[event_col], random_state=42)

        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "optuna_custom.log")
        storage = JournalStorage(JournalFileBackend(log_file))

        study = optuna.create_study(
            study_name="custom_torch_tuning",
            direction="minimize",  # We'll minimize loss
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            lr_trial = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            dropout_trial = trial.suggest_float('dropout', 0.1, 0.5)
            n1 = trial.suggest_categorical('nodes1', [16, 32, 64])
            n2 = trial.suggest_categorical('nodes2', [32, 64, 128])

            model_trial = get_torchsurv_model(in_features, n1, n2, dropout_trial)
            optimizer_trial = torch.optim.Adam(model_trial.parameters(), lr=lr_trial)
            ds_tr = SurvivalDataset(df_tr, duration_col, event_col)
            dl_tr = DataLoader(ds_tr, batch_size=params['batch_size'], shuffle=True)

            # Fast training for tuning
            model_trial.train()
            for _ in range(20):
                for x, (event, time) in dl_tr:
                    optimizer_trial.zero_grad()
                    log_hz = model_trial(x)
                    loss = cox.neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
                    loss.backward()
                    optimizer_trial.step()

            model_trial.eval()
            with torch.no_grad():
                ds_val = SurvivalDataset(df_val, duration_col, event_col)
                dl_val = DataLoader(ds_val, batch_size=len(ds_val))
                x_v, (e_v, t_v) = next(iter(dl_val))
                log_hz_v = model_trial(x_v)
                val_loss = cox.neg_partial_log_likelihood(log_hz_v, e_v, t_v, reduction="mean")
                return float(val_loss.item())

        study.optimize(objective, n_trials=n_trials)
        params.update(study.best_params)

    model = get_torchsurv_model(in_features, params['nodes1'], params['nodes2'], params['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    ds_train = SurvivalDataset(df_train, duration_col, event_col)
    dl_train = DataLoader(ds_train, batch_size=params['batch_size'], shuffle=True)

    # Training Loop
    model.train()
    for _ in range(epochs):
        for x, (event, time) in dl_train:
            optimizer.zero_grad()
            log_hz = model(x)
            loss = cox.neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(df_test.drop(columns=[duration_col, event_col]).values, dtype=torch.float32)
        E_test = torch.tensor(df_test[event_col].values, dtype=torch.bool)
        T_test = torch.tensor(df_test[duration_col].values, dtype=torch.float32)

        log_hz_test = model(X_test)
        risk_scores = log_hz_test.squeeze().numpy()

        # Survival probabilities
        X_train_full = torch.tensor(df_train.drop(columns=[duration_col, event_col]).values, dtype=torch.float32)
        E_train_full = torch.tensor(df_train[event_col].values, dtype=torch.bool)
        T_train_full = torch.tensor(df_train[duration_col].values, dtype=torch.float32)
        log_hz_train = model(X_train_full)

        baseline_surv = cox.baseline_survival_function(log_hz_train, E_train_full, T_train_full)
        # survival_function_cox can return a tensor if input is tensor
        surv_probs_tensor = cox.survival_function_cox(baseline_surv, log_hz_test, T_test)

        # Unique observed times in training define our survival function "steps"
        surv_times = np.unique(T_train_full.numpy())
        surv_probs = surv_probs_tensor.detach().cpu().numpy()  # Shape (n_samples, n_times)

    return model, risk_scores, surv_probs, surv_times
