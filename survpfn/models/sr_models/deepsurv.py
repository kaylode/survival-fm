"""survpfn.models.deep_surv — DeepSurv (CoxPH neural network) model."""
from typing import Optional
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
from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run
import numpy as np

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"


def train_deepsurv(df_train, df_test, duration_col="Follow Up Data", event_col="Total mortality", random_state=42, tune=False, n_trials=10, save_dir: str = "results", out_dir: str = "results", study_id: Optional[str] = None, verbose: bool = True, **kwargs):
    if verbose:
        print(f"  [DeepSurv] Method: CoxPH MLP, Tuning: {tune}, Device: {kwargs.get('device', 'cpu')}", flush=True)
    out_dir = kwargs.get("out_dir", out_dir)
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
        T_val_t = torch.tensor(T_val, dtype=torch.float32)
        E_val_t = torch.tensor(E_val, dtype=torch.float32)

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_deepsurv_{study_id}.log" if study_id else "optuna_deepsurv.log"
        log_file = os.path.join(out_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"deepsurv_tuning_{study_id}" if study_id else "deepsurv_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        config = load_model_config("deepsurv")
        params.update(config["default"])

        def objective(trial):
            p = apply_tuning_params(trial, config["tuning"])
            num_nodes_trial = [p["nodes"]] * p["layers"]
            
            # Epochs from trial
            epochs_trial = p.get("epochs", 30)

            net = tt.practical.MLPVanilla(in_features, num_nodes_trial, 1, batch_norm=True, dropout=p["dropout"], activation=nn.ReLU)
            optimizer = tt.optim.Adam(lr=p["lr"], weight_decay=1e-4)
            model = CoxPH(net, optimizer)

            # Use internal validation for better early stopping and evaluation
            callbacks = [tt.callbacks.EarlyStopping(patience=5)]
            
            current_bs = params['batch_size']
            model.fit(
                X_tr_t, (T_tr_t, E_tr_t), 
                current_bs, epochs_trial, 
                verbose=True, callbacks=callbacks, 
                val_data=(X_val_t, (T_val_t, E_val_t))
            )
            model.compute_baseline_hazards()
            surv = model.predict_surv_df(X_val_t)
            ev = EvalSurv(surv, T_val, E_val, censor_surv='km')
            res = ev.concordance_td('antolini')
            return res if not np.isnan(res) else 0.0

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            if verbose: print(f"  [DeepSurv] Running {n_remaining} Optuna trials...", flush=True)
            study.optimize(objective, n_trials=n_remaining)
        best_p = study.best_params
        if verbose:
            print(f"  [DeepSurv] Best params: {best_p}", flush=True)
        params['lr'] = best_p['lr']
        params['dropout'] = best_p['dropout']
        params['num_nodes'] = [best_p['nodes']] * best_p['layers']
        params['epochs'] = best_p.get('epochs', config["default"].get("epochs", 30))

    # ── Final training ──────────────────────────────────────────────────────
    # For the final model, we'll split off 10% for EarlyStopping if epochs are high
    X_tr_final, X_val_final, T_tr_final, T_val_final, E_tr_final, E_val_final = train_test_split(
        X_train.values, T_train.values, E_train.values, test_size=0.1, random_state=random_state,
        stratify=E_train.values
    )
    
    in_features = X_train.shape[1]
    net = tt.practical.MLPVanilla(in_features, params['num_nodes'], 1, batch_norm=True, dropout=params['dropout'], activation=nn.ReLU)
    optimizer = tt.optim.Adam(lr=params['lr'], weight_decay=1e-4)
    model = CoxPH(net, optimizer)
    
    callbacks = [tt.callbacks.EarlyStopping(patience=5)]
    
    success = False
    current_bs = params['batch_size']
    while not success:
        try:
            model.fit(
                torch.tensor(X_tr_final, dtype=torch.float32), 
                (torch.tensor(T_tr_final, dtype=torch.float32), torch.tensor(E_tr_final, dtype=torch.float32)), 
                current_bs, params.get('epochs', 30), 
                verbose=verbose, callbacks=callbacks,
                val_data=(torch.tensor(X_val_final, dtype=torch.float32), 
                          (torch.tensor(T_val_final, dtype=torch.float32), torch.tensor(E_val_final, dtype=torch.float32)))
            )
            success = True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"  [DeepSurv] CUDA OOM! Reducing batch_size to {current_bs}", flush=True)
                # Re-init model
                net = tt.practical.MLPVanilla(in_features, params['num_nodes'], 1, batch_norm=True, dropout=params['dropout'], activation=nn.ReLU)
                model = CoxPH(net, tt.optim.Adam(lr=params['lr'], weight_decay=1e-4))
            else:
                raise e

    model.compute_baseline_hazards()
    surv_df = model.predict_surv_df(X_test_tensor)

    # Build standard return values matching the unified API:
    # (model, risk_scores, surv_probs, surv_times)
    surv_times = surv_df.index.values.astype(float)
    surv_probs = surv_df.values.T  # shape (n_test, n_times)

    # risk_scores: higher = worse prognosis.
    # Use −AUC(S(t)) = negative integrated survival, consistent with all other
    # models (MTLR, PCHazard, DeepHitSingle, SODEN).  The old -S(T_max) was
    # sensitive to the arbitrary last time point in the grid (BUG-M1).
    import numpy as _np
    _trapz = getattr(_np, "trapezoid", getattr(_np, "trapz", None))
    risk_scores = -_trapz(surv_probs, surv_times, axis=1)

    return model, risk_scores, surv_probs, surv_times
