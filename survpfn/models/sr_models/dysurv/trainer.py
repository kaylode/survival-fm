import os
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run
from survpfn.models.sr_models.dysurv.model_utils import DySurv, DySurvLoss

def train_dysurv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str = "Follow Up Data",
    event_col: str = "Total mortality",
    num_durations: int = 50,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    random_state: int = 42,
    study_id: str = None,
    verbose: bool = True,
    **kwargs
):
    if verbose:
        print(f"  [DySurv] Tuning: {tune}, Device: {kwargs.get('device', 'cpu')}", flush=True)

    # Prepare data
    T_train = df_train[duration_col].values.astype(float)
    E_train = df_train[event_col].values.astype(float)
    X_train = df_train.drop(columns=[duration_col, event_col]).values.astype(np.float32)

    T_test = df_test[duration_col].values.astype(float)
    E_test = df_test[event_col].values.astype(float)
    X_test = df_test.drop(columns=[duration_col, event_col]).values.astype(np.float32)

    # Global time discretization for final training
    labtrans = LogisticHazard.label_transform(num_durations, scheme='quantiles')
    y_train_trans = labtrans.fit_transform(T_train, E_train)
    
    in_features = X_train.shape[1]
    
    # Load default params
    config = load_model_config("dysurv")
    params = config["default"].copy()
    params.update(kwargs)

    device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    def get_model(p, current_out_features, current_labtrans):
        net = DySurv(in_features, p["encoded_features"], current_out_features).to(device)
        optimizer = tt.optim.Adam(lr=p["lr"])
        net.output_all = True
        criterion = DySurvLoss(alphas=p.get("alphas", [1.0, 0.1, 0.1]))
        model = LogisticHazard(net, optimizer, duration_index=current_labtrans.cuts, loss=criterion)
        return model

    if tune:
        X_tr, X_val, T_tr, T_val_raw, E_tr, E_val_raw = train_test_split(
            X_train, T_train, E_train, test_size=0.2, random_state=random_state, stratify=E_train
        )
        # Re-transform for split to avoid data leakage (though for survival binning it's minor)
        labtrans_hpo = LogisticHazard.label_transform(num_durations, scheme='quantiles')
        y_tr = labtrans_hpo.fit_transform(T_tr, E_tr)
        y_val_trans = labtrans_hpo.transform(T_val_raw, E_val_raw)

        train_data = tt.tuplefy(X_tr, (y_tr, X_tr))
        val_data = tt.tuplefy(X_val, (y_val_trans, X_val))
        
        hpo_out_features = labtrans_hpo.out_features

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_dysurv_{study_id}.log" if study_id else "optuna_dysurv.log"
        storage = JournalStorage(JournalFileBackend(os.path.join(out_dir, log_name)))
        
        study_name = f"dysurv_tuning_{study_id}" if study_id else "dysurv_tuning"
        study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, load_if_exists=True)
        
        def objective(trial):
            p = params.copy()
            p.update(apply_tuning_params(trial, config["tuning"]))
            try:
                # Use local hpo_out_features and labtrans_hpo
                model_trial = get_model(p, hpo_out_features, labtrans_hpo)
                callbacks = [tt.callbacks.EarlyStopping(patience=p.get("patience", 10))]
                model_trial.fit(*train_data, batch_size=p["batch_size"], epochs=p["epochs"], 
                                callbacks=callbacks, val_data=val_data, verbose=False)
                
                model_trial.net.output_all = False
                surv = model_trial.interpolate(10).predict_surv_df(X_val)
                ev = EvalSurv(surv, T_val_raw, E_val_raw, censor_surv='km')
                res = ev.concordance_td('antolini')
                return res if not np.isnan(res) else 0.0
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return 0.0
                raise e

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)
        params.update(study.best_params)

    # Final training
    X_tr_final, X_val_final, T_tr_final, T_val_final, E_tr_final, E_val_final = train_test_split(
        X_train, T_train, E_train, test_size=0.2, random_state=random_state, stratify=E_train
    )
    y_tr_f = labtrans.transform(T_tr_final, E_tr_final)
    y_val_f = labtrans.transform(T_val_final, E_val_final)
    
    train_final = tt.tuplefy(X_tr_final, (y_tr_f, X_tr_final))
    val_final = tt.tuplefy(X_val_final, (y_val_f, X_val_final))

    model = get_model(params, labtrans.out_features, labtrans)
    callbacks = [tt.callbacks.EarlyStopping(patience=params.get("patience", 10))]
    model.fit(*train_final, batch_size=params["batch_size"], epochs=params["epochs"], 
              callbacks=callbacks, val_data=val_final, verbose=verbose)
    
    # Prediction
    model.net.output_all = False
    surv_df = model.interpolate(10).predict_surv_df(X_test)
    surv_times = surv_df.index.values.astype(float)
    surv_probs = surv_df.values.T
    
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk_scores = -_trapz(surv_probs, surv_times, axis=1)

    return model, risk_scores, surv_probs, surv_times
