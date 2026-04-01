"""
survpfn.models.survtrace.survival — SurvTRACE training wrapper.

Adapts the SurvTRACE single-event model to the standard survpfn benchmark API:
    train_survtrace(df_train, df_test, duration_col, event_col, ...) ->
        (model, risk_scores, surv_probs, surv_times)

Data handling notes
-------------------
The benchmark pipeline (benchmark.py/_scale_fold) already:
  - Applies StandardScaler to all feature columns (fit on train, applied to test)
  - Binarises competing-risk event columns

SurvTRACE normally distinguishes categorical vs numerical features with separate
embedding layers. Since the benchmark provides fully standardised float arrays with
no known categorical structure, we treat all features as numerical
(num_categorical_feature = 0). This may slightly reduce SurvTRACE's representational
advantage over the categorical-aware variant on real data, but is the correct
design choice for a fair benchmark where feature metadata is unavailable.

Time discretisation
-------------------
We use num_durations=10 quantile-spaced cut points on the training event times —
more granular than SurvTRACE's default of 5.  Set the `num_durations` parameter
to override.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from survpfn.utils.config import load_model_config, apply_tuning_params


def train_survtrace(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    num_durations: int = 10,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.1,
    hidden_size: int = 16,
    num_hidden_layers: int = 3,
    num_attention_heads: int = 2,
    intermediate_size: int = 64,
    early_stop_patience: int = 10,
    device: Optional[str] = None,
    tune: bool = False,
    n_trials: int = 10,
    save_dir: str = "results",
    study_id: Optional[str] = None,
    **_,
) -> tuple:
    """Train SurvTRACE and return benchmark-compatible outputs.

    Parameters
    ----------
    df_train, df_test : pre-processed DataFrames (features already standardised).
    duration_col, event_col : column names.
    num_durations : number of discrete time intervals (default 10).
    epochs : maximum training epochs (default 100; early stopping applies).
    batch_size : mini-batch size (default 64).
    learning_rate : BERTAdam learning rate (default 1e-3).
    val_fraction : fraction of training data used for validation / early stopping.
    hidden_size : BERT hidden dimension (default 16).
    num_hidden_layers : number of transformer layers (default 3).
    num_attention_heads : number of attention heads (default 2).
    intermediate_size : FFN intermediate dimension (default 64).
    early_stop_patience : early stopping patience in epochs (default 10).

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
        risk_scores : (n_test,) — higher = higher risk
        surv_probs  : (n_test, num_durations+1)
        surv_times  : (num_durations+1,) — cut-point boundaries
    """
    from survpfn.models.survtrace.survtrace import (
        SurvTraceSingle, Trainer, STConfig, LabelTransform,
    )
    from easydict import EasyDict

    # ── Load Config ────────────────────────────────────────────────────────
    config = load_model_config("survtrace")
    params = config["default"].copy()
    # CLI / passed-in overrides
    params.update({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "num_durations": num_durations,
        "early_stop_patience": early_stop_patience,
    })

    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    n_feat = len(feat_cols)

    # ── Time discretisation ────────────────────────────────────────────────
    T_train = df_train[duration_col].values.astype(np.float64)
    E_train = (df_train[event_col].values > 0).astype(np.float32)
    T_test  = df_test[duration_col].values.astype(np.float64)
    E_test  = (df_test[event_col].values > 0).astype(np.float32)

    event_times = T_train[E_train == 1]
    if len(event_times) < num_durations:
        num_durations = max(3, len(event_times))

    horizons = np.linspace(0, 1, num_durations + 2)[1:-1].tolist()  # (0,1) exclusive
    times = np.quantile(event_times, horizons).tolist()
    # ensure unique cuts (can collapse for small datasets)
    times = sorted(set(times))

    t_min = float(T_train.min())
    t_max = float(T_train.max()) + 1e-6
    cuts = np.array([t_min] + times + [t_max])

    labtrans = LabelTransform(cuts=cuts)
    labtrans.fit(T_train, E_train)

    if tune:
        # ── Internal train / val split for HPO ─────────────────────────────
        idx = np.arange(len(df_train))
        idx_tr, idx_val = train_test_split(
            idx, test_size=val_fraction, random_state=42,
            stratify=(E_train > 0).astype(int)
        )
        
        df_x_tr_hpo = df_train.iloc[idx_tr][feat_cols].reset_index(drop=True)
        df_x_val_hpo = df_train.iloc[idx_val][feat_cols].reset_index(drop=True)
        
        y_all = labtrans.transform(T_train, E_train)
        def _make_df_y_hpo(indices):
            return pd.DataFrame({
                "duration":   y_all[0][indices],
                "event":      y_all[1][indices],
                "proportion": y_all[2][indices],
            }, index=indices).reset_index(drop=True)
        
        df_y_tr_hpo = _make_df_y_hpo(idx_tr)
        df_y_val_hpo = _make_df_y_hpo(idx_val)

        os.makedirs(save_dir, exist_ok=True)
        log_name = f"optuna_survtrace_{study_id}.log" if study_id else "optuna_survtrace.log"
        log_file = os.path.join(save_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"survtrace_tuning_{study_id}" if study_id else "survtrace_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize", # C-index is max
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            p = params.copy()
            p.update(apply_tuning_params(trial, config["tuning"]))
            
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt = os.path.join(tmpdir, "trial.pt")
                cfg_trial = EasyDict(STConfig.copy())
                cfg_trial.update({
                    "num_feature":              n_feat,
                    "num_numerical_feature":    n_feat,
                    "num_categorical_feature":  0,
                    "vocab_size":               0,
                    "out_feature":              int(labtrans.out_features),
                    "duration_index":           labtrans.cuts,
                    "hidden_size":              p["hidden_size"],
                    "intermediate_size":        p["intermediate_size"],
                    "num_hidden_layers":        p["num_hidden_layers"],
                    "num_attention_heads":      p["num_attention_heads"],
                    "early_stop_patience":      p["early_stop_patience"],
                    "checkpoint":               ckpt,
                })
                
                m = SurvTraceSingle(cfg_trial)
                t = Trainer(m)
                t.fit(
                    train_set=(df_x_tr_hpo, df_y_tr_hpo),
                    val_set=(df_x_val_hpo, df_y_val_hpo),
                    batch_size=p["batch_size"],
                    epochs=20, # reduced for speed
                    learning_rate=p["learning_rate"],
                )
                
                surv = m.predict_surv(df_x_val_hpo, batch_size=256).cpu().numpy()
                _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
                risk = 1.0 - _trapz(surv, labtrans.cuts, axis=1) / (labtrans.cuts[-1] - labtrans.cuts[0] + 1e-8)
                
                from pycox.evaluation import EvalSurv
                ev = EvalSurv(pd.DataFrame(surv.T, index=labtrans.cuts), T_train[idx_val], E_train[idx_val], censor_surv='km')
                return ev.concordance_td()

        study.optimize(objective, n_trials=n_trials)
        params.update(study.best_params)

    # ── Final Training ─────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Final setup ──────────────────────────────────────────────────────
        ckpt_path = os.path.join(tmpdir, "survtrace.pt")

        cfg = EasyDict(STConfig.copy())
        cfg.update({
            "num_feature":              n_feat,
            "num_numerical_feature":    n_feat,   # all features treated as numerical
            "num_categorical_feature":  0,
            "vocab_size":               0,        # no categoricals → no vocab
            "out_feature":              int(labtrans.out_features),
            "duration_index":           labtrans.cuts,
            "hidden_size":              params["hidden_size"],
            "intermediate_size":        params["intermediate_size"],
            "num_hidden_layers":        params["num_hidden_layers"],
            "num_attention_heads":      params["num_attention_heads"],
            "early_stop_patience":      params["early_stop_patience"],
            "checkpoint":               ckpt_path,
        })

        # ── Data setup for final train ─────────────────────────────────────
        idx = np.arange(len(df_train))
        idx_tr, idx_val = train_test_split(
            idx, test_size=val_fraction, random_state=42,
            stratify=(E_train > 0).astype(int)
        )
        X_all = df_train[feat_cols].values.astype(np.float32)
        y_all = labtrans.transform(T_train, E_train)
        def _make_df_y_final(indices):
            return pd.DataFrame({
                "duration":   y_all[0][indices],
                "event":      y_all[1][indices],
                "proportion": y_all[2][indices],
            }, index=indices).reset_index(drop=True)

        df_x_tr_final  = pd.DataFrame(X_all[idx_tr],  columns=feat_cols).reset_index(drop=True)
        df_x_val_final = pd.DataFrame(X_all[idx_val], columns=feat_cols).reset_index(drop=True)
        df_y_tr_final  = _make_df_y_final(idx_tr)
        df_y_val_final = _make_df_y_final(idx_val)

        # ── Model + Trainer ────────────────────────────────────────────────
        model = SurvTraceSingle(cfg)
        trainer = Trainer(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(
                train_set=(df_x_tr_final, df_y_tr_final),
                val_set=(df_x_val_final, df_y_val_final),
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                learning_rate=params["learning_rate"],
            )

    # ── Predict on test set ────────────────────────────────────────────────
    X_test = df_test[feat_cols].values.astype(np.float32)
    df_x_test = pd.DataFrame(X_test, columns=feat_cols)

    surv_tensor = model.predict_surv(df_x_test, batch_size=256)  # (n_test, T+1)
    surv_probs  = surv_tensor.cpu().numpy()                       # (n_test, T+1)
    surv_times  = labtrans.cuts                                   # (T+1,)

    # Risk = 1 − mean survival (integrated risk)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk = 1.0 - _trapz(surv_probs, surv_times, axis=1) / (surv_times[-1] - surv_times[0] + 1e-8)

    return model, risk, surv_probs, surv_times
