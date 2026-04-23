import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from easydict import EasyDict
from scipy.interpolate import interp1d

from survpfn.models.sr_models.survtrace.survtrace import (
    SurvTraceMulti, Trainer, STConfig, LabelTransform,
)

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

def train_survtrace_cr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    num_durations: int = 10,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    hidden_size: int = 16,
    num_hidden_layers: int = 3,
    num_attention_heads: int = 2,
    intermediate_size: int = 64,
    early_stop_patience: int = 10,
    device: str = "cpu",
    **kwargs
) -> tuple:
    """
    Train SurvTraceMulti and return benchmark-compatible outputs.
    """
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    n_feat = len(feat_cols)
    num_causes = int(df_train[event_col].max())

    # ── Time discretisation ────────────────────────────────────────────────
    T_train = df_train[duration_col].values.astype(np.float64)
    E_train = df_train[event_col].values.astype(np.int64)
    
    # We use events of cause 1 to define the cuts (standard practice)
    # Or we can use all events. Let's use all events > 0.
    event_times = T_train[E_train > 0]
    if len(event_times) < num_durations:
        num_durations = max(3, len(event_times))

    horizons = np.linspace(0, 1, num_durations + 2)[1:-1].tolist()
    times = np.quantile(event_times, horizons).tolist()
    t_min = float(T_train.min())
    t_max = float(T_train.max()) + 1e-6
    cuts = np.unique(np.array([t_min] + times + [t_max]))

    labtrans = LabelTransform(cuts=cuts)
    # MUST call fit to initialize duc and di even if cuts are predefined
    labtrans.fit(T_train, (E_train > 0).astype(int))
    
    # Prepare labels for all causes
    df_y_all = pd.DataFrame(index=df_train.index)
    
    # Create event_0, event_1, ... columns
    for k in range(num_causes):
        e_k = (E_train == (k + 1)).astype(int)
        y_k = labtrans.transform(T_train, e_k)
        if k == 0:
            df_y_all["duration"] = y_k[0]
            df_y_all["proportion"] = y_k[2]
        df_y_all[f"event_{k}"] = y_k[1]

    # ── Internal train / val split ─────────────────────────────────────────
    idx = np.arange(len(df_train))
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_fraction, random_state=42,
        stratify=(E_train > 0).astype(int)
    )
    
    df_x_tr = df_train.iloc[idx_tr][feat_cols].reset_index(drop=True)
    df_x_val = df_train.iloc[idx_val][feat_cols].reset_index(drop=True)
    df_y_tr = df_y_all.iloc[idx_tr].reset_index(drop=True)
    df_y_val = df_y_all.iloc[idx_val].reset_index(drop=True)

    # ── Model Configuration ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "survtrace_cr.pt")
        cfg = EasyDict(STConfig.copy())
        cfg.update({
            "num_feature":              n_feat,
            "num_numerical_feature":    n_feat,
            "num_categorical_feature":  0,
            "vocab_size":               0,
            "num_event":                num_causes,
            "out_feature":              int(labtrans.out_features),
            "duration_index":           labtrans.cuts,
            "hidden_size":              hidden_size,
            "intermediate_size":        intermediate_size,
            "num_hidden_layers":        num_hidden_layers,
            "num_attention_heads":      num_attention_heads,
            "early_stop_patience":      early_stop_patience,
            "checkpoint":               ckpt_path,
        })

        # num_attention_heads must divide hidden_size
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            for h in range(cfg.num_attention_heads, 0, -1):
                if cfg.hidden_size % h == 0:
                    cfg.num_attention_heads = h
                    break

        model = SurvTraceMulti(cfg)
        trainer = Trainer(model)
        
        # Fit model
        trainer.fit(
            train_set=(df_x_tr, df_y_tr),
            val_set=(df_x_val, df_y_val),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    # ── Predict on test set ────────────────────────────────────────────────
    X_test = df_test[feat_cols].values.astype(np.float32)
    df_x_test = pd.DataFrame(X_test, columns=feat_cols)
    
    model.eval()
    cif_per_cause = []
    grid = np.linspace(0, cuts.max(), 100)
    
    with torch.no_grad():
        for k in range(num_causes):
            # predict_surv returns survival probability S_k(t)
            # For discrete CR models in SurvTrace, S_k(t) is often 1 - CIF_k(t)
            # or it depends on how hazards are defined.
            # Let's check predict_surv in SurvTraceMulti.
            surv_tensor = model.predict_surv(df_x_test, batch_size=256, event=k)
            surv_np = surv_tensor.cpu().numpy() # (batch, num_time_bins + 1)
            
            # CIF_k = 1 - S_k
            cif_k = 1.0 - surv_np
            
            # Interpolate
            times = labtrans.cuts
            if times[0] > 0:
                times = np.concatenate([[0], times])
                # cif at time 0 is 0
                cif_k = np.concatenate([np.zeros((len(df_test), 1)), cif_k], axis=1)
            
            f = interp1d(times, cif_k, kind='linear', axis=1, fill_value="extrapolate")
            cif_interp = np.clip(f(grid), 0.0, 1.0)
            cif_per_cause.append(cif_interp)

    risk_cause1 = cif_per_cause[0][:, -1]
    
    return model, risk_cause1, cif_per_cause, grid
