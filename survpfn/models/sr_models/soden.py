"""
survpfn.models.soden — SODEN: Survival ODE Network baseline.

SODEN (Tang et al., NeurIPS 2022) models the hazard function h(t|x) with a
neural network and integrates it via a Neural ODE to obtain the cumulative
hazard H(t|x) = ∫₀ᵗ h(s|x) ds.  The survival function is S(t|x) = exp(-H(t)).

Training loss (NLL):
    L = E[ -D · log h(T|x) + H(T|x) ]

Prediction is continuous-time: S(t|x) can be evaluated at any t without
re-training, unlike discrete-time models.

This module is self-contained — it bundles a minimal ``BaseSurvODEFunc``
so the external SODEN package is not required.  The ODE solver comes from
``torchdiffeq`` (adjoint method for memory-efficient backpropagation).

References
----------
Tang, W. et al. (2022). "SODEN: A Scalable Continuous-Time Survival Model
through Ordinary Differential Equation Networks."  NeurIPS 2022.
https://github.com/jiaqima/SODEN

Chen et al. (2018). "Neural Ordinary Differential Equations."
NeurIPS 2018.  (torchdiffeq package)

Benchmark API
-------------
    train_soden(df_train, df_test, dur_col, ev_col, ...) → (model, risk, surv_probs, surv_times)
"""

from __future__ import annotations
import os
import math
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from survpfn.utils.optuna import get_n_trials_to_run


# ---------------------------------------------------------------------------
# Bundled BaseSurvODEFunc (minimal port from github.com/jiaqima/SODEN)
# ---------------------------------------------------------------------------

class BaseSurvODEFunc(nn.Module):
    """Base class for the ODE right-hand-side function used in SODEN.

    Handles the `batch_time_mode` flag required by torchdiffeq when
    integrating over a batch of (per-sample) time horizons.

    Subclasses must implement ``forward(t, y)`` following the SODEN paper:
    - y shape when batch_time_mode=False: (2 + n_features,)
        [H(t|x), T_horizon, x...]
    - y shape when batch_time_mode=True:  (batch_size, 2 + n_features)
    """

    def __init__(self):
        super().__init__()
        self.nfe: int = 0          # number of function evaluations (diagnostic)
        self.batch_time_mode: bool = False

    def set_batch_time_mode(self, mode: bool) -> None:
        """Toggle between per-sample (False) and batched (True) integration."""
        self.batch_time_mode = mode

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SODEN ODE function (from Tang et al. 2022 — ContextRecMLPODEFunc)
# ---------------------------------------------------------------------------

class SODENODEFunc(BaseSurvODEFunc):
    """ODE RHS for SODEN: dH/dt = h(t, H(t), x) · T_horizon.

    The Rescaling trick (∫₀ᵀ f(s;x)ds = ∫₀¹ T·f(tT;x)dt with s=tT) is used
    so the ODE is always integrated over [0, 1] regardless of the observed
    time horizon per sample.

    Arguments
    ---------
    base_net     : neural network mapping (H(t), tT, x) → positive hazard.
    num_features : dimension of the covariate vector x.
    """

    def __init__(self, base_net: nn.Module, num_features: int):
        super().__init__()
        self.net = base_net
        self.feature_size = num_features

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        device = next(self.parameters()).device

        Lambda_t = y.index_select(-1, torch.tensor([0], device=device)).view(-1, 1)
        T        = y.index_select(-1, torch.tensor([1], device=device)).view(-1, 1)
        x        = y.index_select(-1, torch.arange(2, y.size(-1), device=device))

        # Rescaling: s = t * T_horizon  (integrate over [0,1] → real time [0,T])
        inp = torch.cat(
            [Lambda_t,
             t.expand(T.size()) * T,   # s = t * T
             x.view(-1, self.feature_size)],
            dim=1,
        )
        output = self.net(inp) * T    # h(tT; x) * T  (chain rule)

        zeros = torch.zeros_like(
            y.index_select(-1, torch.arange(1, y.size(-1), device=device))
        )
        output = torch.cat([output, zeros], dim=1)

        return output


# ---------------------------------------------------------------------------
# SODEN model wrapper (fit / predict)
# ---------------------------------------------------------------------------

class SODENModel:
    """SODEN survival model: neural-ODE hazard + NLL training.

    Parameters
    ----------
    n_features   : number of input covariates (after preprocessing).
    hidden_size  : hidden layer size of the hazard neural net.
    n_layers     : number of hidden layers.
    lr           : learning rate.
    batch_size   : mini-batch size.
    epochs       : training epochs.
    rtol / atol  : ODE solver tolerances (looser = faster, less accurate).
    device       : "cuda:0" or "cpu".
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        n_layers: int = 2,
        lr: float = 1e-2,
        batch_size: int = 128,
        epochs: int = 100,
        rtol: float = 1e-2,
        atol: float = 1e-4,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.scaler = None

        # Hazard network: (H(t), t, x) → positive scalar
        # Input dim = 2 + n_features (cumulative hazard, time, covariates)
        layers: list[nn.Module] = [
            nn.Linear(n_features + 2, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1), nn.Softplus()]
        base_net = nn.Sequential(*layers).to(device)

        self.ode_func = SODENODEFunc(base_net, n_features).to(device)
        self.optimizer = torch.optim.Adam(base_net.parameters(), lr=lr, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _ode_loss(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        D_batch: torch.Tensor,
    ) -> torch.Tensor:
        """SODEN NLL loss for a mini-batch.

        L = mean[ -D * log h(T|x) + H(T|x) ]
        """
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as exc:
            raise ImportError(
                "torchdiffeq is required for SODEN.  Run: uv add torchdiffeq"
            ) from exc

        B = X_batch.size(0)
        device = X_batch.device
        all_zeros = torch.zeros(B, dtype=torch.float, device=device)
        init_cond = torch.cat(
            [all_zeros.view(-1, 1), Y_batch.view(-1, 1), X_batch], dim=1
        )
        t = torch.tensor([0.0, 1.0], device=device)

        self.ode_func.set_batch_time_mode(False)
        try:
            cum_haz = odeint(
                self.ode_func, init_cond, t,
                rtol=self.rtol, atol=self.atol
            )[1:].squeeze(0)
        except (AssertionError, RuntimeError) as e:
            # Catch 'underflow in dt' or 'non-finite values' and return high loss
            return torch.tensor(1e6, device=device, requires_grad=True)

        self.ode_func.set_batch_time_mode(True)
        hazards = self.ode_func(t[1:], cum_haz)
        
        # Force 2D if batch_size=1
        if cum_haz.dim() == 1:
            cum_haz = cum_haz.unsqueeze(0)
            hazards = hazards.unsqueeze(0)

        cum_haz = cum_haz[:, 0]
        hazards = hazards[:, 0] / Y_batch.view(-1).clamp(min=1e-6)

        log_h = torch.log(hazards.clamp(min=1e-8))
        loss = (-D_batch.float().view(-1) * log_h + cum_haz).mean()
        
        # Final safety check on loss
        if not torch.isfinite(loss):
            return torch.tensor(1e6, device=device, requires_grad=True)
            
        return loss

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        verbose: bool = True,
    ) -> "SODENModel":
        """Train the SODEN model.

        Parameters
        ----------
        x         : (N, n_features) features.
        durations : (N,) observed times > 0.
        events    : (N,) binary event indicator.
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Robustness: Remove NaNs and Infs if they slipped through
        # and ensure durations are > 0
        mask = np.isfinite(x).all(axis=1) & np.isfinite(durations) & (durations > 0)
        x = x[mask]
        durations = durations[mask]
        events = events[mask]

        if self.scaler is None:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.transform(x)

        X  = torch.tensor(x,         dtype=torch.float32, device=self.device)
        Y  = torch.tensor(durations, dtype=torch.float32, device=self.device)
        D  = torch.tensor(events,    dtype=torch.int32,   device=self.device)

        dataset = TensorDataset(X, Y, D)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss   = float("inf")
        best_params = None

        for epoch in range(self.epochs):
            self.ode_func.net.train()
            epoch_loss = 0.0
            for Xb, Yb, Db in loader:
                loss = self._ode_loss(Xb, Yb, Db)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ode_func.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            epoch_loss /= len(dataset)

            # Simple best-epoch checkpoint (on training loss; val split optional)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params = deepcopy(self.ode_func.net.state_dict())

            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"[SODEN] epoch {epoch:3d}/{self.epochs}  loss={epoch_loss:.4f}")

        if best_params is not None:
            self.ode_func.net.load_state_dict(best_params)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_survival_df(
        self,
        x: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        n_time_points: int = 100,
    ) -> pd.DataFrame:
        """Predict survival functions S(t|x) for test samples.

        Parameters
        ----------
        x           : (N, n_features) test features.
        time_grid   : custom time grid.  If None, uses quantile grid from
                      training data stored in ``self.time_grid_``.
        n_time_points : number of quantile points (used if time_grid is None).

        Returns
        -------
        pd.DataFrame with rows = times and columns = subjects (pycox convention).
        """
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError as exc:
            raise ImportError(
                "torchdiffeq is required for SODEN.  Run: uv add torchdiffeq"
            ) from exc

        if time_grid is None:
            if not hasattr(self, "time_grid_"):
                raise RuntimeError("Call fit() first to store the training time grid.")
            time_grid = self.time_grid_

        # Ensure 0 not in grid (integrate from 0)
        inserted_0 = False
        if time_grid[0] != 0.0:
            time_grid = np.insert(time_grid, 0, 0.0)
            inserted_0 = True

        max_time = float(time_grid.max())
        t_rescaled = torch.tensor(
            time_grid / max_time, dtype=torch.float32, device=self.device
        )

        if self.scaler is not None:
            # Handle NaNs in test data by replacing with 0 (which is the mean after scaling)
            # and ensure finite values.
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = self.scaler.transform(x)

        X = torch.tensor(x, dtype=torch.float32, device=self.device)
        B = X.size(0)
        device = self.device

        all_zeros = torch.zeros(B, dtype=torch.float32, device=device)
        all_max   = max_time * torch.ones(B, dtype=torch.float32, device=device)
        init_cond = torch.cat(
            [all_zeros.view(-1, 1), all_max.view(-1, 1), X], dim=1
        )

        self.ode_func.net.eval()
        with torch.no_grad():
            self.ode_func.set_batch_time_mode(False)
            cum_haz = odeint(
                self.ode_func, init_cond, t_rescaled,
                rtol=self.rtol, atol=self.atol
            )  # (len_t, B, 2+n_feat)
            self.ode_func.set_batch_time_mode(True)

        cum_haz_np = cum_haz[:, :, 0].cpu().numpy()   # (len_t, B)
        surv_np    = np.exp(-cum_haz_np)               # (len_t, B)

        if inserted_0:
            surv_np   = surv_np[1:]
            time_grid = time_grid[1:]

        # pycox convention: rows = times, cols = subjects
        return pd.DataFrame(surv_np, index=time_grid)

    def set_time_grid(self, durations: np.ndarray, n_points: int = 100) -> None:
        """Store the training-set time grid (quantile-based) for prediction."""
        quantiles = np.linspace(0.01, 0.99, n_points)
        self.time_grid_ = np.unique(np.quantile(durations[durations > 0], quantiles))


# ---------------------------------------------------------------------------
# Benchmark API
# ---------------------------------------------------------------------------

def train_soden(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    hidden_size: int = 32,
    n_layers: int = 2,
    lr: float = 1e-2,
    batch_size: int = 128,
    epochs: int = 100,
    n_time_points: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    device: str = "cpu",
    tune: bool = False,
    n_trials: int = 20,
    save_dir: str = "results", # deprecated
    out_dir: str = "results",
    study_id: Optional[str] = None,
    **kwargs,
) -> tuple:
    """Train SODEN and return benchmark-compatible outputs.

    Parameters
    ----------
    df_train / df_test : DataFrames with features + duration_col + event_col.
    (other params)     : hyperparameters passed to SODENModel.

    Returns
    -------
    (model, risk, surv_probs, surv_times)
        risk       : (N_test,) float — integrated survival (negated, higher=higher risk)
        surv_probs : (N_test, n_times) float — S(t|x) array
        surv_times : (n_times,) float — time grid
    """
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    x_train = df_train[feat_cols].values.astype(np.float32)
    x_test  = df_test[feat_cols].values.astype(np.float32)
    dur_tr  = df_train[duration_col].values.astype(np.float32)
    ev_tr   = df_train[event_col].values.astype(np.float32)

    # ── Optional Optuna HPO ─────────────────────────────────────────────
    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            _hs = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
            _nl = trial.suggest_int("n_layers", 1, 3)
            _lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
            _bs = trial.suggest_categorical("batch_size", [64, 128, 256])
            # quick 20-epoch probe
            m = SODENModel(
                n_features=x_train.shape[1],
                hidden_size=_hs, n_layers=_nl, lr=_lr,
                batch_size=_bs, epochs=20, device=device,
            )
            m.set_time_grid(dur_tr)
            m.fit(x_train, dur_tr, ev_tr, verbose=False)
            surv_df = m.predict_survival_df(x_train, n_time_points=50)
            from pycox.evaluation import EvalSurv
            ev = EvalSurv(surv_df, dur_tr, ev_tr, censor_surv="km")
            return -ev.concordance_td("antolini")

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_soden_{study_id}.log" if study_id else "optuna_soden.log"
        log_file = os.path.join(out_dir, log_name)
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"soden_tuning_{study_id}" if study_id else "soden_tuning"
        study = optuna.create_study(
            study_name=study_name, direction="minimize", 
            storage=storage, load_if_exists=True
        )
        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)
        best = study.best_params
        hidden_size = best["hidden_size"]
        n_layers    = best["n_layers"]
        lr          = best["lr"]
        batch_size  = best["batch_size"]

    # ── Train ───────────────────────────────────────────────────────────
    model = SODENModel(
        n_features=x_train.shape[1],
        hidden_size=hidden_size,
        n_layers=n_layers,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        rtol=rtol,
        atol=atol,
        device=device,
    )
    model.set_time_grid(dur_tr, n_points=n_time_points)
    model.fit(x_train, dur_tr, ev_tr, verbose=False)

    # ── Predict ─────────────────────────────────────────────────────────
    surv_df = model.predict_survival_df(x_test)
    times      = surv_df.index.values          # (n_times,)
    surv_array = surv_df.values                # (n_times, N_test)

    # Risk = −AUC (negative mean survival = higher risk for worse prognosis)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = _trapz(surv_array.T, times)
    risk = -auc

    return model, risk, surv_array.T, times
