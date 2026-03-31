"""
survpfn.models.heads — Shared survival head layer for FM-embedding pipelines.

This module is the single source of truth for all embedding-based survival
heads.  Any FM backbone (TabPFN, TabDPT, TabICL, …) can produce a fixed
embedding matrix and then delegate the survival-head logic here.

Public API
----------
MLPVanilla          — Kaiming-initialised MLP backbone
EmbeddingSurvHead   — Unified survival head (Cox | DeepHit | PCHazard | MTLR)
EmbeddingCoxPH      — Backward-compatible factory (returns EmbeddingSurvHead)
tune_embedding_head — Optuna HPO helper shared by all FM training functions
train_fm_embedding_surv — End-to-end generic training function

Supported head_type values
--------------------------
  "cox"       — CoxPH (partial likelihood, continuous time)
  "deephit"   — DeepHitSingle (discrete-time, ranking + NLL loss)
  "pchazard"  — PCHazard (piecewise-constant hazard, discrete time)
  "mtlr"      — MTLR (multi-task logistic regression, discrete time)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, DeepHitSingle, MTLR, PCHazard
from survpfn.utils.config import load_model_config, apply_tuning_params


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------

class MLPVanilla(nn.Module):
    """Fully-connected network with Kaiming initialisation.

    Parameters
    ----------
    in_features      : input dimension
    num_nodes        : hidden layer widths (e.g. [128, 64])
    out_features     : output dimension (1 for Cox, num_durations for discrete)
    batch_norm       : apply BatchNorm1d after each hidden layer
    dropout          : dropout probability (None = disabled)
    activation       : hidden-layer activation class (default: ReLU)
    output_activation: optional activation on the output layer
    output_bias      : include bias on the output layer
    """

    def __init__(
        self,
        in_features: int,
        num_nodes: list,
        out_features: int,
        batch_norm: bool = True,
        dropout: Optional[float] = None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias: bool = True,
    ):
        super().__init__()
        nodes = [in_features] + list(num_nodes)
        layers: list[nn.Module] = []

        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

        out_layer = nn.Linear(nodes[-1], out_features, bias=output_bias)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity="relu")
        if output_bias:
            nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)
        self._init_hidden_weights()

    def _init_hidden_weights(self):
        children = list(self.net.children())
        last_linear = next(
            (m for m in reversed(children) if isinstance(m, nn.Linear)), None
        )
        for m in self.net:
            if isinstance(m, nn.Linear) and m is not last_linear:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Unified survival head
# ---------------------------------------------------------------------------

class EmbeddingSurvHead:
    """Survival head that accepts pre-computed FM embeddings.

    Supports all four pycox head types via a single ``head_type`` switch.
    The pycox model (and, for discrete heads, the label transform) are built
    lazily inside ``fit()`` so that time-quantile cut points are determined
    from the actual training data.

    Parameters
    ----------
    embedding_dim : dimensionality of the input embedding vector
    head_type     : one of "cox", "deephit", "pchazard", "mtlr"
    num_durations : number of discrete time intervals (ignored for Cox)
    num_nodes     : hidden layer widths for the MLP head
    dropout       : MLP dropout probability
    lr            : Adam learning rate
    batch_norm    : apply BatchNorm1d in MLP hidden layers
    """

    def __init__(
        self,
        embedding_dim: int,
        head_type: str = "cox",
        num_durations: int = 100,
        num_nodes: list = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_norm: bool = True,
    ):
        if num_nodes is None:
            num_nodes = [64, 64]
        self.embedding_dim = embedding_dim
        self.head_type = head_type.lower()
        self.num_durations = num_durations
        self.num_nodes = list(num_nodes)
        self.dropout = dropout
        self.lr = lr
        self.batch_norm = batch_norm

        # Built lazily in fit()
        self.model = None
        self.labtrans = None
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[tuple] = None

        if self.head_type not in {"cox", "deephit", "pchazard", "mtlr"}:
            raise ValueError(
                f"Unknown head_type '{head_type}'. "
                "Choose from: cox, deephit, pchazard, mtlr."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_net(self) -> MLPVanilla:
        n_out = 1 if self.head_type == "cox" else self.num_durations
        return MLPVanilla(
            in_features=self.embedding_dim,
            num_nodes=self.num_nodes,
            out_features=n_out,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> "EmbeddingSurvHead":
        """Fit the survival head on pre-computed embeddings.

        For Cox, ``compute_baseline()`` must be called afterwards before
        ``predict_survival()``.  For discrete heads (deephit / pchazard / mtlr)
        it is not needed.
        """
        x = embeddings.astype(np.float32)

        # 1. Determine label transform and actual number of output nodes.
        # For discrete heads, the requested num_durations might be reduced
        # if there are not enough unique time points in the training data.
        y_train = None
        if self.head_type == "cox":
            self.labtrans = None
            n_out = 1
        elif self.head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            self.labtrans = LabTransDiscreteTime(self.num_durations, scheme='quantiles')
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = len(self.labtrans.cuts)
        elif self.head_type == "pchazard":
            try:
                self.labtrans = PCHazard.label_transform(self.num_durations, scheme='quantiles')
            except TypeError:
                self.labtrans = PCHazard.label_transform(self.num_durations)
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = len(self.labtrans.cuts)
        elif self.head_type == "mtlr":
            try:
                self.labtrans = MTLR.label_transform(self.num_durations, scheme='quantiles')
            except TypeError:
                self.labtrans = MTLR.label_transform(self.num_durations)
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = len(self.labtrans.cuts)

        # Update num_durations to match actual bins (critical for discrete heads)
        if self.head_type != "cox":
            self.num_durations = n_out

        # 2. Build net and optimizer now that n_out is finalised
        net = self._build_net()
        opt = tt.optim.Adam(lr=self.lr)

        # 3. Initialize pycox model and fit
        if self.head_type == "cox":
            self.model = CoxPH(net, opt)
            y_train = (durations.astype(np.float32), events.astype(np.float32))
            self._x_train = x
            self._y_train = y_train
            self.model.fit(x, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=verbose)

        elif self.head_type == "deephit":
            self.model = DeepHitSingle(net, opt, duration_index=self.labtrans.cuts)
            self.model.fit(x, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=verbose)

        elif self.head_type == "pchazard":
            self.model = PCHazard(net, opt, duration_index=self.labtrans.cuts)
            self.model.fit(x, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=verbose)

        elif self.head_type == "mtlr":
            self.model = MTLR(net, opt, duration_index=self.labtrans.cuts)
            self.model.fit(x, y_train, batch_size=batch_size, epochs=epochs,
                           verbose=verbose)

        return self

    def compute_baseline(self) -> "EmbeddingSurvHead":
        """Compute Nelson-Aalen baseline hazard (Cox only; no-op otherwise)."""
        if self.head_type == "cox":
            if self._x_train is None:
                raise RuntimeError("Call fit() before compute_baseline().")
            self.model.compute_baseline_hazards(
                input=self._x_train, target=self._y_train
            )
        return self

    def predict_survival(self, embeddings: np.ndarray) -> pd.DataFrame:
        """Return a (time × subject) survival-probability DataFrame."""
        if self.model is None:
            raise RuntimeError("Call fit() (and compute_baseline() for Cox) first.")
        x = embeddings.astype(np.float32)
        if self.head_type == "cox":
            return self.model.predict_surv_df(x)
        # Discrete heads: interpolate to smooth step-function output
        if hasattr(self.model, "interpolate"):
            return self.model.interpolate(10).predict_surv_df(x)
        return self.model.predict_surv_df(x)

    def concordance_index(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        method: str = "antolini",
    ) -> float:
        """Time-dependent concordance index (Antolini, 2005)."""
        surv_df = self.predict_survival(embeddings)
        ev = EvalSurv(
            surv=surv_df,
            durations=durations,
            events=events,
            censor_surv="km",
        )
        return ev.concordance_td(method)


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

def EmbeddingCoxPH(
    embedding_dim: int,
    num_nodes: list = None,
    out_features: int = 1,      # ignored — kept for API compat
    batch_norm: bool = True,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> EmbeddingSurvHead:
    """Backward-compatible constructor.

    Returns an ``EmbeddingSurvHead`` with ``head_type='cox'``.
    The ``out_features`` parameter is accepted but ignored
    (Cox always produces a scalar risk score).
    """
    if num_nodes is None:
        num_nodes = [64, 64]
    return EmbeddingSurvHead(
        embedding_dim=embedding_dim,
        head_type="cox",
        num_nodes=num_nodes,
        dropout=dropout,
        lr=learning_rate,
        batch_norm=batch_norm,
    )


# ---------------------------------------------------------------------------
# Shared Optuna tuning helper
# ---------------------------------------------------------------------------

_NODES_MAP: dict[str, list[int]] = {
    "64-64":   [64, 64],
    "128-64":  [128, 64],
    "256-128": [256, 128],
}


def tune_embedding_head(
    train_emb: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    emb_dim: int,
    head_type: str,
    n_trials: int,
    save_dir: str,
    fm_name: str,
    study_id: Optional[str],
    fixed_params: dict,
    num_durations: int = 100,
) -> dict:
    """Run Optuna HPO for an embedding head.

    Searches over (lr, dropout, hidden-layer architecture).

    Parameters
    ----------
    train_emb    : pre-computed training embeddings
    t_train      : training durations
    e_train      : training events
    emb_dim      : embedding dimensionality
    head_type    : one of cox / deephit / pchazard / mtlr
    n_trials     : Optuna trial budget
    save_dir     : directory for the Optuna journal file
    fm_name      : short model name for log-file naming (e.g. "tabpfn")
    study_id     : optional suffix for multi-fold isolation
    fixed_params : fallback / non-tuned hyper-params (must contain "batch_size")
    num_durations: discrete time bins (forwarded to EmbeddingSurvHead)

    Returns
    -------
    Updated params dict with best lr / dropout / nodes filled in.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from sklearn.model_selection import train_test_split

    os.makedirs(save_dir, exist_ok=True)
    slug       = f"{fm_name}_{head_type}"
    log_name   = f"optuna_{slug}_{study_id}.log" if study_id else f"optuna_{slug}.log"
    study_name = f"{slug}_{study_id}"            if study_id else slug
    storage    = JournalStorage(
        JournalFileBackend(os.path.join(save_dir, log_name))
    )

    X_tr, X_val, T_tr, T_val, E_tr, E_val = train_test_split(
        train_emb, t_train, e_train,
        test_size=0.2, random_state=42,
        stratify=(e_train > 0).astype(int),
    )

    config = load_model_config("embedding_head")

    def objective(trial: optuna.Trial) -> float:
        p = apply_tuning_params(trial, config["tuning"])
        nd_t = _NODES_MAP[p["nodes"]]
        m = EmbeddingSurvHead(
            embedding_dim=emb_dim,
            head_type=head_type,
            num_durations=num_durations,
            num_nodes=nd_t,
            dropout=p["dropout"],
            lr=p["lr"],
        )
        m.fit(X_tr, T_tr, E_tr, epochs=50,
              batch_size=fixed_params["batch_size"], verbose=False)
        if head_type == "cox":
            m.compute_baseline()
        return m.concordance_index(X_val, T_val, E_val)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    return {
        **fixed_params,
        "lr":      best["lr"],
        "dropout": best["dropout"],
        "nodes":   _NODES_MAP[best["nodes"]],
    }


# ---------------------------------------------------------------------------
# Generic end-to-end training function
# ---------------------------------------------------------------------------

def train_fm_embedding_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    embedding_fn: Callable,
    emb_kwargs: Optional[dict] = None,
    head_type: str = "cox",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 20,
    save_dir: str = "results",
    study_id: Optional[str] = None,
    fm_name: str = "fm",
) -> tuple:
    """Train any FM backbone + any survival head end-to-end.

    This is the single implementation that TabPFN / TabDPT / TabICL
    training modules delegate to.  Each FM module provides a thin wrapper
    that resolves its own checkpoint / device config and passes the
    appropriate ``embedding_fn``.

    Parameters
    ----------
    df_train / df_test : DataFrames with features + outcome columns
    duration_col       : name of the duration column
    event_col          : name of the event indicator column
    embedding_fn       : callable with signature
                         ``(X_train, y_bin, X_test, **emb_kwargs)
                           -> (train_emb, test_emb)``
                         where arrays are ``np.ndarray`` of ``float32``.
    emb_kwargs         : extra keyword args forwarded to ``embedding_fn``
    head_type          : survival head — cox | deephit | pchazard | mtlr
    num_durations      : discrete time bins (ignored for Cox)
    tune               : run Optuna HPO before final fit
    n_trials           : Optuna trial budget (used when tune=True)
    save_dir           : directory for Optuna logs and artefacts
    study_id           : optional suffix for log-file isolation (e.g. fold id)
    fm_name            : short label for logging / log-file naming

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
        model       — fitted EmbeddingSurvHead
        risk_scores — (n_test,) array, higher = higher risk
        surv_probs  — (n_test, n_times) survival probability matrix
        surv_times  — (n_times,) evaluation time points
    """
    emb_kwargs = emb_kwargs or {}

    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train   = df_train[feat_cols].values.astype(np.float32)
    t_train   = df_train[duration_col].values.astype(np.float32)
    e_train   = df_train[event_col].values.astype(np.float32)
    X_test    = df_test[feat_cols].values.astype(np.float32)
    y_bin     = (e_train > 0).astype(np.float32)

    print(f"[{fm_name}] Extracting embeddings (head={head_type}) …")
    train_emb, test_emb = embedding_fn(X_train, y_bin, X_test, **emb_kwargs)
    emb_dim = train_emb.shape[1]
    print(f"[{fm_name}] Embedding dim = {emb_dim}")

    config = load_model_config("embedding_head")
    params = config["default"]

    if tune:
        params = tune_embedding_head(
            train_emb, t_train, e_train,
            emb_dim=emb_dim,
            head_type=head_type,
            n_trials=n_trials,
            save_dir=save_dir,
            fm_name=fm_name,
            study_id=study_id,
            fixed_params=params,
            num_durations=num_durations,
        )

    model = EmbeddingSurvHead(
        embedding_dim=emb_dim,
        head_type=head_type,
        num_durations=num_durations,
        num_nodes=params["nodes"],
        dropout=params["dropout"],
        lr=params["lr"],
    )
    model.fit(train_emb, t_train, e_train,
              epochs=200, batch_size=params["batch_size"], verbose=True)
    if head_type == "cox":
        model.compute_baseline()

    surv_df     = model.predict_survival(test_emb)
    
    # Use integrated survival (Area Under Curve) as a more robust risk score.
    # Lower AUC (shorter life expectancy) = Higher risk.
    times = surv_df.index.values
    surv_array = surv_df.values  # (n_times, n_subjects)
    # Average survival over time for each subject
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise AttributeError("module 'numpy' has no attribute 'trapezoid' or 'trapz'")
    auc = _trapz(surv_array.T, times)
    risk_scores = -auc

    return model, risk_scores, surv_df.values.T, surv_df.index.values
