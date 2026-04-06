"""
survpfn.models.beta_surv — BetaSurv: finetunable backbone + frozen TabPFN + survival.

Architecture
------------
Inspired by the Beta model in TALENT (Ye et al., 2024).

    raw features (N, d)
        ↓ trainable MLP backbone
    backbone features (N, d_hidden)
        ↓ frozen TabPFN (ICL binary classifier)
    zero-shot ICL survival (K time-bin binary classification)
        ↓ isotonic regression
    S(t₁), …, S(tₖ) — survival curve

The backbone is trained end-to-end via survival loss.  Gradients flow
through TabPFN's attention computation (not into TabPFN's weights, which
are frozen) back to the backbone parameters.

Key distinction from plain zero-shot ICL (Approach B):
- Approach B:   raw features ──→ TabPFN zero-shot ICL survival   (no learning)
- BetaSurv (E): backbone(raw) ──→ TabPFN zero-shot ICL survival  (learned repr.)

The backbone learns survival-relevant feature transformations that improve
the quality of ICL context for TabPFN.

Zero-shot ICL survival formulation follows Kim, Lai & Zhang (arXiv:2601.22259):
- K=10 quantile-spaced time bins t₁ < … < tₖ
- For bin k: include sample i only if tₖ < Cᵢ; label Yᵢₖ = 1[Tᵢ ≤ tₖ]
- Append tₖ as extra feature (time-aware ICL)
- S(tₖ) = 1 − p̂(tₖ);  isotonic regression enforces monotonicity

Benchmark API
-------------
    train_beta_surv(df_train, df_test, dur_col, ev_col, ...) → (model, risk, surv_probs, surv_times)
"""
from __future__ import annotations

import pathlib
from typing import List, Optional

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.utils.optuna import get_n_trials_to_run


# ---------------------------------------------------------------------------
# Backbone MLP
# ---------------------------------------------------------------------------

class BackboneMLP(nn.Module):
    """Simple MLP feature extractor (trainable).

    Maps (N, d_in) → (N, d_out) with BatchNorm + ReLU + Dropout.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int] = [128, 64],
        d_out: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [d_in] + list(hidden_dims) + [d_out]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# BetaSurv model
# ---------------------------------------------------------------------------

class BetaSurvModel(nn.Module):
    """Backbone MLP + frozen TabPFN as ICL oracle for zero-shot survival.

    The backbone transforms raw features; TabPFN performs zero-shot ICL
    binary classification on the transformed features to predict survival
    at each time bin.  Only backbone parameters receive gradient updates.
    """

    def __init__(
        self,
        n_features: int,
        backbone_hidden: List[int] = [128, 64],
        backbone_out: int = 64,
        dropout: float = 0.1,
        n_bins: int = 10,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.device = device

        # +1 for the appended time feature
        self.backbone = BackboneMLP(
            d_in=n_features + 1,
            hidden_dims=backbone_hidden,
            d_out=backbone_out,
            dropout=dropout,
        )

        # Load frozen TabPFN
        base_path = pathlib.Path(__file__).parent / "tabpfn" / "backbone"
        from survpfn.models.tabpfn.backbone.utils import load_model_workflow
        model_tuple, self.config, _ = load_model_workflow(
            0, 42, add_name="",
            base_path=pathlib.Path(__file__).parent / "tabpfn",
            device=device,
            only_inference=True,
        )
        self.tabpfn = model_tuple[2]
        self.tabpfn.to(torch.float32)
        for param in self.tabpfn.parameters():
            param.requires_grad = False

        self.max_features = self.config.get("num_features", 100)

        # Capture transformer output via hook
        self._tfm_output: Optional[torch.Tensor] = None
        def _hook(module, inp, out):
            self._tfm_output = out
        self.tabpfn.transformer_encoder.register_forward_hook(_hook)

    def _pad_to_tabpfn(self, x: torch.Tensor) -> torch.Tensor:
        """Pad backbone output (N, d) to TabPFN input width (N, max_features)."""
        d = x.size(-1)
        if d < self.max_features:
            pad = torch.zeros(*x.shape[:-1], self.max_features - d,
                              device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)
        return x[..., : self.max_features]

    def forward_bin(
        self,
        x_query_raw: torch.Tensor,
        x_context_raw: torch.Tensor,
        y_context: torch.Tensor,
        t_bin: float,
    ) -> torch.Tensor:
        """Predict P(T ≤ t_bin | x) for query samples using ICL.

        Parameters
        ----------
        x_query_raw   : (B, n_features) raw query features.
        x_context_raw : (K, n_features) raw context features.
        y_context     : (K,) binary labels 1[T ≤ t_bin] for context.
        t_bin         : current time bin value (appended as feature).

        Returns
        -------
        (B,) predicted probability P(T ≤ t_bin | x).
        """
        B = x_query_raw.size(0)
        K = x_context_raw.size(0)

        t_qry = torch.full((B, 1), t_bin, device=self.device, dtype=torch.float32)
        t_ctx = torch.full((K, 1), t_bin, device=self.device, dtype=torch.float32)

        # Concatenate time as extra feature then transform via backbone
        x_qry_aug = torch.cat([x_query_raw,   t_qry], dim=-1)  # (B, d+1)
        x_ctx_aug = torch.cat([x_context_raw, t_ctx], dim=-1)  # (K, d+1)

        emb_qry = self.backbone(x_qry_aug)   # (B, backbone_out)
        emb_ctx = self.backbone(x_ctx_aug)   # (K, backbone_out)

        # Pad to TabPFN input dimension
        emb_qry_padded = self._pad_to_tabpfn(emb_qry)  # (B, max_features)
        emb_ctx_padded = self._pad_to_tabpfn(emb_ctx)  # (K, max_features)

        # Stack for TabPFN: (K+B, max_features), y: (K+B,) with zeros for queries
        x_all = torch.cat([emb_ctx_padded, emb_qry_padded], dim=0)  # (K+B, max_features)
        y_all = torch.cat([y_context.float(),
                           torch.zeros(B, device=self.device)], dim=0)  # (K+B,)

        # TabPFN forward — gradients flow through x_all back to backbone
        logits = self.tabpfn(
            (None, x_all.float(), y_all.float()),
            single_eval_pos=K,
        )  # (B, n_classes)

        # P(event by t_bin) = softmax[1] for binary
        probs = torch.softmax(logits[:, :2], dim=-1)[:, 1]  # (B,)
        return probs


# ---------------------------------------------------------------------------
# BetaSurv high-level wrapper
# ---------------------------------------------------------------------------

class BetaSurvPH:
    """BetaSurv: trainable backbone MLP + frozen TabPFN + zero-shot survival.

    Parameters
    ----------
    n_bins          : number of quantile time bins for survival ICL.
    context_size    : ICL context size (samples per bin).
    backbone_hidden : hidden layer widths for the backbone MLP.
    backbone_out    : output dimension of the backbone MLP.
    lr              : learning rate for backbone.
    epochs          : training epochs.
    batch_size      : number of query samples per training step.
    """

    def __init__(
        self,
        n_bins: int = 10,
        context_size: int = 100,
        backbone_hidden: List[int] = [128, 64],
        backbone_out: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cuda:0",
    ):
        self.n_bins = n_bins
        self.context_size = context_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.backbone_hidden = backbone_hidden
        self.backbone_out = backbone_out
        self.dropout = dropout

        # Filled by fit()
        self.model_: Optional[BetaSurvModel] = None
        self.time_bins_: Optional[np.ndarray] = None
        self._x_train_pt: Optional[torch.Tensor] = None
        self._y_bin_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        verbose: bool = True,
    ) -> "BetaSurvPH":
        """Train the backbone MLP end-to-end via zero-shot ICL survival loss.

        Loss: Binary cross-entropy summed across K bins, using only
        observable (non-censored) label assignments as in Kim et al. 2026.
        """
        N, d = x.shape

        # ── Build model ───────────────────────────────────────────────
        self.model_ = BetaSurvModel(
            n_features=d,
            backbone_hidden=self.backbone_hidden,
            backbone_out=self.backbone_out,
            dropout=self.dropout,
            n_bins=self.n_bins,
            device=self.device,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.backbone.parameters(), lr=self.lr
        )
        criterion = nn.BCELoss(reduction="mean")

        # ── Quantile time bins from observed event times ───────────────
        event_times = durations[events > 0]
        quantiles = np.linspace(0.1, 0.9, self.n_bins)
        self.time_bins_ = np.quantile(event_times, quantiles)

        # ── Pre-compute observable binary labels per bin ───────────────
        # Label Y_{ik} = 1[T_i <= t_k]; only defined if t_k < C_i
        # Shape: (N, n_bins);  NaN = masked (censored before t_k)
        Y_mat = np.full((N, self.n_bins), np.nan, dtype=np.float32)
        for k, tk in enumerate(self.time_bins_):
            observable = durations > tk   # t_k < C_i (or T_i)
            Y_mat[observable, k] = (durations[observable] <= tk).astype(np.float32)
            # Event cases: T_i <= t_k → 1; censored/event cases T_i > t_k → 0
            # Purely censored with C_i <= t_k → NaN (not observable)

        x_pt = torch.from_numpy(x.astype(np.float32)).to(self.device)
        self._x_train_pt = x_pt
        self._y_bin_matrix = Y_mat

        # Normalise time bins to [0, 1] for numerical stability
        t_max = float(self.time_bins_.max()) + 1e-8
        t_bins_norm = self.time_bins_ / t_max

        # ── Training loop ─────────────────────────────────────────────
        for epoch in range(self.epochs):
            self.model_.backbone.train()
            indices = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, self.batch_size):
                batch_idx = indices[i : i + self.batch_size].numpy()
                x_query = x_pt[batch_idx]

                # Context: random subset of training data
                context_mask = np.ones(N, dtype=bool)
                context_mask[batch_idx] = False
                pool = np.where(context_mask)[0]
                n_ctx = min(self.context_size, len(pool))
                ctx_idx = np.random.choice(pool, n_ctx, replace=False)
                x_ctx = x_pt[ctx_idx]

                total_loss = torch.tensor(0.0, device=self.device)
                n_terms = 0

                for k, (tk, tk_norm) in enumerate(
                    zip(self.time_bins_, t_bins_norm)
                ):
                    # Only use context samples with observable labels at bin k
                    ctx_labels_k = Y_mat[ctx_idx, k]
                    obs_mask_ctx = ~np.isnan(ctx_labels_k)
                    if obs_mask_ctx.sum() < 2:
                        continue

                    y_ctx_k = torch.from_numpy(
                        ctx_labels_k[obs_mask_ctx]
                    ).float().to(self.device)
                    x_ctx_obs = x_ctx[torch.from_numpy(obs_mask_ctx)]

                    # Query labels (for loss)
                    q_labels_k = Y_mat[batch_idx, k]
                    obs_mask_q = ~np.isnan(q_labels_k)
                    if obs_mask_q.sum() < 1:
                        continue

                    y_q_true = torch.from_numpy(
                        q_labels_k[obs_mask_q]
                    ).float().to(self.device)
                    x_qry_obs = x_query[torch.from_numpy(obs_mask_q)]

                    probs = self.model_.forward_bin(
                        x_qry_obs, x_ctx_obs, y_ctx_k, float(tk_norm)
                    )
                    total_loss = total_loss + criterion(probs, y_q_true)
                    n_terms += 1

                if n_terms == 0:
                    continue

                total_loss = total_loss / n_terms
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1

            if verbose and epoch % 10 == 0:
                avg = epoch_loss / max(1, n_batches)
                print(f"[BetaSurv] epoch {epoch:3d}/{self.epochs}  loss={avg:.4f}")

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_survival_df(self, x: np.ndarray) -> pd.DataFrame:
        """Return survival DataFrame (rows=times, cols=subjects).

        Uses zero-shot ICL with stored training context.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")

        self.model_.backbone.eval()
        x_pt = torch.from_numpy(x.astype(np.float32)).to(self.device)

        t_max = float(self.time_bins_.max()) + 1e-8
        t_bins_norm = self.time_bins_ / t_max

        N = len(x)
        K = self.n_bins
        surv = np.ones((N, K), dtype=np.float32)

        # Use a random context from training data
        n_ctx = min(self.context_size, len(self._x_train_pt))
        ctx_perm = np.random.permutation(len(self._x_train_pt))[:n_ctx]
        x_ctx = self._x_train_pt[ctx_perm]

        with torch.no_grad():
            for k, (tk, tk_norm) in enumerate(
                zip(self.time_bins_, t_bins_norm)
            ):
                ctx_labels_k = self._y_bin_matrix[ctx_perm, k]
                obs_mask = ~np.isnan(ctx_labels_k)
                if obs_mask.sum() < 2:
                    # Fall back: S(t_k) ≈ previous value
                    if k > 0:
                        surv[:, k] = surv[:, k - 1]
                    continue

                y_ctx_k = torch.from_numpy(
                    ctx_labels_k[obs_mask]
                ).float().to(self.device)
                x_ctx_obs = x_ctx[torch.from_numpy(obs_mask)]

                # Predict in mini-batches to avoid OOM
                probs_list = []
                for i in range(0, N, 256):
                    p = self.model_.forward_bin(
                        x_pt[i : i + 256], x_ctx_obs, y_ctx_k, float(tk_norm)
                    )
                    probs_list.append(p.cpu().numpy())
                probs_np = np.concatenate(probs_list)  # (N,)
                surv[:, k] = 1.0 - probs_np

        # Enforce monotonically non-increasing survival per subject
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        for i in range(N):
            surv[i] = iso.fit_transform(t_bins_norm, surv[i])
        surv = np.clip(surv, 0.0, 1.0)

        # pycox convention: rows = times, cols = subjects
        return pd.DataFrame(surv.T, index=self.time_bins_)


# ---------------------------------------------------------------------------
# Benchmark API
# ---------------------------------------------------------------------------

def train_beta_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    n_bins: int = 10,
    context_size: int = 100,
    backbone_hidden: Optional[List[int]] = None,
    backbone_out: int = 64,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cuda:0",
    tune: bool = False,
    n_trials: int = 20,
    out_dir: str = "results",
    random_state: int = 42,
    study_id: str = None,
    **kwargs,
) -> tuple:
    """Train BetaSurv and return benchmark-compatible outputs.

    Returns
    -------
    (model, risk, surv_probs, surv_times)
    """
    if backbone_hidden is None:
        backbone_hidden = [128, 64]

    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    x_train = df_train[feat_cols].values.astype(np.float32)
    x_test  = df_test[feat_cols].values.astype(np.float32)
    dur_tr  = df_train[duration_col].values.astype(np.float32)
    ev_tr   = df_train[event_col].values.astype(np.float32)

    # ── Optional HPO ────────────────────────────────────────────────────
    if tune:
        config = load_model_config("beta_surv")
        tuning_cfg = config["tuning"]
        params = config["default"].copy()

        # Update params with passed-in values if they differ from defaults
        params.update({
            "lr": lr,
            "context_size": context_size,
            "backbone_dim": backbone_out,
            "epochs": epochs,
            "batch_size": batch_size,
        })

        os.makedirs(out_dir, exist_ok=True)
        log_name = f"optuna_beta_surv_{study_id}.log" if study_id else "optuna_beta_surv.log"
        log_file = os.path.join(out_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))

        study_name = f"beta_surv_tuning_{study_id}" if study_id else "beta_surv_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial: optuna.Trial) -> float:
            p = params.copy()
            p.update(apply_tuning_params(trial, tuning_cfg))
            
            # Use smaller epochs for speed in tuning if possible
            # or just use the p["epochs"]
            ph = BetaSurvPH(
                lr=p["lr"], 
                backbone_out=p["backbone_dim"], 
                backbone_hidden=backbone_hidden,
                epochs=50, # Slightly faster tuning
                context_size=p["context_size"],
                device=device,
            )
            ph.fit(x_train, dur_tr, ev_tr, verbose=False)
            surv_df = ph.predict_survival_df(x_train)
            from pycox.evaluation import EvalSurv
            ev = EvalSurv(surv_df, dur_tr, ev_tr, censor_surv="km")
            return ev.concordance_td("antolini")

        n_remaining = get_n_trials_to_run(study, n_trials)
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)
        best = study.best_params
        lr           = best.get("lr", lr)
        context_size = best.get("context_size", context_size)
        backbone_out = best.get("backbone_dim", backbone_out)
     

    # ── Train ─────────────────────────────────────────────────────────
    ph = BetaSurvPH(
        n_bins=n_bins,
        context_size=context_size,
        backbone_hidden=backbone_hidden,
        backbone_out=backbone_out,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    ph.fit(x_train, dur_tr, ev_tr, verbose=False)

    # ── Predict ────────────────────────────────────────────────────────
    surv_df = ph.predict_survival_df(x_test)
    times      = surv_df.index.values
    surv_array = surv_df.values           # (n_times, N_test)

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc  = _trapz(surv_array.T, times)
    risk = -auc

    return ph, risk, surv_array.T, times
