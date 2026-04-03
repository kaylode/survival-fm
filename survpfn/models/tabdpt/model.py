"""
survpfn.models.tabdpt.model — TabDPT backbone + survival head (jointly-trained).

Classes / functions
-------------------
* TabDPTSurvModel  — PyTorch nn.Module: TabDPTModel backbone + survival head MLP.
                     Gradients flow from the survival loss back through the full
                     TabDPT transformer (no torch.no_grad() wrapper).
* TabDPTSurvPH     — High-level wrapper with fit / predict_survival_df API
                     (compatible with the benchmark runner).

Context management
------------------
During training each mini-batch draws a random K ≤ 128 context from the rest
of the training set.  After training a fixed context subset is stored in the
model so that pycox can call ``net(x_query)`` without needing an explicit
context argument.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt

from survpfn.models.tabdpt.tabdpt.model import TabDPTModel, TABDPT_CHECKPOINT_PATH
from survpfn.models.tabdpt.tabdpt.models_utils import pad_x


_DEFAULT_CONTEXT_SIZE = 128


# ---------------------------------------------------------------------------
# TabDPT Survival Model (Backbone + Survival Head, gradient-enabled)
# ---------------------------------------------------------------------------

class TabDPTSurvModel(nn.Module):
    """TabDPT backbone with a survival head MLP.

    The backbone's ``forward()`` is called *without* ``torch.no_grad()``,
    so gradients propagate through the transformer encoder into the survival
    loss.

    Parameters
    ----------
    tabdpt_model    : pre-loaded ``TabDPTModel`` instance.
    n_out           : 1 for Cox; ``num_durations`` for discrete heads.
    head_num_nodes  : hidden layer widths of the survival head MLP.
    dropout         : dropout applied between head layers.
    freeze_tabdpt   : if True only the survival head is trainable.
    """

    def __init__(
        self,
        tabdpt_model: TabDPTModel,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        freeze_tabdpt: bool = False,
    ):
        super().__init__()
        self.tabdpt = tabdpt_model
        self.max_features: int = tabdpt_model.num_features
        self.ninp: int = tabdpt_model.ninp

        for param in self.tabdpt.parameters():
            param.requires_grad = not freeze_tabdpt

        # Survival Head: ninp → head_num_nodes → n_out
        nodes = [self.ninp] + list(head_num_nodes)
        layers: list[nn.Module] = []
        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(nodes[-1], n_out, bias=False))
        self.survival_head = nn.Sequential(*layers)

        # Stored context — set once after training; used during inference
        self._ctx_x: Optional[torch.Tensor] = None  # (K, 1, max_features)
        self._ctx_y: Optional[torch.Tensor] = None  # (K,)  float
        self._pca_v: Optional[torch.Tensor] = None  # (n_features, max_features)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def set_context(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
    ) -> None:
        """Store context for use during prediction (after training).

        Args:
            x_context: (K, 1, max_features) — already padded/projected.
            y_context: (K,) float — binary event indicator (0 / 1).
        """
        self._ctx_x = x_context
        self._ctx_y = y_context

    def set_pca(self, V: Optional[torch.Tensor]) -> None:
        """Store PCA projection matrix (used when n_features > max_features)."""
        self._pca_v = V

    def _to_padded(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, n_features) → (N, 1, max_features).

        Applies PCA projection first if stored, then zero-pads to max_features.
        """
        if self._pca_v is not None:
            x = x @ self._pca_v                  # (N, max_features)
        x = x.unsqueeze(1)                        # (N, 1, n_features)
        n_feat = x.shape[-1]
        if n_feat < self.max_features:
            x = pad_x(x, self.max_features)       # (N, 1, max_features)
        elif n_feat > self.max_features:
            x = x[..., : self.max_features]
        return x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Jointly-trained forward pass.

        Args:
            x_query:   (B, n_features) — query samples (current batch).
            x_context: (K, 1, max_features) — context features.
                       Defaults to stored context when None.
            y_context: (K,) float — context binary event labels.
                       Defaults to stored context when None.

        Returns:
            (B,) for Cox; (B, num_durations) for discrete heads.
        """
        if x_context is None:
            x_context = self._ctx_x
            y_context = self._ctx_y
        if x_context is None:
            raise RuntimeError(
                "No context supplied and none stored. "
                "Call set_context() before prediction, or supply context explicitly."
            )

        K = x_context.shape[0]

        # Map query to (B, 1, max_features)
        x_qry = self._to_padded(x_query)

        # Concatenate: (K + B, 1, max_features)
        x_full = torch.cat([x_context, x_qry], dim=0)

        # TabDPTModel.forward(x, y, eval_pos) — NO torch.no_grad()
        # return_embeddings=True → returns (pred[K:], src_full)
        # src_full: (K+B, 1, ninp)  → query embeddings: src_full[K:, 0, :]
        _, src = self.tabdpt(
            x_full,
            y_context.float(),
            eval_pos=K,
            return_embeddings=True,
        )
        query_embs = src[K:, 0, :]   # (B, ninp)

        out = self.survival_head(query_embs)   # (B, n_out)
        if out.size(-1) == 1:
            out = out.squeeze(-1)              # (B,) for Cox
        return out


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class TabDPTSurvPH:
    """Jointly-trained TabDPT + survival head.

    Benchmark-compatible: returns ``(model, risk, surv_probs, surv_times)``.

    Parameters
    ----------
    head_type       : "cox" | "deephit" | "pchazard" | "mtlr"
    checkpoint_path : path to TabDPT ``.pt`` / ``.ckpt`` checkpoint.
                      Falls back to env var ``TABDPT_CHECKPOINT``.
    context_size    : context samples per training step (≤ 128 recommended).
    freeze_tabdpt   : train survival head only (faster but weaker).
    """

    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 100,
        head_num_nodes: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        freeze_tabdpt: bool = True,
        device: str = "cuda:0",
        checkpoint_path: Optional[str] = None,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        n_out: Optional[int] = None,
    ):
        self.head_type = head_type.lower()
        self.device = device
        self.context_size = context_size
        self.num_durations = num_durations

        # ── Load backbone ───────────────────────────────────────────────

        checkpoint = torch.load(TABDPT_CHECKPOINT_PATH, map_location=device, weights_only=False)
        tabdpt_model = TabDPTModel.load(
            model_state=checkpoint["model"],
            config=checkpoint["cfg"],
        )
        # tabdpt_model.train()   # enable training mode for gradient flow
        tabdpt_model.eval()   # enable training mode for gradient flow

        if n_out is None:
            n_out = 1 if self.head_type == "cox" else num_durations

        self.net = TabDPTSurvModel(
            tabdpt_model=tabdpt_model,
            n_out=n_out,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            freeze_tabdpt=True,
        ).to(device)

        # ── pycox model + optimizer ──────────────────────────────────────
        from pycox.models import CoxPH, DeepHitSingle, PCHazard, MTLR

        if self.head_type == "cox":
            self.model = CoxPH(self.net, tt.optim.Adam(lr=learning_rate))
            self.labtrans = None

        elif self.head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            self.labtrans = LabTransDiscreteTime(num_durations, scheme="quantiles")
            self.model = DeepHitSingle(
                self.net, tt.optim.Adam(lr=learning_rate),
                duration_index=self.labtrans.cuts,
            )

        elif self.head_type == "pchazard":
            try:
                self.labtrans = PCHazard.label_transform(num_durations, scheme="quantiles")
            except TypeError:
                self.labtrans = PCHazard.label_transform(num_durations)
            self.model = PCHazard(
                self.net, tt.optim.Adam(lr=learning_rate),
                duration_index=self.labtrans.cuts,
            )

        elif self.head_type == "mtlr":
            try:
                self.labtrans = MTLR.label_transform(num_durations, scheme="quantiles")
            except TypeError:
                self.labtrans = MTLR.label_transform(num_durations)
            self.model = MTLR(
                self.net, tt.optim.Adam(lr=learning_rate),
                duration_index=self.labtrans.cuts,
            )

        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> "TabDPTSurvPH":
        """Train the backbone + head end-to-end.

        Parameters
        ----------
        x         : (N, n_features) float array — already standardised.
        durations : (N,) event / censoring times.
        events    : (N,) 0/1 event indicator.
        epochs    : training epochs.
        batch_size: mini-batch size.
        verbose   : print loss every 10 epochs.
        """
        from pycox.models.loss import (
            CoxPHLoss,
            DeepHitSingleLoss,
            NLLPCHazardLoss,
            NLLMTLRLoss,
        )
        from pycox.models.data import pair_rank_mat

        if self.head_type == "cox":
            criterion_surv = CoxPHLoss()
        elif self.head_type == "deephit":
            criterion_surv = DeepHitSingleLoss(alpha=0.2, sigma=0.1)
        elif self.head_type == "pchazard":
            criterion_surv = NLLPCHazardLoss()
        else:  # mtlr
            criterion_surv = NLLMTLRLoss()

        optimizer = self.model.optimizer
        N = len(x)

        # ── Tensors ─────────────────────────────────────────────────────
        x_pt = torch.from_numpy(x.copy()).float().to(self.device)

        # PCA if feature count exceeds backbone capacity (rare for typical datasets)
        if x.shape[1] > self.net.max_features:
            _, _, V = torch.pca_lowrank(x_pt, q=self.net.max_features)
            self.net.set_pca(V)

        # Label transforms for discrete heads
        if self.labtrans is not None:
            targets = self.labtrans.fit_transform(durations, events)
            dur_pt = torch.from_numpy(targets[0]).to(self.device)
            ev_pt = torch.from_numpy(targets[1]).to(self.device)
            frac_pt = (
                torch.from_numpy(targets[2]).float().to(self.device)
                if len(targets) > 2
                else None
            )
        else:
            dur_pt = torch.from_numpy(durations.copy()).float().to(self.device)
            ev_pt = torch.from_numpy(events.copy()).float().to(self.device)
            frac_pt = None

        # Binary event indicator for context labels (always 0/1)
        y_bin_pt = torch.from_numpy((events > 0).astype(np.float32)).to(self.device)

        # Pre-pad full training set once (context sampling re-uses this)
        x_padded = self.net._to_padded(x_pt)   # (N, 1, max_features)

        # ── Training loop ────────────────────────────────────────────────
        for epoch in range(epochs):
            self.net.train()
            indices = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                batch_idx = indices[i : i + batch_size]
                bx   = x_pt[batch_idx]
                bdur = dur_pt[batch_idx]
                bev  = ev_pt[batch_idx]
                bfrac = frac_pt[batch_idx] if frac_pt is not None else None

                # ── Random context: K samples not in current batch ───────
                context_mask = torch.ones(N, dtype=torch.bool, device=self.device)
                context_mask[batch_idx] = False
                pool = context_mask.nonzero(as_tuple=True)[0]
                n_ctx = min(self.context_size, len(pool))
                perm = torch.randperm(len(pool), device=self.device)
                ctx_idx = pool[perm[:n_ctx]]

                x_ctx = x_padded[ctx_idx]    # (K, 1, max_features)
                y_ctx = y_bin_pt[ctx_idx]    # (K,)

                # ── Forward ──────────────────────────────────────────────
                optimizer.zero_grad()
                head_out = self.net(bx, x_context=x_ctx, y_context=y_ctx)

                # ── Loss ─────────────────────────────────────────────────
                if self.head_type == "cox":
                    loss = criterion_surv(head_out, bdur, bev)
                elif self.head_type == "deephit":
                    _bdur = bdur.cpu().numpy()
                    _bev  = bev.cpu().numpy()
                    rank_mat = torch.from_numpy(
                        pair_rank_mat(_bdur, _bev)
                    ).float().to(self.device)
                    loss = criterion_surv(head_out, bdur, bev, rank_mat)
                elif self.head_type == "pchazard":
                    loss = criterion_surv(head_out, bdur, bev, bfrac)
                else:  # mtlr
                    loss = criterion_surv(head_out, bdur, bev)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if verbose and epoch % 10 == 0:
                avg = epoch_loss / max(1, n_batches)
                print(
                    f"[TabDPTSurv/{self.head_type}] "
                    f"epoch {epoch:3d}/{epochs}  loss={avg:.4f}"
                )

        # ── Store fixed context for prediction (pycox calls net(x)) ─────
        ctx_size = min(self.context_size, N)
        ctx_perm = torch.randperm(N)[:ctx_size]
        self.net.set_context(
            x_context=x_padded[ctx_perm].detach(),
            y_context=y_bin_pt[ctx_perm].detach(),
        )

        # ── Baseline hazards (Cox only) ──────────────────────────────────
        if self.head_type == "cox":
            self.net.eval()
            with torch.no_grad():
                self.model.compute_baseline_hazards(
                    input=x_pt, target=(dur_pt, ev_pt)
                )

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_survival_df(self, x: np.ndarray):
        """Return a DataFrame of survival probabilities (rows = times, cols = subjects)."""
        self.net.eval()
        x_pt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            if self.head_type == "cox":
                return self.model.predict_surv_df(x_pt)
            if hasattr(self.model, "interpolate"):
                return self.model.interpolate(10).predict_surv_df(x_pt)
            return self.model.predict_surv_df(x_pt)
