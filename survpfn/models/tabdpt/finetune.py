"""
survpfn.models.tabdpt.finetune — TabDPT finetuning for survival analysis.

Architecture
------------
Mirrors ``tabpfn/finetune.py`` in structure but drives a *custom* PyTorch
training loop instead of delegating to ``FinetunedTabPFNClassifier``.

Key design decisions
--------------------
1. **Survival → binary classification via temporal expansion** (identical to
   TabPFN variant): each (patient, time-bin) pair that has definitive
   information becomes a training row with binary label ``y=1`` if the event
   occurred by that bin, else ``y=0``.

2. **Custom ``TabDPTFinetuneTrainer``**: wraps the raw ``TabDPTModel`` backbone
   and routes gradients through the transformer (no ``torch.no_grad()``).
   The backbone's own next-token head is *also* trained via an auxiliary
   cross-entropy loss — the same dual-objective used in ``model.py``.

3. **Context sampling**: each mini-batch draws a random held-out context
   from the *expanded* dataset.  This matches the in-context learning setup
   that TabDPT expects at inference time.

4. **Evaluation**: ``predict_survival_df`` reconstructs per-patient survival
   curves from per-bin probabilities, with isotonic regression for
   monotonicity, identical to the TabPFN variant.

Classes
-------
SurvivalTimeBinEncoder  — re-exported from tabpfn.finetune (shared logic).
TabDPTFinetuneTrainer   — custom nn.Module: backbone + binary classification head.
TabDPTSurvPHFinetune    — high-level sklearn-style wrapper (fit / predict_survival_df).
"""

from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from survpfn.models.tabdpt.tabdpt.model import TabDPTModel, TABDPT_CHECKPOINT_PATH
from survpfn.models.tabdpt.tabdpt.models_utils import pad_x
from survpfn.models.shared.preprocessing import FMDataPrep, SurvivalTimeBinEncoder
from survpfn.models.shared.finetune import BaseSurvExpandedFinetune


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_SIZE = 128
C_YELLOW = "\033[93m"
C_RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Component 1 — TabDPTFinetuneTrainer (custom nn.Module)
# ---------------------------------------------------------------------------

class TabDPTFinetuneTrainer(nn.Module):
    """TabDPT backbone + binary classification head for survival finetuning.

    This module is used **only** for the temporal-expansion training regime
    (survival data converted to per-bin binary classification tasks).  It is
    intentionally separate from ``TabDPTSurvModel`` in ``model.py`` so that
    the two approaches remain independently maintainable.

    Parameters
    ----------
    tabdpt_model  : Pre-loaded ``TabDPTModel`` instance (weights retained).
    hidden_dim    : Width of the single hidden layer in the binary head.
    dropout       : Dropout applied after the hidden activation.
    freeze_backbone : If True, freeze all backbone parameters and only train
                      the classification head.  Usually False (joint training).
    """

    def __init__(
        self,
        tabdpt_model: TabDPTModel,
        n_out: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.tabdpt       = tabdpt_model
        self.max_features: int = tabdpt_model.num_features
        self.ninp: int         = tabdpt_model.ninp
        self.n_out: int        = n_out

        # Optionally freeze backbone
        for param in self.tabdpt.parameters():
            param.requires_grad = not freeze_backbone

        # Layer-norm → 2-layer binary MLP
        self.norm = nn.LayerNorm(self.ninp)
        self.classifier = nn.Sequential(
            nn.Linear(self.ninp, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_out),   # logits for {0, 1, ..., n_out-1}
        )

        # Inference-time context (populated by set_context after training)
        self._ctx_x: Optional[torch.Tensor] = None   # (K, 1, max_features)
        self._ctx_y: Optional[torch.Tensor] = None   # (K,)  float binary

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _to_padded(self, x: torch.Tensor) -> torch.Tensor:
        """(N, n_features) → (N, 1, max_features)."""
        x = x.unsqueeze(1)                         # (N, 1, n_features)
        n_feat = x.shape[-1]
        if n_feat < self.max_features:
            x = pad_x(x, self.max_features)
        elif n_feat > self.max_features:
            x = x[..., : self.max_features]
        return x

    def set_context(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
    ) -> None:
        """Store fixed context for inference.

        Args:
            x_context: (K, 1, max_features) — already padded.
            y_context: (K,) float binary labels.
        """
        self._ctx_x = x_context
        self._ctx_y = y_context

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        return_backbone_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through TabDPT + binary head.

        Args:
            x_query:  (B, n_features) — query samples for the current batch.
            x_context: (K, 1, max_features) — in-context samples.
                       Falls back to stored context when ``None``.
            y_context: (K,) float — binary labels for context samples.
                       Falls back to stored context when ``None``.
            return_backbone_logits: If True, also return the backbone's own
                classification logits ``(B, 2)`` used for the auxiliary loss.

        Returns:
            cls_logits: (B, n_out) — classification logits from the head.
            backbone_logits (optional): (B, 2) — TabDPT's own next-token logits.
        """
        if x_context is None:
            x_context = self._ctx_x
            y_context = self._ctx_y
        if x_context is None:
            raise RuntimeError(
                "No context supplied and none stored. "
                "Call set_context() or pass x_context explicitly."
            )

        device = x_query.device
        self.to(device)
        x_context = x_context.to(device)
        y_context  = y_context.to(device)

        K = x_context.shape[0]

        # Map query to padded format
        x_qry = self._to_padded(x_query)           # (B, 1, max_features)

        # Concatenate context + query
        x_full = torch.cat([x_context, x_qry], dim=0)  # (K+B, 1, max_features)

        # TabDPTModel forward: returns (dpt_pred, src)
        #   dpt_pred: (B, 1, n_classes+1)  — backbone class logits for query
        #   src:      (K+B, 1, ninp)       — transformer hidden states
        dpt_pred, src = self.tabdpt(
            x_full,
            y_context.float(),
            eval_pos=K,
            return_embeddings=True,
        )

        # Extract query embeddings
        query_embs = src[K:, 0, :].to(torch.float32)   # (B, ninp)

        # Head: LayerNorm → MLP → logits
        cls_logits = self.classifier(self.norm(query_embs))   # (B, n_out)

        if return_backbone_logits:
            # Backbone's own binary logits (first 2 class slots)
            backbone_logits = dpt_pred[:, 0, :2].to(torch.float32)  # (B, 2)
            return cls_logits, backbone_logits

        return cls_logits

    # ------------------------------------------------------------------
    # predict_proba (inference only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return (N, n_out) probability array.
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        logits = self(x)                             # (N, n_out)
        return F.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Component 2 — High-level wrapper
# ---------------------------------------------------------------------------

class TabDPTSurvPHFinetune(BaseSurvExpandedFinetune):
    """Survival model: TabDPT backbone finetuned via temporal-expansion BCE.

    API mirrors ``TabPFNSurvPHFinetune`` so it can be used as a drop-in
    replacement in the benchmark runner.

    Training strategy
    -----------------
    1. Fit a ``SurvivalTimeBinEncoder`` to extract K time bins + KM features.
    2. Expand the dataset: each (patient, bin_k) pair with definitive
       information becomes one training row
           ``[x_patient || bin_features_k]  →  y_binary_k``
    3. Fine-tune ``TabDPTFinetuneTrainer`` (backbone + head) on this expanded
       dataset.  A random context of size ``context_size`` is drawn per batch.
    4. At inference, per-bin survival probabilities are assembled into a
       monotone survival curve via isotonic regression.

    Parameters
    ----------
    num_durations       : Number of time bins K for the encoder.
    learning_rate       : Adam learning rate.
    epochs              : Maximum training epochs.
    batch_size          : Query batch size per step.
    context_size        : Context samples per step (≤ 128 recommended for TabDPT).
    alpha               : Weight of backbone auxiliary loss.
    hidden_dim          : Hidden width of the classification head.
    dropout             : Dropout in the classification head.
    freeze_backbone     : If True train only the head (faster, less powerful).
    patience            : Early-stopping patience (in epochs).
    device              : Torch device string (e.g. ``"cuda:0"``).
    random_state        : Numpy random seed for reproducibility.
    """

    def __init__(
        self,
        num_durations: int = 50,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        self.context_size    = context_size
        self.hidden_dim      = hidden_dim
        self.dropout         = dropout
        self.freeze_backbone = freeze_backbone
        self.backbone_name   = "tabdpt"

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Load TabDPT backbone
        checkpoint  = torch.load(TABDPT_CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        tabdpt_model = TabDPTModel.load(
            model_state=checkpoint["model"],
            config=checkpoint["cfg"],
        )
        tabdpt_model.eval()

        self.net = TabDPTFinetuneTrainer(
            tabdpt_model=tabdpt_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        print(
            f"TabDPTSurvPHFinetune initialised  "
            f"(device={self.device}, bins={num_durations}, "
            f"freeze_backbone={freeze_backbone})",
            flush=True,
        )

        # Print parameter summary
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"TabDPTSurvPHFinetune Initialized: {trainable_params:,} trainable / {total_params:,} total parameters "
              f"({trainable_params/total_params:.2%})", flush=True)
        

    def fit(self, x, durations, events, **kwargs):
        # Infer n_out from events
        num_events = int(events.max())
        n_out = num_events + 1
        
        if n_out != self.net.n_out:
            # Re-initialize head with correct dimension
            self.net.n_out = n_out
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, n_out).to(self.device)
            print(f"  [TabDPT/Expanded] Adjusting head for {n_out} classes (competing risks).", flush=True)

        return super().fit(x, durations, events, **kwargs)

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        x_bin_pt = torch.from_numpy(x_bin).float().to(self.device)
        return self.net.predict_proba(x_bin_pt)
