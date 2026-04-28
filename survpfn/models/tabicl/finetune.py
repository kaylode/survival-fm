"""
survpfn.models.tabicl.finetune — TabICL finetuning for survival analysis.

Architecture
------------
Mirrors ``tabdpt/finetune.py`` in structure but adapts to TabICL's distinct
forward API:

  tabicl(X_t, y_train=y_t, return_embeddings=True)
    X_t      : (1, K+B, n_features)  — context rows first, then query rows
    y_train  : (1, K)  int64        — context binary labels
    returns  : (logits, emb)
    logits   : (1, B, n_classes)
    emb      : (1, K+B, icl_dim)   — icl_dim = embed_dim × row_num_cls (≈512)

Key design decisions
--------------------
1. **Survival → binary classification via temporal expansion** (identical to
   TabPFN / TabDPT variants): each valid (patient, bin_k) pair becomes a
   training row  ``[x_patient || bin_features_k] → y_binary_k``.

2. **Custom ``TabICLFinetuneTrainer``**: wraps the raw ``TabICL`` backbone and
   a two-layer MLP classification head.  The backbone is kept in eval() at all
   times (via a train()-override) so that its internal batch-norm stats are not
   corrupted; gradients still flow through it when ``freeze_backbone=False``.

3. **Context sampling**: each mini-batch draws a random held-out context from
   the *expanded* dataset, matching TabICL's in-context-learning inference.

4. **No padding required**: TabICL handles variable feature counts internally,
   so the temporal bin features (6-dim) are simply concatenated to X before
   being passed to the backbone.

5. **Evaluation**: ``predict_survival_df`` assembles per-bin survival curves
   with isotonic-regression monotonicity enforcement.

Classes
-------
TabICLFinetuneTrainer   — custom nn.Module: backbone + binary classification head.
TabICLSurvPHFinetune    — high-level sklearn-style wrapper (fit / predict_survival_df).
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from survpfn.models.shared.preprocessing import FMDataPrep, SurvivalTimeBinEncoder
from survpfn.models.shared.finetune import BaseSurvExpandedFinetune


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_SIZE = 256   # TabICL supports larger contexts than TabDPT
C_YELLOW = "\033[93m"
C_RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Component 1 — TabICLFinetuneTrainer (custom nn.Module)
# ---------------------------------------------------------------------------

class TabICLFinetuneTrainer(nn.Module):
    """TabICL backbone + binary classification head for survival finetuning.

    The backbone is always kept in eval() mode (via a ``train()`` override)
    so that its internal inference path (``_inference_forward``) is always
    active.  Gradients still flow through it unless ``freeze_backbone=True``.

    Parameters
    ----------
    tabicl_model    : Pre-loaded ``TabICL`` nn.Module (from TabICLClassifier).
    hidden_dim      : Width of the single hidden layer in the binary head.
    dropout         : Dropout applied after the hidden activation.
    freeze_backbone : If True, freeze backbone params; only train the head.
    """

    def __init__(
        self,
        tabicl_model: nn.Module,
        n_out: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.tabicl = tabicl_model
        self.n_out = n_out

        # Embedding dimension: embed_dim × row_num_cls  (default 128×4 = 512)
        self.icl_dim: int = tabicl_model.embed_dim * tabicl_model.row_num_cls

        # Freeze / unfreeze backbone
        for param in self.tabicl.parameters():
            param.requires_grad = not freeze_backbone

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.icl_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_out),   # {0, 1, ..., n_out-1}
        )

        # Inference-time context (set after training via set_context)
        self._ctx_x: Optional[torch.Tensor] = None   # (K, n_features)
        self._ctx_y: Optional[torch.Tensor] = None   # (K,) int64

    # ------------------------------------------------------------------
    # Keep backbone in eval() at all times
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "TabICLFinetuneTrainer":
        """Keep TabICL backbone permanently in eval mode."""
        super().train(mode)
        # self.tabicl.eval()
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_context(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
    ) -> None:
        """Store fixed context for inference.

        Args:
            x_context: (K, n_features) float tensor.
            y_context: (K,) int64 binary labels.
        """
        self._ctx_x = x_context
        self._ctx_y = y_context

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        return_backbone_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward through TabICL backbone → binary head.

        Args:
            x_query:   (B, n_features) — query samples.
            x_context: (K, n_features) — ICL context features.
                       Falls back to stored context when ``None``.
            y_context: (K,) int64 — context binary labels.
                       Falls back to stored context when ``None``.
            return_backbone_logits: Also return backbone's own binary logits.

        Returns:
            cls_logits        : (B, n_out) from the head.
            backbone_logits   : (B, 2) from TabICL's own head  [optional].
        """
        if x_context is None:
            x_context = self._ctx_x
            y_context = self._ctx_y
        if x_context is None:
            raise RuntimeError(
                "No context supplied and none stored. "
                "Call set_context() or pass x_context explicitly."
            )

        # Derive device from the input tensor and ensure entire module follows.
        # This is robust to TabICL backbone being loaded on CPU before .to(device).
        device = x_query.device
        self.to(device)
        x_context = x_context.to(device)
        y_context = y_context.to(device)

        K = x_context.shape[0]
        B = x_query.shape[0]

        # Concatenate  context || query → (K+B, n_features)
        X_all = torch.cat([x_context, x_query], dim=0)

        # TabICL expects (1, K+B, n_features) and y_train=(1, K) int64
        X_t = X_all.unsqueeze(0)                  # (1, K+B, n_features)
        y_t = y_context.long().unsqueeze(0)        # (1, K)

        # Backbone forward (always in eval/inference mode)
        # icl_logits: (1, B, n_classes)
        # emb:        (1, K+B, icl_dim)
        icl_logits, emb = self.tabicl(X_t, y_train=y_t, return_embeddings=True)
        if emb is None:
            raise RuntimeError(
                "TabICL did not return embeddings. "
                "Ensure the checkpoint supports return_embeddings=True."
            )

        # Query embeddings: (B, icl_dim) — force float32 and correct device
        query_embs = emb[0, K:, :].to(device=device, dtype=torch.float32)

        # Head forward
        cls_logits = self.classifier(query_embs)   # (B, n_out)

        if return_backbone_logits:
            # First 2 class columns from backbone — force float32 and correct device
            backbone_logits = icl_logits[0, :, :2].to(device=device, dtype=torch.float32)  # (B, 2)
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
        logits = self(x)                              # (N, n_out) — device-routed by forward
        return F.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Component 2 — High-level wrapper
# ---------------------------------------------------------------------------

class TabICLSurvPHFinetune(BaseSurvExpandedFinetune):
    """Survival model: TabICL backbone finetuned via temporal-expansion BCE.

    API mirrors ``TabDPTSurvPHFinetune`` / ``TabPFNSurvPHFinetune`` for
    drop-in use in the benchmark runner.

    Training strategy
    -----------------
    1. Fit a ``SurvivalTimeBinEncoder`` to extract K time bins + KM features.
    2. Expand the dataset: each valid (patient, bin_k) pair → one training row
           ``[x_patient || bin_feats_k] → y_binary_k``
    3. Fine-tune ``TabICLFinetuneTrainer`` (backbone + head) on the expanded
       dataset.  A random context of size ``context_size`` is drawn per batch.
    4. At inference, per-bin survival probabilities are assembled into a
       monotone survival curve via isotonic regression.

    Parameters
    ----------
    num_durations   : Number of time bins K for the encoder.
    learning_rate   : AdamW learning rate.
    epochs          : Maximum training epochs.
    batch_size      : Query batch size per training step.
    context_size    : ICL context size per step (256 default for TabICL).
    hidden_dim      : Hidden width of the classification head.
    dropout         : Dropout in the classification head.
    freeze_backbone : If True, only train the head (faster, less powerful).
    patience        : Early-stopping patience (epochs without improvement).
    device          : Torch device string (e.g. ``"cuda:0"``).
    random_state    : Numpy/torch seed for reproducibility.
    checkpoint_version : TabICL HuggingFace checkpoint name.
    model_path      : Local checkpoint path; None = auto-download.
    """

    def __init__(
        self,
        num_durations: int = 50,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        self.context_size     = context_size
        self.hidden_dim       = hidden_dim
        self.dropout          = dropout
        self.freeze_backbone  = freeze_backbone
        self.backbone_name    = "tabicl"

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # ── Load TabICL backbone ──────────────────────────────────────────
        from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier

        clf = TabICLClassifier(
            n_estimators=1,
            norm_methods="none",
            feat_shuffle_method="none",
            use_amp=False,
            allow_auto_download=True,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            device=self.device,
            random_state=self.random_state,
            verbose=False,
        )
        # Warm-up to trigger checkpoint load
        _X_d = np.random.randn(10, 4).astype(np.float32)
        _y_d = np.array([0, 1] * 5)
        try:
            clf.fit(_X_d, _y_d)
            clf.predict_proba(_X_d[:2])
        except Exception:
            pass

        tabicl_model: nn.Module = clf.model_
        # tabicl_model.eval()

        self.net = TabICLFinetuneTrainer(
            tabicl_model=tabicl_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        total_params     = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(
            f"TabICLSurvPHFinetune initialised  "
            f"(device={self.device}, bins={num_durations}, "
            f"freeze_backbone={freeze_backbone})  "
            f"{trainable_params:,} trainable / {total_params:,} total params "
            f"({trainable_params/total_params:.2%})",
            flush=True,
        )



    def fit(self, x, durations, events, **kwargs):
        # Infer n_out from events
        num_events = int(events.max())
        n_out = num_events + 1
        
        if n_out != self.net.n_out:
            # Re-initialize head with correct dimension
            self.net.n_out = n_out
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, n_out).to(self.device)
            print(f"  [TabICL/Expanded] Adjusting head for {n_out} classes (competing risks).", flush=True)

        return super().fit(x, durations, events, **kwargs)

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        x_bin_pt = torch.from_numpy(x_bin).float().to(self.device)
        return self.net.predict_proba(x_bin_pt)
