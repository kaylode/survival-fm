"""
survpfn.models.tabicl.model — TabICL backbone + survival head (jointly-trained / frozen).

Classes / functions
-------------------
* TabICLSurvModel  — PyTorch nn.Module: TabICL backbone + survival head MLP.
                     TabICL is kept in eval() mode (its @no_grad free ICL
                     forward) while the survival head trains freely.
                     With freeze_tabicl=False gradients also flow through the
                     TabICL transformer (Approach C); with freeze_tabicl=True
                     only the head trains (Approach A, faster).
* TabICLSurvPH     — High-level wrapper with fit / predict_survival_df API
                     compatible with the benchmark runner.

Context management
------------------
During training each mini-batch draws a random K ≤ context_size training
samples *excluding* the current batch as the ICL context.  After training a
fixed random context subset is stored so that pycox can call ``net(x_query)``
without an explicit context argument.

TabICL API recap
----------------
  tabicl_model(X, y_train, return_embeddings=True)
    X       : (1, K+B, n_features)   — ctx rows first, then query rows
    y_train : (1, K)                 — integer class labels (0 or 1)
    returns : (logits, emb)
    logits  : (1, B, max_classes)
    emb     : (1, K+B, icl_dim)     — icl_dim = embed_dim * row_num_cls (≈ 512)
    query embeddings: emb[0, K:, :] → (B, icl_dim)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt


_DEFAULT_CONTEXT_SIZE = 256   # TabICL handles larger contexts than TabDPT


# ---------------------------------------------------------------------------
# TabICL Survival Model
# ---------------------------------------------------------------------------

class TabICLSurvModel(nn.Module):
    """TabICL backbone (eval-locked) + survival head MLP.

    Parameters
    ----------
    tabicl_model    : pretrained ``TabICL`` nn.Module (loaded via TabICLClassifier).
    n_out           : 1 for Cox; ``num_durations`` for discrete heads.
    head_num_nodes  : hidden layer widths of the survival head MLP.
    dropout         : dropout between head layers.
    freeze_tabicl   : if True, only the survival head trains (Approach A).
                      if False, gradients also flow through TabICL (Approach C).
    """

    def __init__(
        self,
        tabicl_model: nn.Module,
        n_out: int,
        head_num_nodes: List[int] = [256, 128],
        dropout: float = 0.2,
        freeze_tabicl: bool = True,
    ):
        super().__init__()
        self.tabicl = tabicl_model

        # Freeze backbone if requested
        for param in self.tabicl.parameters():
            param.requires_grad = not freeze_tabicl

        # Embedding dimension from model architecture
        self.icl_dim: int = (
            tabicl_model.embed_dim * tabicl_model.row_num_cls
        )  # default: 128 * 4 = 512

        # Survival head MLP: icl_dim → hidden_nodes → n_out
        nodes = [self.icl_dim] + list(head_num_nodes)
        layers: list[nn.Module] = []
        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(nodes[-1], n_out, bias=False))
        self.survival_head = nn.Sequential(*layers)

        # Stored context — set once after training for pycox inference calls
        self._ctx_x: Optional[torch.Tensor] = None   # (K, n_features)
        self._ctx_y: Optional[torch.Tensor] = None   # (K,) int64

    # ------------------------------------------------------------------
    # keep TabICL permanently in eval mode (always use _inference_forward)
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "TabICLSurvModel":
        """Override train() so TabICL backbone stays in eval mode at all times.

        The survival head follows the normal train/eval cycle so that BatchNorm
        statistics are updated correctly during training.
        """
        super().train(mode)
        self.tabicl.eval()   # always eval → _inference_forward (returns emb)
        return self

    # ------------------------------------------------------------------
    # context helpers
    # ------------------------------------------------------------------

    def set_context(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
    ) -> None:
        """Store context for use during prediction (after training).

        Args:
            x_context: (K, n_features) float tensor.
            y_context: (K,) int64 tensor — binary event label (0 / 1).
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
    ) -> torch.Tensor:
        """Forward pass through TabICL backbone → survival head.

        Args:
            x_query:   (B, n_features) — query samples (current batch).
            x_context: (K, n_features) — ICL context features.
                       Uses stored context when None.
            y_context: (K,) int64 — context binary event labels.
                       Uses stored context when None.

        Returns:
            (B,) for Cox; (B, num_durations) for discrete heads.
        """
        device = next(self.survival_head.parameters()).device

        if x_context is None:
            x_context = self._ctx_x
            y_context = self._ctx_y
        if x_context is None:
            raise RuntimeError(
                "No context supplied and none stored. "
                "Call set_context() before prediction, or supply context explicitly."
            )

        # Ensure all inputs are on the correct device
        x_query = x_query.to(device)
        x_context = x_context.to(device)
        y_context = y_context.to(device)

        K = x_context.shape[0]
        B = x_query.shape[0]

        # Concatenate context + query: (K+B, n_features)
        X_all = torch.cat([x_context, x_query], dim=0)

        # TabICL expects (batch_of_tables=1, T, n_features) and (1, K)
        X_t = X_all.unsqueeze(0)                          # (1, K+B, n_features)
        y_t = y_context.long().unsqueeze(0)               # (1, K)

        # Forward through TabICL backbone (always in eval / inference mode)
        # returns (logits, emb); emb shape: (1, K+B, icl_dim)
        _, emb = self.tabicl(X_t, y_train=y_t, return_embeddings=True)

        if emb is None:
            raise RuntimeError(
                "TabICL did not return embeddings. "
                "Ensure the model checkpoint supports return_embeddings=True."
            )

        # Query row embeddings: (B, icl_dim)
        # Ensure they are on the correct device (backbone may return CPU)
        query_embs = emb[0, K:, :].to(device)

        out = self.survival_head(query_embs)   # (B, n_out)
        if out.size(-1) == 1:
            out = out.squeeze(-1)              # (B,) for Cox
        return out


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class TabICLSurvPH:
    """TabICL backbone + survival head — benchmark-compatible wrapper.

    Parameters
    ----------
    head_type       : "cox" | "deephit" | "pchazard" | "mtlr"
    context_size    : max ICL context samples per training step.
    freeze_tabicl   : if True, only survival head trained (Approach A).
                      if False, full end-to-end training (Approach C).
    model_path      : local checkpoint path; None = auto-download from HuggingFace.
    checkpoint_version : TabICL HuggingFace checkpoint name.
    """

    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 10,
        head_num_nodes: List[int] = [256, 128],
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        freeze_tabicl: bool = True,
        device: str = "cuda:0",
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        n_out: Optional[int] = None,
        model_path: Optional[str] = None,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
    ):
        self.head_type     = head_type.lower()
        self.device        = device
        self.context_size  = context_size
        self.num_durations = num_durations

        # ── Load TabICL backbone ─────────────────────────────────────────
        from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier

        clf = TabICLClassifier(
            n_estimators=1,
            norm_methods="none",
            feat_shuffle_method="none",
            use_amp=False,           # disable AMP for stable gradient flow
            allow_auto_download=True,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            device=device,
            random_state=42,
            verbose=False,
        )
        # Trigger checkpoint load via warm-up on tiny dummy data
        _X_d = np.random.randn(10, 4).astype(np.float32)
        _y_d = np.array([0, 1] * 5)
        try:
            clf.fit(_X_d, _y_d)
            clf.predict_proba(_X_d[:2])
        except Exception:
            pass

        tabicl_model: nn.Module = clf.model_          # raw TabICL nn.Module
        tabicl_model.eval()                            # start in eval mode

        # ── Build survival net ────────────────────────────────────────────
        if n_out is None:
            n_out = 1 if self.head_type == "cox" else num_durations

        self.net = TabICLSurvModel(
            tabicl_model=tabicl_model,
            n_out=n_out,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            freeze_tabicl=freeze_tabicl,
        ).to(device)

        # ── pycox model + optimizer ───────────────────────────────────────
        from pycox.models import CoxPH, DeepHitSingle, PCHazard, MTLR

        if self.head_type == "cox":
            self.model     = CoxPH(self.net, tt.optim.Adam(lr=learning_rate))
            self.labtrans  = None

        elif self.head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            self.labtrans = LabTransDiscreteTime(num_durations, scheme="quantiles")
            self.model    = DeepHitSingle(
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
    ) -> "TabICLSurvPH":
        """Train backbone + head end-to-end.

        Parameters
        ----------
        x         : (N, n_features) — already standardised float array.
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
        else:   # mtlr
            criterion_surv = NLLMTLRLoss()

        optimizer = self.model.optimizer
        N = len(x)

        # ── Tensors ─────────────────────────────────────────────────────
        x_pt = torch.from_numpy(x.copy()).float().to(self.device)

        # Label transforms for discrete heads
        if self.labtrans is not None:
            targets = self.labtrans.fit_transform(durations, events)
            dur_pt  = torch.from_numpy(targets[0]).to(self.device)
            ev_pt   = torch.from_numpy(targets[1]).to(self.device)
            frac_pt = (
                torch.from_numpy(targets[2]).float().to(self.device)
                if len(targets) > 2 else None
            )
        else:
            dur_pt  = torch.from_numpy(durations.copy()).float().to(self.device)
            ev_pt   = torch.from_numpy(events.copy()).float().to(self.device)
            frac_pt = None

        # Binary event indicator as ICL context labels (0/1 int)
        y_bin_pt = torch.from_numpy((events > 0).astype(np.int64)).to(self.device)

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(epochs):
            self.net.train()   # survival head → train; TabICL stays eval (via override)

            indices   = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, N, batch_size):
                batch_idx = indices[i : i + batch_size]
                bx   = x_pt[batch_idx]
                bdur = dur_pt[batch_idx]
                bev  = ev_pt[batch_idx]
                bfrac = frac_pt[batch_idx] if frac_pt is not None else None

                # ── Random context: K samples not in current batch ────────
                context_mask = torch.ones(N, dtype=torch.bool, device=self.device)
                context_mask[batch_idx] = False
                pool  = context_mask.nonzero(as_tuple=True)[0]
                n_ctx = min(self.context_size, len(pool))
                perm  = torch.randperm(len(pool), device=self.device)
                ctx_idx = pool[perm[:n_ctx]]

                x_ctx = x_pt[ctx_idx]         # (K, n_features)
                y_ctx = y_bin_pt[ctx_idx]      # (K,) int64

                # ── Forward ───────────────────────────────────────────────
                optimizer.zero_grad()
                head_out = self.net(bx, x_context=x_ctx, y_context=y_ctx)

                # ── Loss ──────────────────────────────────────────────────
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
                else:   # mtlr
                    loss = criterion_surv(head_out, bdur, bev)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            if verbose and epoch % 10 == 0:
                avg = epoch_loss / max(1, n_batches)
                print(
                    f"[TabICLSurv/{self.head_type}] "
                    f"epoch {epoch:3d}/{epochs}  loss={avg:.4f}"
                )

        # ── Store fixed context for pycox prediction calls ────────────────
        ctx_size = min(self.context_size, N)
        ctx_perm = torch.randperm(N)[:ctx_size]
        self.net.set_context(
            x_context=x_pt[ctx_perm].detach(),
            y_context=y_bin_pt[ctx_perm].detach(),
        )

        # ── Baseline hazards (Cox only) ────────────────────────────────────
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
        """Return survival probability DataFrame (rows=times, cols=subjects)."""
        self.net.eval()
        x_pt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            if self.head_type == "cox":
                return self.model.predict_surv_df(x_pt)
            if hasattr(self.model, "interpolate"):
                return self.model.interpolate(10).predict_surv_df(x_pt)
            return self.model.predict_surv_df(x_pt)
