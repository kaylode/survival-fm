"""
survpfn.models.tabdpt.embedding — Frozen embedding extraction from TabDPT.

TabDPT (aka EHRPFNModel / ehrdpt.models.tabdpt.model.TabDPTModel) is a
transformer Prior-Fitted Network adapted for EHR/clinical tabular data.

Architecture recap
------------------
forward(x, y, eval_pos):
  x : (eval_pos + n_test, B, num_features)   — context + test tokens
  y : (eval_pos, B)                           — context labels only
  → clip + normalise → linear encoder → RMS norm
  → y_encoder fused into context tokens
  → N transformer layers  →  src : (eval_pos + n_test, B, ninp)
  → self.head(src)        →  logits (returned from eval_pos onward)

The penultimate embedding for test samples is `src[eval_pos:, 0, :]`,
captured via a forward hook on `model.head`.

Usage
-----
    from survpfn.models.tabdpt.embedding import get_tabdpt_embeddings

    train_emb, test_emb = get_tabdpt_embeddings(
        X_train, y_train_binary, X_test,
        checkpoint_path="/path/to/tabdpt.ckpt",
        context_size=128,
        device="cuda",
    )

Checkpoint format (from ehrdpt training):
    {
        "model": <state_dict>,
        "cfg":   <OmegaConf DictConfig with model/training/env sub-keys>
    }

Environment variable
--------------------
Set TABDPT_CHECKPOINT to the checkpoint path to avoid passing it explicitly.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Package resolution — try installed package, fall back to source tree
# ---------------------------------------------------------------------------

# From here, we only use the local tabdpt copy



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_features(x: torch.Tensor, max_features: int) -> torch.Tensor:
    """Zero-pad the feature dimension to max_features."""
    n, f = x.shape
    if f < max_features:
        pad = torch.zeros(n, max_features - f, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    return x[:, :max_features]


def _knn_indices(X_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    """Brute-force cosine k-NN (no FAISS dependency).

    Returns
    -------
    indices : (n_query, k) int array — indices into X_train
    """
    X_tr = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    X_q  = X_query / (np.linalg.norm(X_query,  axis=1, keepdims=True) + 1e-8)
    sims = X_q @ X_tr.T                         # (n_query, n_train)
    return np.argsort(-sims, axis=1)[:, :k]     # descending similarity


# ---------------------------------------------------------------------------
# Core embedding extractor
# ---------------------------------------------------------------------------

class TabDPTEmbeddingExtractor:
    """Extract frozen penultimate-layer embeddings from a TabDPT checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved ``{"model": state_dict, "cfg": DictConfig}`` checkpoint.
    device : str
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    context_size : int
        Number of training samples to use as in-context prompt per test sample.
        When ``n_train <= context_size``, the full training set is used.
        Otherwise, k-NN retrieval selects the closest ``context_size`` neighbours.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        context_size: int = 128,
    ) -> None:
        from survpfn.models.tabdpt.tabdpt.model import TabDPTModel


        self.device = device
        self.context_size = context_size

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model: nn.Module = TabDPTModel.load(
            model_state=ckpt["model"], config=ckpt["cfg"]
        )
        self.model.eval()
        self.model.to(device)

        self.max_features: int = self.model.num_features
        self.ninp: int = self.model.ninp

        # Forward hook: capture transformer output (head input) for ALL positions
        self._hook_output: Optional[torch.Tensor] = None

        def _head_hook(module, inp, out):  # noqa: ANN001
            # inp[0] shape: (eval_pos + n_test, B, ninp)
            self._hook_output = inp[0].detach()

        self.model.head.register_forward_hook(_head_hook)

        # Preprocessing: standardise features (fit on training split)
        self._scaler = StandardScaler()

    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabDPTEmbeddingExtractor":
        """Store and preprocess the training context.

        Parameters
        ----------
        X_train : (n_train, n_features)
        y_train : (n_train,)  binary event labels (0/1)
        """
        self._X_train = self._scaler.fit_transform(X_train.astype(np.float32))
        self._y_train = y_train.astype(np.float32)
        return self

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed_batch(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """Run one TabDPT forward pass and return test embeddings.

        Parameters
        ----------
        X_ctx   : (ctx_size, n_features) — context features (already scaled)
        y_ctx   : (ctx_size,)            — context labels
        X_query : (n_query, n_features)  — query features (already scaled)

        Returns
        -------
        emb : (n_query, ninp)
        """
        ctx_size = X_ctx.shape[0]
        n_query  = X_query.shape[0]

        # Stack [context | query] → (ctx_size + n_query, n_features)
        X_all = np.concatenate([X_ctx, X_query], axis=0)

        # Convert to tensors; TabDPT expects (seq_len, batch=1, features)
        x = torch.from_numpy(X_all).float().to(self.device)
        x = _pad_features(x, self.max_features)
        x = x.unsqueeze(1)   # (seq_len, 1, max_features)

        y = torch.from_numpy(y_ctx).float().to(self.device)
        y = y.unsqueeze(1)   # (ctx_size, 1)

        with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu",
                            dtype=torch.bfloat16, enabled="cuda" in self.device):
            _ = self.model(x, y, eval_pos=ctx_size)

        # self._hook_output shape: (ctx_size + n_query, 1, ninp)
        hook = self._hook_output                 # (total, 1, ninp)
        test_emb = hook[ctx_size:, 0, :].float() # (n_query, ninp)
        return test_emb.cpu().numpy()

    # ------------------------------------------------------------------

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Extract embeddings for X_test using stored training context.

        Parameters
        ----------
        X_test : (n_test, n_features)

        Returns
        -------
        emb : (n_test, ninp)
        """
        if not hasattr(self, "_X_train"):
            raise RuntimeError("Call fit() before transform().")

        X_test_scaled = self._scaler.transform(X_test.astype(np.float32))
        n_train = self._X_train.shape[0]
        n_test  = X_test_scaled.shape[0]

        if n_train <= self.context_size:
            # Full training set as context — single forward pass
            return self._embed_batch(self._X_train, self._y_train, X_test_scaled)

        # k-NN retrieval: for each test sample, pick context_size nearest neighbours
        knn_idx = _knn_indices(self._X_train, X_test_scaled, k=self.context_size)

        embeddings = np.zeros((n_test, self.ninp), dtype=np.float32)
        # Process in batches of inf_batch_size to avoid OOM
        inf_batch = 64
        for start in range(0, n_test, inf_batch):
            end = min(n_test, start + inf_batch)
            batch_q = X_test_scaled[start:end]   # (batch, features)
            batch_idx = knn_idx[start:end]        # (batch, context_size)

            # For each query in the batch, build a (context_size, features) context.
            # We flatten to one joint forward pass: contexts are row-concatenated
            # and each test query is evaluated against its own context.
            # Because TabDPT processes seq_len tokens jointly, we run one query at a time
            # when contexts differ per sample.
            for q in range(end - start):
                ctx_idx = batch_idx[q]            # (context_size,)
                X_ctx = self._X_train[ctx_idx]
                y_ctx = self._y_train[ctx_idx]
                X_q   = batch_q[q:q+1]            # (1, features)
                embeddings[start + q] = self._embed_batch(X_ctx, y_ctx, X_q)[0]

        return embeddings

    # ------------------------------------------------------------------

    def fit_transform(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """Fit on training set and extract training embeddings (leave-one-out via k-NN).

        For the frozen embedding Cox model, training embeddings are used to fit
        the Cox head and compute baseline hazards.  We use the same k-NN retrieval
        but exclude the query sample from its own context.
        """
        self.fit(X_train, y_train)
        n = X_train.shape[0]
        X_scaled = self._X_train

        if n <= self.context_size + 1:
            # Small dataset: leave-one-out context (exclude self)
            embeddings = np.zeros((n, self.ninp), dtype=np.float32)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                X_ctx = X_scaled[mask]
                y_ctx = self._y_train[mask]
                X_q   = X_scaled[i:i+1]
                embeddings[i] = self._embed_batch(X_ctx, y_ctx, X_q)[0]
            return embeddings

        # Large dataset: k-NN excluding self
        knn_idx = _knn_indices(X_scaled, X_scaled, k=self.context_size + 1)
        embeddings = np.zeros((n, self.ninp), dtype=np.float32)
        for i in range(n):
            # First neighbour is always self — skip it
            ctx_idx = knn_idx[i, 1:self.context_size + 1]
            X_ctx = X_scaled[ctx_idx]
            y_ctx = self._y_train[ctx_idx]
            X_q   = X_scaled[i:i+1]
            embeddings[i] = self._embed_batch(X_ctx, y_ctx, X_q)[0]
        return embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tabdpt_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    checkpoint_path: Optional[str] = None,
    context_size: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen TabDPT embeddings for train and test sets.

    Parameters
    ----------
    X_train, X_test : np.ndarray  — raw (unscaled) features
    y_train         : np.ndarray  — binary event labels (0=censored, 1=event)
    checkpoint_path : str | None  — path to TabDPT checkpoint;
                                    falls back to TABDPT_CHECKPOINT env var
    context_size    : int         — k-NN context window size (default 128)
    device          : str         — torch device

    Returns
    -------
    train_emb : (n_train, ninp)
    test_emb  : (n_test,  ninp)
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("TABDPT_CHECKPOINT", "")
    if not checkpoint_path:
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/models_diff/tabdpt1_1.pth")
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            # Absolute path fallback from user
            fallback_abs = "/home/mpham/workspace/source/ehrfm/survpfn/survpfn/models/models_diff/tabdpt1_1.pth"
            if os.path.exists(fallback_abs):
                checkpoint_path = fallback_abs

    if not checkpoint_path:
        raise ValueError(
            "No TabDPT checkpoint found.  Pass checkpoint_path= or set "
            "the TABDPT_CHECKPOINT environment variable."
        )

    extractor = TabDPTEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        context_size=context_size,
    )

    train_emb = extractor.fit_transform(X_train, y_train)
    test_emb  = extractor.transform(X_test)
    return train_emb, test_emb
