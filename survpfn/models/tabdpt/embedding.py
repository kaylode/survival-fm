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
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        context_size: int = 128,
        use_retrieval: bool = True,
    ) -> None:
        from survpfn.models.tabdpt.tabdpt.tabdpt import TabDPTClassifier

        self.device = device
        self.context_size = context_size
        self.use_retrieval = use_retrieval
        self._clf = TabDPTClassifier(
            path=checkpoint_path,
            device=device,
        )
        self.ninp = self._clf.model.ninp

    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabDPTEmbeddingExtractor":
        """Fit the underlying classifier with training context."""
        self._clf.fit(X_train, y_train)
        return self

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed_all(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings for X_train (leave-one-out) and X_test.

        Returns
        -------
        train_emb : (n_train, ninp)
        test_emb  : (n_test, ninp)
        """
        # 1. Test set embeddings (standard retrieval)
        test_emb = self._clf.predict_embeddings(
            X_test,
            context_size=self.context_size,
            use_retrieval=self.use_retrieval,
            return_full=False
        )

        # 2. Train set embeddings (leave-one-out retrieval if using retrieval)
        n_train = self._clf.X_train.shape[0]
        if not self.use_retrieval:
            # Full context pass (same context for all)
            full_res = self._clf.predict_embeddings(
                self._clf.X_train[:1], # Just to trigger context preparation
                context_size=self.context_size,
                use_retrieval=False,
                return_full=True
            )
            # full_res is [n_ctx + 1, 1, ninp] => train embeddings are first n_ctx
            train_emb = full_res[:n_train, 0, :]
        else:
            # For each train point, we need to pick a context that excludes self.
            # We bypass the classifier's batch_retrieval_data to do leave-one-out.
            from .tabdpt.retrieval import FAISS
            faiss_knn = FAISS(self._clf.X_train)
            # Get k+1 nearest neighbours
            indices_knn_all = faiss_knn.get_knn_indices(self._clf.X_train, k=self.context_size + 1)
            
            train_embeddings = []
            # Batch these to avoid too many small forward passes? 
            # Actually, each sample has a *unique* context now, so we run B=1.
            for i in range(n_train):
                # First neighbour is usually self — skip it
                indices = indices_knn_all[i]
                ctx_idx = indices[indices != i][:self.context_size]
                
                # Manual forward pass for this LOO sample
                X_ctx = self._clf.X_train[ctx_idx]
                y_ctx = self._clf.y_train[ctx_idx]
                X_q   = self._clf.X_train[i:i+1]
                
                emb_q = self._clf.predict_embeddings(
                    X_q, context_size=len(ctx_idx), use_retrieval=False, return_full=False
                ) # This is a bit slow but correct
                train_embeddings.append(emb_q)
            
            train_emb = np.concatenate(train_embeddings, axis=0)

        return train_emb, test_emb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tabdpt_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    context_size: int = 128,
    device: str = "cpu",
    use_retrieval: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen TabDPT embeddings for train and test sets.

    Parameters
    ----------
    X_train, X_test : np.ndarray  — raw (unscaled) features
    y_train         : np.ndarray  — binary event labels (0=censored, 1=event)
    checkpoint_path : str | None  — path to TabDPT checkpoint;
                                    falls back to TABDPT_CHECKPOINT env var
    device          : str         — torch device

    Returns
    -------
    train_emb : (n_train, ninp)
    test_emb  : (n_test,  ninp)
    """
    from survpfn.models.tabdpt.tabdpt.model import TABDPT_CHECKPOINT_PATH
    checkpoint_path = TABDPT_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        pass
    else:
        raise FileNotFoundError(
            "No TabDPT checkpoint found.  Pass checkpoint_path= or set "
            "the TABDPT_CHECKPOINT environment variable."
        )

    extractor = TabDPTEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        context_size=context_size,
        use_retrieval=use_retrieval,
    )
    extractor.fit(X_train, y_train)

    train_emb, test_emb = extractor._embed_all(X_test)
    return train_emb, test_emb
