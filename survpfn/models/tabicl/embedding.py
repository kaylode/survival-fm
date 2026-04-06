"""
survpfn.models.tabicl.embedding — Frozen embedding extraction from TabICL.

TabICL is a 3-stage tabular in-context learning model:
  1. ColEmbedding  — distribution-aware per-column embeddings
  2. RowInteraction — cross-feature interaction via CLS tokens per row
  3. ICLearning     — dataset-wise in-context learning transformer

The embedding we extract is the output of RowInteraction / input of ICLearning,
specifically for the test rows.  This captures the per-row representation
*before* in-context label conditioning, giving a representation that reflects
the feature structure independently of the classification task.

Shape: (B=1, T, icl_dim) where icl_dim = embed_dim * row_num_cls (typically 128*4=512)
After hook: test_emb = hook[0, train_size:, :] → (n_test, icl_dim)

Alternatively, we hook *after* ICLearning to capture the post-ICL representation,
which has seen the training labels.  Both are provided; post-ICL is the default
as it produces richer survival-relevant features.

Usage
-----
    from survpfn.models.tabicl.embedding import get_tabicl_embeddings

    train_emb, test_emb = get_tabicl_embeddings(
        X_train, y_train_binary, X_test,
        device="cuda",
    )

Dependencies
------------
All TabICL source is vendored locally under ``survpfn/models/tabicl/tabicl/``.
No external ``ehrdpt`` package is required.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# TabICL embedding extractor
# ---------------------------------------------------------------------------

class TabICLEmbeddingExtractor:
    """Extract frozen embeddings from a TabICL classifier checkpoint.

    TabICL downloads its pretrained weights from HuggingFace automatically
    (repo ``jingang/TabICL-clf``).  Set ``model_path`` to a local file to
    avoid re-downloading.

    Parameters
    ----------
    device : str
    model_path : str | None    — local checkpoint; None = auto-download
    checkpoint_version : str   — which HuggingFace checkpoint to download
    context_size : int         — max training samples to use as ICL context
    hook_point : str           — ``"pre_icl"`` (RowInteraction output) or
                                 ``"post_icl"`` (ICLearning output, default)
    """

    def __init__(
        self,
        device: str = "cpu",
        model_path: Optional[str] = None,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
    ) -> None:
        from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier

        self.device = device

        # Build a 1-estimator classifier (no ensembling) to expose the raw model
        self._clf = TabICLClassifier(
            n_estimators=1,
            norm_methods="none",
            feat_shuffle_method="none",
            use_amp=True,
            allow_auto_download=True,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            device=device,
            random_state=42,
            verbose=False,
        )

        # Force model load by calling a dummy fit+predict on tiny fake data
        # (TabICLClassifier is lazy — it only loads the checkpoint on first predict)
        self._warm_up()

        # Now access the underlying TabICL nn.Module
        self._tabicl_model: nn.Module = self._clf.model_  # TabICL instance

        # Embedding dim
        embed_dim     = self._tabicl_model.col_embedder.embed_dim
        row_num_cls   = self._tabicl_model.row_interactor.num_cls
        self.emb_dim  = embed_dim * row_num_cls

    def _warm_up(self) -> None:
        """Trigger checkpoint load on minimal dummy data."""
        import numpy as np
        X_dummy = np.random.randn(10, 4).astype(np.float32)
        y_dummy = np.array([0, 1] * 5)
        self._clf.fit(X_dummy, y_dummy)
        self._clf.predict_proba(X_dummy[:2])

    @torch.no_grad()
    def _embed_batch(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """Single TabICL forward pass for one context + query batch.

        Parameters
        ----------
        X_ctx   : (ctx_size, n_features)
        y_ctx   : (ctx_size,)
        X_query : (n_query, n_features)

        Returns
        -------
        emb : (n_query, emb_dim)
        """
        n_query  = X_query.shape[0]

        X_all = np.concatenate([X_ctx, X_query], axis=0)  # (ctx+query, features)

        # TabICL expects (B, T, H) where B=1
        x_t = torch.from_numpy(X_all).float().unsqueeze(0).to(self.device)   # (1, T, H)
        y_t = torch.from_numpy(y_ctx).long().unsqueeze(0).to(self.device)    # (1, ctx_size)

        # Ensure model is in eval mode and on the correct device
        self._tabicl_model.eval()
        self._tabicl_model.to(self.device)

        # Forward pass
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
            enabled=True
        ):
            _, embs = self._tabicl_model(x_t, y_train=y_t, return_embeddings=True)

        return embs.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tabicl_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
    model_path: Optional[str] = None,
    checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen TabICL embeddings for train and test sets.

    Parameters
    ----------
    X_train, X_test  : raw (unscaled) feature arrays
    y_train          : binary event labels (0/1)
    device           : torch device
    model_path       : local checkpoint path; None = auto-download from HuggingFace
    context_size     : max training samples used as ICL context (default 1000)
    hook_point       : ``"post_icl"`` (default) or ``"pre_icl"``

    Returns
    -------
    train_emb : (n_train, emb_dim)
    test_emb  : (n_test,  emb_dim)
    """
    extractor = TabICLEmbeddingExtractor(
        device=device,
        model_path=model_path,
        checkpoint_version=checkpoint_version,
    )

    emb = extractor._embed_batch(X_train, y_train, X_test)
    train_emb = emb[:, :X_train.shape[0]].squeeze(0)
    test_emb  = emb[:, X_train.shape[0]:].squeeze(0)
    return train_emb, test_emb
