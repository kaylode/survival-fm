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
        context_size: int = 1000,
        hook_point: str = "post_icl",
    ) -> None:
        from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier

        self.device = device
        self.context_size = context_size
        self.hook_point = hook_point

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

        # Register hook to capture representations
        self._hook_output: Optional[torch.Tensor] = None

        if self.hook_point == "pre_icl":
            # After RowInteraction, before ICLearning: representations
            def _hook(module, inp, out):
                self._hook_output = out.detach()   # (B, T, icl_dim)
            self._tabicl_model.row_interactor.register_forward_hook(_hook)
        else:
            # After ICLearning (default): post-ICL representation
            # ICLearning.forward returns logits; we need the pre-head activations.
            # Hook on the last transformer block inside ICLearning.
            icl = self._tabicl_model.icl_predictor
            last_block = icl.blocks[-1]  # Last ICL transformer block
            def _hook(module, inp, out):
                self._hook_output = out.detach()   # (B, T, icl_dim)
            last_block.register_forward_hook(_hook)

        # Embedding dim
        embed_dim     = self._tabicl_model.col_embedder.embed_dim
        row_num_cls   = self._tabicl_model.row_interactor.num_cls
        self.emb_dim  = embed_dim * row_num_cls

        # Feature preprocessor
        self._scaler = StandardScaler()

    def _warm_up(self) -> None:
        """Trigger checkpoint load on minimal dummy data."""
        import numpy as np
        X_dummy = np.random.randn(10, 4).astype(np.float32)
        y_dummy = np.array([0, 1] * 5)
        try:
            self._clf.fit(X_dummy, y_dummy)
            self._clf.predict_proba(X_dummy[:2])
        except Exception:
            pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabICLEmbeddingExtractor":
        """Store scaled training context.

        Parameters
        ----------
        X_train : (n_train, n_features) raw features
        y_train : (n_train,) binary event labels (0/1)
        """
        self._X_train = self._scaler.fit_transform(X_train.astype(np.float32))
        self._y_train = y_train.astype(np.float32)
        return self

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
        ctx_size = X_ctx.shape[0]
        n_query  = X_query.shape[0]

        X_all = np.concatenate([X_ctx, X_query], axis=0)  # (ctx+query, features)

        # TabICL expects (B, T, H) where B=1
        x_t = torch.from_numpy(X_all).float().unsqueeze(0).to(self.device)   # (1, T, H)
        y_t = torch.from_numpy(y_ctx).long().unsqueeze(0).to(self.device)    # (1, ctx_size)

        # Forward pass through the raw TabICL model
        with torch.autocast(
            device_type="cuda" if "cuda" in self.device else "cpu",
            dtype=torch.float16, enabled="cuda" in self.device
        ):
            _ = self._tabicl_model(x_t, y_train=y_t)

        # self._hook_output: (1, T, emb_dim)
        hook = self._hook_output                         # (1, ctx+query, emb_dim)
        test_emb = hook[0, ctx_size:, :].float()         # (n_query, emb_dim)
        return test_emb.cpu().numpy()

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Extract embeddings for X_test using stored training context.

        Returns
        -------
        emb : (n_test, emb_dim)
        """
        if not hasattr(self, "_X_train"):
            raise RuntimeError("Call fit() before transform().")

        X_test_scaled = self._scaler.transform(X_test.astype(np.float32))
        n_train = self._X_train.shape[0]
        n_test  = X_test_scaled.shape[0]

        if n_train <= self.context_size:
            # All training samples fit in context — single forward pass per test batch
            # TabICL can handle large test sets in one pass (no N² over test)
            return self._embed_batch(self._X_train, self._y_train, X_test_scaled)

        # Subsample training context (random subset, no retrieval for simplicity)
        rng = np.random.default_rng(42)
        ctx_idx = rng.choice(n_train, size=self.context_size, replace=False)
        X_ctx = self._X_train[ctx_idx]
        y_ctx = self._y_train[ctx_idx]
        return self._embed_batch(X_ctx, y_ctx, X_test_scaled)

    def fit_transform(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """Fit and extract training embeddings (leave-out random subset as context)."""
        self.fit(X_train, y_train)
        n = X_train.shape[0]
        X_scaled = self._X_train

        if n <= self.context_size + 1:
            # LOO: for each sample use all others as context
            embeddings = np.zeros((n, self.emb_dim), dtype=np.float32)
            for i in range(n):
                mask = np.ones(n, dtype=bool); mask[i] = False
                embeddings[i] = self._embed_batch(
                    X_scaled[mask], self._y_train[mask], X_scaled[i:i+1]
                )[0]
            return embeddings

        # Large dataset: use a fixed random context excluding each sample
        rng = np.random.default_rng(42)
        embeddings = np.zeros((n, self.emb_dim), dtype=np.float32)
        # Process in chunks to amortise context prep cost
        chunk = 64
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            # Random context from samples NOT in this chunk
            non_chunk = np.concatenate([np.arange(0, start), np.arange(end, n)])
            ctx_idx = rng.choice(non_chunk, size=min(self.context_size, len(non_chunk)), replace=False)
            X_ctx = X_scaled[ctx_idx]
            y_ctx = self._y_train[ctx_idx]
            embeddings[start:end] = self._embed_batch(X_ctx, y_ctx, X_scaled[start:end])
        return embeddings


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
    context_size: int = 1000,
    hook_point: str = "post_icl",
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
        context_size=context_size,
        hook_point=hook_point,
    )
    train_emb = extractor.fit_transform(X_train, y_train)
    test_emb  = extractor.transform(X_test)
    return train_emb, test_emb
