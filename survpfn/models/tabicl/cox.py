"""
survpfn.models.tabicl.cox — Frozen TabICL embedding + survival head.

The pipeline is:
  1. Extract frozen TabICL embeddings for train and test sets.
  2. Fit a survival head (with optional Optuna tuning) on the embeddings.
  3. Predict survival on the test embeddings.

All survival-head logic lives in survpfn.models.heads; this module is a
thin configuration wrapper that resolves TabICL-specific settings and
delegates to train_fm_embedding_surv.

Supported head types
--------------------
  cox, deephit, pchazard, mtlr

Environment variables
---------------------
TABICL_DEVICE       : torch device (default "cpu")
TABICL_CONTEXT_SIZE : int, max training context (default 1000)
TABICL_CHECKPOINT   : local checkpoint path (default: auto-download from HuggingFace)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from survpfn.models.tabicl.embedding import get_tabicl_embeddings
from survpfn.models.heads import train_fm_embedding_surv


def train_tabicl_embedding_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    head_type: str = "cox",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 20,
    save_dir: str = "results",
    study_id: Optional[str] = None,
    device: Optional[str] = None,
    context_size: Optional[int] = None,
    model_path: Optional[str] = None,
    hook_point: str = "post_icl",
) -> tuple:
    """Frozen TabICL embedding + any survival head.

    Parameters
    ----------
    head_type     : survival head — cox | deephit | pchazard | mtlr
    num_durations : discrete time bins (ignored for Cox)
    hook_point    : TabICL layer to hook — "post_icl" (default) or "row_interactor"

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    dev  = device       or os.environ.get("TABICL_DEVICE", "cpu")
    ctx  = context_size or int(os.environ.get("TABICL_CONTEXT_SIZE", "1000"))
    ckpt = model_path   or os.environ.get("TABICL_CHECKPOINT", None)

    emb_kwargs = dict(
        device=dev,
        context_size=ctx,
        model_path=ckpt,
        hook_point=hook_point,
    )

    return train_fm_embedding_surv(
        df_train, df_test, duration_col, event_col,
        embedding_fn=get_tabicl_embeddings,
        emb_kwargs=emb_kwargs,
        head_type=head_type,
        num_durations=num_durations,
        tune=tune,
        n_trials=n_trials,
        save_dir=save_dir,
        study_id=study_id,
        fm_name="tabicl",
    )


# ---------------------------------------------------------------------------
# Convenience aliases — one function per head type
# ---------------------------------------------------------------------------

def train_tabicl_embedding_cox(
    df_train, df_test, duration_col, event_col,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    device=None, context_size=None, model_path=None, hook_point="post_icl",
) -> tuple:
    """Backward-compatible alias for train_tabicl_embedding_surv(head_type='cox')."""
    return train_tabicl_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="cox",
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        device=device, context_size=context_size,
        model_path=model_path, hook_point=hook_point,
    )


def train_tabicl_embedding_deephit(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    device=None, context_size=None, model_path=None, hook_point="post_icl",
) -> tuple:
    return train_tabicl_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="deephit", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        device=device, context_size=context_size,
        model_path=model_path, hook_point=hook_point,
    )


def train_tabicl_embedding_pchazard(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    device=None, context_size=None, model_path=None, hook_point="post_icl",
) -> tuple:
    return train_tabicl_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="pchazard", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        device=device, context_size=context_size,
        model_path=model_path, hook_point=hook_point,
    )


def train_tabicl_embedding_mtlr(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    device=None, context_size=None, model_path=None, hook_point="post_icl",
) -> tuple:
    return train_tabicl_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="mtlr", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        device=device, context_size=context_size,
        model_path=model_path, hook_point=hook_point,
    )
