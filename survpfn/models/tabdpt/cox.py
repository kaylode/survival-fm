"""
survpfn.models.tabdpt.cox — Frozen TabDPT embedding + survival head.

The pipeline is:
  1. Extract frozen TabDPT embeddings for train and test sets.
  2. Fit a survival head (with optional Optuna tuning) on the embeddings.
  3. Predict survival on the test embeddings.

All survival-head logic lives in survpfn.models.heads; this module is a
thin configuration wrapper that resolves TabDPT-specific settings and
delegates to train_fm_embedding_surv.

Supported head types
--------------------
  cox, deephit, pchazard, mtlr

Environment variables
---------------------
TABDPT_CHECKPOINT   : path to the .ckpt file (required unless passed explicitly)
TABDPT_CONTEXT_SIZE : int, default 128
TABDPT_DEVICE       : torch device string, default "cpu"
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from survpfn.models.tabdpt.embedding import get_tabdpt_embeddings
from survpfn.models.heads import train_fm_embedding_surv


def train_tabdpt_embedding_surv(
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
    checkpoint_path: Optional[str] = None,
    context_size: Optional[int] = None,
    device: Optional[str] = None,
) -> tuple:
    """Frozen TabDPT embedding + any survival head.

    Parameters
    ----------
    head_type     : survival head — cox | deephit | pchazard | mtlr
    num_durations : discrete time bins (ignored for Cox)

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    ckpt = checkpoint_path or os.environ.get("TABDPT_CHECKPOINT", "")
    ctx  = context_size    or int(os.environ.get("TABDPT_CONTEXT_SIZE", "128"))
    dev  = device          or os.environ.get("TABDPT_DEVICE", "cpu")

    emb_kwargs = dict(checkpoint_path=ckpt, context_size=ctx, device=dev)

    return train_fm_embedding_surv(
        df_train, df_test, duration_col, event_col,
        embedding_fn=get_tabdpt_embeddings,
        emb_kwargs=emb_kwargs,
        head_type=head_type,
        num_durations=num_durations,
        tune=tune,
        n_trials=n_trials,
        save_dir=save_dir,
        study_id=study_id,
        fm_name="tabdpt",
    )


# ---------------------------------------------------------------------------
# Convenience aliases — one function per head type
# ---------------------------------------------------------------------------

def train_tabdpt_embedding_cox(
    df_train, df_test, duration_col, event_col,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    checkpoint_path=None, context_size=None, device=None,
) -> tuple:
    """Backward-compatible alias for train_tabdpt_embedding_surv(head_type='cox')."""
    return train_tabdpt_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="cox",
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        checkpoint_path=checkpoint_path, context_size=context_size, device=device,
    )


def train_tabdpt_embedding_deephit(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    checkpoint_path=None, context_size=None, device=None,
) -> tuple:
    return train_tabdpt_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="deephit", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        checkpoint_path=checkpoint_path, context_size=context_size, device=device,
    )


def train_tabdpt_embedding_pchazard(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    checkpoint_path=None, context_size=None, device=None,
) -> tuple:
    return train_tabdpt_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="pchazard", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        checkpoint_path=checkpoint_path, context_size=context_size, device=device,
    )


def train_tabdpt_embedding_mtlr(
    df_train, df_test, duration_col, event_col,
    num_durations=100,
    tune=False, n_trials=20, save_dir="results", study_id=None,
    checkpoint_path=None, context_size=None, device=None,
) -> tuple:
    return train_tabdpt_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="mtlr", num_durations=num_durations,
        tune=tune, n_trials=n_trials, save_dir=save_dir, study_id=study_id,
        checkpoint_path=checkpoint_path, context_size=context_size, device=device,
    )
