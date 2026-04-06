"""
survpfn.models.tabicl.cox — Frozen TabICL embedding + survival head.

The pipeline is:
  1. Extract frozen TabICL embeddings for train and test sets.
  2. Fit a survival head (with optional Optuna tuning) on the embeddings.
  3. Predict survival on the test embeddings.

All survival-head logic lives in survpfn.models.shared.heads; this module is a
thin configuration wrapper that resolves TabICL-specific settings and
delegates to train_fm_embedding_surv.

Supported head types
--------------------
  cox, deephit, pchazard, mtlr

Environment variables
---------------------
TABICL_DEVICE       : torch device (default "cpu")
TABICL_CHECKPOINT   : local checkpoint path (default: auto-download from HuggingFace)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from survpfn.models.tabicl.embedding import get_tabicl_embeddings
from survpfn.models.shared.heads import train_fm_embedding_surv


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
    model_path: Optional[str] = None,
    task_type: str = "sr",
    cr_loss_type: str = "deephit",
    **kwargs
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
    ckpt = model_path   or os.environ.get("TABICL_CHECKPOINT", None)

    emb_kwargs = dict(
        device=dev,
        model_path=ckpt,
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
        task_type=task_type,
        cr_loss_type=cr_loss_type,
        **kwargs
    )