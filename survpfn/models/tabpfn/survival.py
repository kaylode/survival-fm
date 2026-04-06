"""
survpfn.models.tabpfn.cox — TabPFN-aware and embedding-based survival models.

Merged from: aware_cox.py + embedding_cox.py

* train_tabpfn_embedding_cox — Thin wrapper over train_fm_embedding_surv (head_type="cox")
* train_tabpfn_embedding_cox — Full wrapper supporting all four head types
"""

import numpy as np
import torchtuples as tt

from typing import Union, List, Optional
import pandas as pd

from survpfn.models.shared.heads import (
    train_fm_embedding_surv,
)


def train_tabpfn_embedding_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    head_type: str = "cox",
    tune: bool = False,
    n_trials: int = 10,
    save_dir: str = "results",
    study_id: Optional[str] = None,
    num_durations: int = 100,
    task_type: str = "sr",
    cr_loss_type: str = "deephit",
    **kwargs
) -> tuple:
    """Frozen TabPFN embedding + any survival head.

    Parameters
    ----------
    head_type : one of "cox", "deephit", "pchazard", "mtlr"

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    from survpfn.models.tabpfn.embedding import get_tabpfn_embeddings

    def _embedding_fn(X_train: np.ndarray, y_bin: np.ndarray,
                      X_test: np.ndarray) -> tuple:
        # TabPFN's extractor also accepts test labels (used as a dummy class
        # for the in-context classifier); pass zeros for the test set.
        y_test_zeros = np.zeros(len(X_test), dtype=np.float32)
        return get_tabpfn_embeddings(
            pd.DataFrame(X_train),
            pd.Series(y_bin.astype(int)),
            pd.DataFrame(X_test),
            pd.Series(y_test_zeros.astype(int)),
        )

    return train_fm_embedding_surv(
        df_train, df_test, duration_col, event_col,
        embedding_fn=_embedding_fn,
        head_type=head_type,
        tune=tune,
        n_trials=n_trials,
        save_dir=save_dir,
        study_id=study_id,
        fm_name="tabpfn",
        num_durations=num_durations,
        task_type=task_type,
        cr_loss_type=cr_loss_type,
        **kwargs
    )