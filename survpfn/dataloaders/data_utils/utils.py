from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove extreme 'sentinel' values (e.g. 10^33) used as missing placeholders."""
    # We apply this to all columns EXCEPT potential duration/event (handled elsewhere)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Values > 1e15 are almost certainly placeholders or corruption
            mask_extreme = df[col].abs() > 1e15
            if mask_extreme.any():
                df.loc[mask_extreme, col] = np.nan
    return df


def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; leave numerics unchanged.

    Returns a DataFrame with named columns (no scaling applied).
    Boolean columns are cast to float.
    """
    # Clean extreme numerics (sentinels) before we start encoding
    df = _clean_numerics(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=False)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    time_col: str | None = None,
    frac_train: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test index arrays for a temporal (prospective) split.

    If *time_col* is provided the rows are sorted by that column first
    (ascending).  Otherwise the existing row order is treated as the
    enrollment/diagnosis order (oldest → newest).

    Parameters
    ----------
    df         : the full dataset DataFrame (before any scaling).
    time_col   : column to sort by (e.g. ``"diagnosis_year"``).
                 Pass ``None`` to preserve the current row order.
    frac_train : fraction of rows assigned to train (default 0.70).

    Returns
    -------
    (train_idx, test_idx) as 1-D integer numpy arrays (row positions, not
    index labels).
    """
    n = len(df)
    if time_col is not None:
        order = np.argsort(df[time_col].values, kind="stable")
    else:
        order = np.arange(n)

    cutoff = int(np.floor(n * frac_train))
    train_idx = order[:cutoff]
    test_idx  = order[cutoff:]
    return train_idx, test_idx