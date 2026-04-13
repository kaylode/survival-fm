from __future__ import annotations

import warnings
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

    Uses ``drop_first=True`` so that a binary variable (2 levels) produces
    only one indicator column, avoiding the perfect anti-collinearity that
    ``drop_first=False`` would introduce (e.g. ``cvd_0`` + ``cvd_1`` with
    |r|=1.0 everywhere).

    Returns a DataFrame with named columns (no scaling applied).
    Boolean columns are cast to float.
    """
    # Clean extreme numerics (sentinels) before we start encoding
    df = _clean_numerics(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


def _drop_low_prevalence_binary(
    df: pd.DataFrame,
    exclude_cols: list[str],
    min_count: int = 5,
    event_col: str | None = None,
) -> pd.DataFrame:
    """Drop binary (0/1) columns whose minority class has fewer than
    ``min_count`` observations — globally AND within each event stratum.

    The stratum-aware check catches ICD-10 rare chapters (e.g. Pregnancy,
    Congenital anomalies) that have >``min_count`` total occurrences but
    appear in almost no event=1 patients, triggering lifelines
    ConvergenceWarning via complete separation.

    Parameters
    ----------
    df          : full DataFrame (including duration/event cols).
    exclude_cols: column names to skip (duration + event).
    min_count   : minimum count required in BOTH the 0/1 classes globally
                  AND in BOTH event strata when ``event_col`` is given.
    event_col   : name of the binary event column in ``df``.  When provided,
                  stratum-aware filtering is applied in addition to the
                  global minority-class check.

    Returns
    -------
    Filtered DataFrame (copy).
    """
    df = df.copy()
    drop_cols = []

    event_series = None
    if event_col is not None and event_col in df.columns:
        event_series = (df[event_col] > 0).astype(int)

    for col in df.columns:
        if col in exclude_cols:
            continue
        s = df[col]
        uniq = s.dropna().unique()
        if not set(uniq).issubset({0, 1, 0.0, 1.0, True, False}):
            continue

        # ── Global minority-class check ──
        minority = min(int((s == 1).sum()), int((s == 0).sum()))
        if minority < min_count:
            drop_cols.append(col)
            continue

        # ── Stratum-aware check (prevents complete separation) ──
        if event_series is not None:
            ev1 = s[event_series == 1]
            ev0 = s[event_series == 0]
            # Within each stratum, both 0-class and 1-class must appear >= min_count times
            for stratum_s in (ev1, ev0):
                n1 = int((stratum_s == 1).sum())
                n0 = int((stratum_s == 0).sum())
                if min(n1, n0) < min_count:
                    drop_cols.append(col)
                    break

    if drop_cols:
        warnings.warn(
            f"_drop_low_prevalence_binary: dropping {len(drop_cols)} sparse binary "
            f"feature(s) with minority-class count < {min_count} "
            f"(global or per-event-stratum): "
            f"{drop_cols[:10]}{'...' if len(drop_cols) > 10 else ''}",
            UserWarning,
            stacklevel=2,
        )
        df = df.drop(columns=drop_cols)
    return df


def _drop_duplicate_columns(
    df: pd.DataFrame,
    exclude_cols: list[str],
) -> pd.DataFrame:
    """Drop columns that are byte-identical to an earlier column.

    This handles the case where lab aggregates (min/max/mean) are all equal
    because a patient had only a single measurement — making the three
    columns redundant and creating spurious |r|=1.0 collinearity.

    The *first* occurrence of each unique column is kept; all later duplicates
    are dropped.

    Parameters
    ----------
    df          : DataFrame that may contain duplicate columns.
    exclude_cols: duration and event column names to skip.

    Returns
    -------
    Deduplicated DataFrame (copy).
    """
    df = df.copy()
    seen: dict[bytes, str] = {}   # hash → first column name
    drop_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        key = df[col].values.tobytes() if hasattr(df[col].values, "tobytes") else None
        if key is None:
            continue
        if key in seen:
            drop_cols.append(col)
        else:
            seen[key] = col
    if drop_cols:
        warnings.warn(
            f"_drop_duplicate_columns: dropping {len(drop_cols)} byte-identical "
            f"column(s): {drop_cols[:10]}{'...' if len(drop_cols) > 10 else ''}",
            UserWarning,
            stacklevel=2,
        )
        df = df.drop(columns=drop_cols)
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