"""
src/data.py — Data loading, cleaning, imputation, and task-specific splits.

Design decisions vs. the original notebook / survpfn package
-------------------------------------------------------------
1. **No global dropna**: `clean_and_impute()` uses column-stratified imputation
   (median for continuous, mode for binary) instead of dropping all rows with any
   missing value. The original code discarded ~22% of the dataset.

2. **Scaler fitted only on train split**: Each `prepare_*` helper now accepts an
   optional pre-fitted `StandardScaler` so callers (cross-validation loops) can
   pass a scaler fitted exclusively on the training fold, preventing leakage.

3. **Outcome-adjacent columns excluded before scaling**: CABG date, PCI date,
   Non-Fatal AMI date, Ictus date, and related event flags are kept out of the
   feature matrix passed to models. Those columns encode future event timing and
   constitute direct data leakage when used as covariates.

4. **Stratified train/val/test split utility**: `make_splits()` produces
   reproducible, stratified splits suitable for k-fold CV.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that encode future event timing — excluding them prevents leakage.
# These are the raw date columns for competing-risk endpoints.
LEAKY_DATE_COLS: list[str] = [
    "CABG ",           # CABG procedure date (days from baseline)
    "PCI",             # PCI procedure date
    "Non Fatal AMI (Follow-Up)",   # AMI event date
    "Ictus",           # Stroke event date
    "Data of death",   # Date of death (days)
]

# Binary flag columns derived from the leaky dates — also kept out of
# continuous feature scaling but retained as binary covariates where appropriate.
LEAKY_EVENT_FLAGS: list[str] = [
    "CABG _event",
    "PCI_event",
    "Non Fatal AMI (Follow-Up)_event",
    "Ictus_event",
]

# Administrative / quasi-identifier columns never used as model features.
ADMIN_COLS: list[str] = [
    "Collected by",
    "Cause of death",
    "CardiopatiaCongenita",
    "Number",           # patient identifier
    "Data prelievo",    # raw blood-draw date (already converted to reference point)
]

# Columns to drop when building the total-mortality Cox feature matrix.
# These are competing-outcome labels that would constitute label leakage.
MORTALITY_LEAKAGE_COLS: list[str] = [
    "Fatal MI or Sudden death",
    "UnKnown",
    "Accident",
    "Suicide",
    "CVD Death",
]


# ---------------------------------------------------------------------------
# 1.  Loading & merging raw files
# ---------------------------------------------------------------------------

def load_and_merge_data(data_dir: str | Path = "Dataset Sirbu") -> pd.DataFrame:
    """Load and merge the three raw Excel files from the Sirbu dataset.

    The raw dataset is spread across three files:
    - ``DataPrelievo.xlsx``          — blood-draw dates per patient
    - ``Creatinina_AltriEsamiCorretti.xlsx`` — corrected lab values
    - ``OrmoniTiroidei3Aprile2024.xlsx``     — main dataset (thyroid hormones + outcomes)

    Overlapping columns in the corrected file overwrite the main file values.
    New columns from the corrected file are joined in.

    Parameters
    ----------
    data_dir:
        Path to the directory containing the three Excel files.

    Returns
    -------
    pd.DataFrame
        Merged dataframe indexed by patient ``Number``, with blood-draw date
        attached.
    """
    data_dir = Path(data_dir)

    df_date = pd.read_excel(data_dir / "DataPrelievo.xlsx")
    df_update = pd.read_excel(data_dir / "Creatinina_AltriEsamiCorretti.xlsx")
    df_main = pd.read_excel(data_dir / "OrmoniTiroidei3Aprile2024.xlsx")

    # Index by patient number for update/join operations
    df_main = df_main.set_index("Number")
    df_update = df_update.set_index("Number")

    # Overwrite shared columns with corrected values
    cols_to_update = df_update.columns.intersection(df_main.columns)
    df_main.update(df_update[cols_to_update])

    # Append new columns that exist only in the corrected file
    new_cols = df_update.columns.difference(df_main.columns)
    df_main = df_main.join(df_update[new_cols])

    df_main = df_main.reset_index()

    # Attach blood-draw date
    df_main = df_main.merge(
        df_date[["Number", "Data prelievo"]],
        on="Number",
        how="left",
    )

    return df_main


# ---------------------------------------------------------------------------
# 2.  Cleaning & imputation  (replaces aggressive dropna)
# ---------------------------------------------------------------------------

def _infer_binary_cols(df: pd.DataFrame) -> list[str]:
    """Return column names whose non-null unique values are a subset of {0, 1}."""
    return [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]


def clean_and_impute(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw merged dataframe and impute missing values.

    Steps
    -----
    1. Fix known data-entry errors (hard-coded row patches from notebook analysis).
    2. Drop the small number of rows with no patient identifier.
    3. Parse date columns; convert all event dates to *days from blood-draw*.
    4. Create binary event-indicator flags for the four competing-risk endpoints.
    5. Drop administrative columns that carry no predictive value.
    6. **Impute** missing values:
       - Binary / ordinal columns  → mode imputation (0 if mode is ambiguous).
       - Continuous columns        → median imputation.
       This preserves ~22% of rows that the original notebook dropped.

    Parameters
    ----------
    df_raw:
        Output of :func:`load_and_merge_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with imputed values and all date columns converted
        to numeric (days).  Index is reset to RangeIndex.

    Notes
    -----
    Hard-coded row patches (row labels 8143 and 7285) were identified during
    exploratory analysis in the notebook.  Row 8143 had an outlier value of 2
    for a binary ischemia column; row 7285 had a death date recorded but
    ``Total mortality`` == 0.
    """
    df = df_raw.copy()

    # ---- 2a. Known data-entry fixes -----------------------------------------
    # Row 8143: binary ischemia column had value 2; should be 1
    if 8143 in df.index:
        df.loc[8143, "Documented resting \nor exertional ischemia"] = 1

    # Row 7285: death date present but event flag = 0
    if 7285 in df.index:
        df.loc[7285, "Total mortality"] = 1
        df.loc[7285, "UnKnown"] = 1

    # ---- 2b. Drop rows with no patient identifier ---------------------------
    df = df.dropna(subset=["Number"])

    # ---- 2c. Date parsing and conversion to days ----------------------------
    df["Data prelievo"] = pd.to_datetime(df["Data prelievo"])
    df["Follow Up Data"] = pd.to_datetime(df["Follow Up Data"])

    df["Follow Up Data"] = (df["Follow Up Data"] - df["Data prelievo"]).dt.days

    df["Data of death"] = pd.to_datetime(df["Data of death"], errors="coerce")
    df["Data of death"] = (df["Data of death"] - df["Data prelievo"]).dt.days
    # 0 means no death recorded (censored); NaN → 0 is the original notebook convention
    df["Data of death"] = df["Data of death"].fillna(0)

    # Competing-risk endpoints: create binary flags and convert dates to days
    endpoint_date_cols = ["CABG ", "Non Fatal AMI (Follow-Up)", "Ictus", "PCI"]
    for col in endpoint_date_cols:
        if col in df.columns:
            df[col + "_event"] = df[col].notna().astype(int)
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = (df[col] - df["Data prelievo"]).dt.days
            df[col] = df[col].fillna(0)

    # ---- 2d. Drop purely administrative columns -----------------------------
    drop_cols = [c for c in ADMIN_COLS if c in df.columns]
    # "Data prelievo" must be kept until here for the date arithmetic above
    drop_cols_final = [c for c in drop_cols if c != "Data prelievo"]
    df = df.drop(columns=drop_cols_final, errors="ignore")
    # Now we can drop the reference date too
    df = df.drop(columns=["Data prelievo"], errors="ignore")

    # ---- 2e. Column-stratified imputation  (replaces dropna) ----------------
    binary_cols = _infer_binary_cols(df)
    continuous_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in binary_cols
    ]

    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        # Binary / categorical: mode imputation (fallback to 0)
        for col in binary_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)

        # Continuous: median imputation
        for col in continuous_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        missing_after = df.isna().sum().sum()
        if missing_after > 0:
            warnings.warn(
                f"{missing_after} values remain missing after imputation "
                "(likely non-numeric columns). Dropping affected rows.",
                stacklevel=2,
            )
            df = df.dropna()

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3.  Task-specific feature-matrix builders
# ---------------------------------------------------------------------------

def _scale_continuous(
    df: pd.DataFrame,
    binary_cols: list[str],
    exclude_cols: list[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale continuous columns with StandardScaler.

    Binary columns and ``exclude_cols`` are left untouched.

    Parameters
    ----------
    df:
        Input dataframe.
    binary_cols:
        Columns with binary values — not scaled.
    exclude_cols:
        Additional columns to skip (e.g. duration, event).
    scaler:
        Pre-fitted scaler (for test-fold scaling in CV). If ``None``, a new
        scaler is fitted on ``df``.

    Returns
    -------
    (scaled_df, fitted_scaler)
    """
    skip = set(binary_cols) | set(exclude_cols)
    cont_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in skip
    ]

    df_out = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_out[cont_cols] = scaler.fit_transform(df[cont_cols])
    else:
        df_out[cont_cols] = scaler.transform(df[cont_cols])

    return df_out, scaler


def prepare_cox_data(
    df_main: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare data for the total-mortality Cox / DeepSurv task.

    Outcome columns
    ---------------
    ``duration_col = "Follow Up Data"``  (days to death or censoring)
    ``event_col    = "Total mortality"``  (1 = died, 0 = censored)

    Data-leakage prevention
    -----------------------
    * CABG/PCI/AMI/Ictus date columns and their event flags are excluded —
      they encode information about future events that occurred *after* the
      baseline blood draw.
    * Cause-of-death sub-categories (``CVD Death``, ``Fatal MI or Sudden
      death``, ``UnKnown``, ``Accident``, ``Suicide``) are excluded — they
      are a direct disaggregation of the outcome label.

    Parameters
    ----------
    df_main:
        Output of :func:`clean_and_impute`.
    scaler:
        Pre-fitted scaler for test-fold scaling.  If ``None``, a new scaler
        is fitted and returned.

    Returns
    -------
    (df_cox, fitted_scaler)
        ``df_cox`` contains only model-safe covariates plus the two outcome
        columns.
    """
    df = df_main.copy()

    # Drop outcome sub-components that leak the target
    leakage_cols = MORTALITY_LEAKAGE_COLS + LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
    leakage_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # Also drop "Data of death" — it IS the outcome for deceased patients
    df = df.drop(columns=["Data of death"], errors="ignore")

    binary_cols = _infer_binary_cols(df)
    df, scaler = _scale_continuous(
        df,
        binary_cols=binary_cols,
        exclude_cols=["Follow Up Data", "Total mortality"],
        scaler=scaler,
    )
    return df, scaler


def prepare_cardiovascular_data(
    df_main: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare data for the cardiovascular competing-risks task.

    Competing events
    ----------------
    * 0 — censored
    * 1 — cardiovascular death (``CVD Death == 1``)
    * 2 — non-cardiovascular death (all other deaths)

    Outcome columns
    ---------------
    ``duration_col = "Follow Up Data"``
    ``event_col    = "death"``  (0 / 1 / 2)

    Parameters
    ----------
    df_main:
        Output of :func:`clean_and_impute`.
    scaler:
        Pre-fitted scaler; fitted here if ``None``.

    Returns
    -------
    (df_cv, fitted_scaler)
    """
    df = df_main.copy()

    # Build competing-risk indicator
    df["Other deaths"] = (df["Total mortality"] - df["CVD Death"]).clip(lower=0)
    df["death"] = 0
    df.loc[df["CVD Death"] == 1, "death"] = 1
    df.loc[df["Other deaths"] == 1, "death"] = 2

    # Drop leakage and outcome sub-components
    drop_cols = (
        LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
        + ["Fatal MI or Sudden death", "UnKnown", "Total mortality",
           "Accident", "Suicide", "CVD Death", "Other deaths", "Data of death"]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    binary_cols = _infer_binary_cols(df)
    df, scaler = _scale_continuous(
        df,
        binary_cols=binary_cols,
        exclude_cols=["Follow Up Data", "death"],
        scaler=scaler,
    )
    return df, scaler


def prepare_mi_data(
    df_main: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare data for the myocardial infarction competing-risks task.

    Competing events
    ----------------
    * 0 — censored (no MI, no other death)
    * 1 — first MI event (fatal or non-fatal)
    * 2 — other death (competing risk)

    The event *time* is the earlier of: Non-Fatal AMI date, fatal MI date,
    or the follow-up end-date if no event occurred.

    Outcome columns
    ---------------
    ``duration_col = "MI_date"``
    ``event_col    = "events"``  (0 / 1 / 2)

    Parameters
    ----------
    df_main:
        Output of :func:`clean_and_impute`.
    scaler:
        Pre-fitted scaler; fitted here if ``None``.

    Returns
    -------
    (df_mi, fitted_scaler)
    """
    df = df_main.copy()

    # Build MI event indicator: fatal MI or non-fatal AMI
    df["MI_event"] = df[["Fatal MI or Sudden death", "Non Fatal AMI (Follow-Up)_event"]].max(axis=1)
    df["Other events"] = (df["Total mortality"] - df["MI_event"]).clip(lower=0)
    df["events"] = 0
    df.loc[df["MI_event"] == 1, "events"] = 1
    df.loc[df["Other events"] == 1, "events"] = 2

    # Event time: use Non-Fatal AMI date if non-fatal event; death date if fatal;
    # otherwise use follow-up end.
    conds = [
        df["Non Fatal AMI (Follow-Up)_event"] == 1,
        df["Fatal MI or Sudden death"] == 1,
    ]
    choices = [
        df["Non Fatal AMI (Follow-Up)"],
        df["Data of death"],
    ]
    df["MI_date"] = np.select(conds, choices, default=df["Follow Up Data"])

    drop_cols = (
        LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
        + ["Fatal MI or Sudden death", "UnKnown", "Total mortality",
           "Accident", "Suicide", "CVD Death", "Other events",
           "Data of death", "Follow Up Data", "MI_event"]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    binary_cols = _infer_binary_cols(df)
    df, scaler = _scale_continuous(
        df,
        binary_cols=binary_cols,
        exclude_cols=["MI_date", "events"],
        scaler=scaler,
    )
    return df, scaler


def prepare_stroke_data(
    df_main: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare data for the stroke competing-risks task.

    Competing events
    ----------------
    * 0 — censored
    * 1 — stroke (``Ictus``)
    * 2 — death from any other cause

    Event time is the stroke date if an event occurred; otherwise follow-up end.

    Outcome columns
    ---------------
    ``duration_col = "Ictus_date"``
    ``event_col    = "events"``  (0 / 1 / 2)

    Parameters
    ----------
    df_main:
        Output of :func:`clean_and_impute`.
    scaler:
        Pre-fitted scaler; fitted here if ``None``.

    Returns
    -------
    (df_stroke, fitted_scaler)
    """
    df = df_main.copy()

    df["events"] = 0
    df.loc[df["Ictus_event"] == 1, "events"] = 1
    df.loc[df["Total mortality"] == 1, "events"] = 2

    df["Ictus_date"] = np.where(
        df["Ictus_event"] == 1,
        df["Ictus"],
        df["Follow Up Data"],
    )

    drop_cols = (
        LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
        + ["Fatal MI or Sudden death", "UnKnown", "Total mortality",
           "Accident", "Suicide", "CVD Death", "Data of death", "Follow Up Data"]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    binary_cols = _infer_binary_cols(df)
    df, scaler = _scale_continuous(
        df,
        binary_cols=binary_cols,
        exclude_cols=["Ictus_date", "events"],
        scaler=scaler,
    )
    return df, scaler


# ---------------------------------------------------------------------------
# 4.  Train / val / test split utilities
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    event_col: str,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce stratified train / validation / test splits.

    Stratification is on the *binary* event indicator (event occurred vs.
    censored) so that all three splits have similar event rates.

    Parameters
    ----------
    df:
        Feature matrix including outcome columns.
    event_col:
        Column name of the event indicator (0 = censored, ≥1 = event).
    test_size:
        Fraction of data held out as the final test set.
    val_size:
        Fraction of the *training* data used as validation (for early stopping
        / hyperparameter selection).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    # Binarise event for stratification (competing risks have values > 1)
    strat_label = (df[event_col] > 0).astype(int)

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=strat_label,
        random_state=random_state,
    )
    strat_train_val = (df_train_val[event_col] > 0).astype(int)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size / (1.0 - test_size),
        stratify=strat_train_val,
        random_state=random_state,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def make_kfold_splits(
    df: pd.DataFrame,
    event_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified k-fold index pairs (train_idx, test_idx).

    Parameters
    ----------
    df:
        Full dataset.
    event_col:
        Event-indicator column used for stratification.
    n_splits:
        Number of folds.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    List of (train_indices, test_indices) arrays — one per fold.
    """
    strat_label = (df[event_col] > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(df, strat_label))
