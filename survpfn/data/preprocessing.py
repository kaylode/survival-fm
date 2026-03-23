"""survpfn.data.preprocessing — data cleaning, imputation, and task preparation."""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants — columns that encode future events or are purely administrative
# ---------------------------------------------------------------------------

LEAKY_DATE_COLS: list[str] = [
    "CABG ",
    "PCI",
    "Non Fatal AMI (Follow-Up)",
    "Ictus",
    "Data of death",
]

LEAKY_EVENT_FLAGS: list[str] = [
    "CABG _event",
    "PCI_event",
    "Non Fatal AMI (Follow-Up)_event",
    "Ictus_event",
]

ADMIN_COLS: list[str] = [
    "Collected by",
    "Cause of death",
    "CardiopatiaCongenita",
    "Number",
    "Data prelievo",
]

MORTALITY_LEAKAGE_COLS: list[str] = [
    "Fatal MI or Sudden death",
    "UnKnown",
    "Accident",
    "Suicide",
    "CVD Death",
]


def _infer_binary_cols(df: pd.DataFrame) -> list[str]:
    """Return column names whose non-null unique values are a subset of {0, 1}."""
    return [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]


def clean_and_impute(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw merged dataframe and impute missing values.

    Improvements over the original notebook version
    ------------------------------------------------
    * Uses column-stratified imputation (median for continuous, mode for
      binary/ordinal) instead of an aggressive ``dropna()`` that discarded
      ~22 % of the dataset.
    * Hard-coded row patches are guarded with index-existence checks.
    * Administrative columns are dropped after date arithmetic, not before.

    Parameters
    ----------
    df_raw:
        Output of :func:`survpfn.data.loader.load_and_merge_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with imputed values and all date columns converted
        to numeric (days).  Index is reset to RangeIndex.
    """
    df = df_raw.copy()

    # ---- Known data-entry fixes (guarded by index existence) ----------------
    if 8143 in df.index:
        df.loc[8143, "Documented resting \nor exertional ischemia"] = 1
    if 7285 in df.index:
        df.loc[7285, "Total mortality"] = 1
        df.loc[7285, "UnKnown"] = 1

    # ---- Drop rows with no patient identifier --------------------------------
    df = df.dropna(subset=["Number"])

    # ---- Date parsing and conversion to days --------------------------------
    df["Data prelievo"] = pd.to_datetime(df["Data prelievo"])
    df["Follow Up Data"] = pd.to_datetime(df["Follow Up Data"])
    df["Follow Up Data"] = (df["Follow Up Data"] - df["Data prelievo"]).dt.days

    df["Data of death"] = pd.to_datetime(df["Data of death"], errors="coerce")
    df["Data of death"] = (df["Data of death"] - df["Data prelievo"]).dt.days
    df["Data of death"] = df["Data of death"].fillna(0)

    # Competing-risk endpoints: create binary flags and convert dates to days
    endpoint_date_cols = ["CABG ", "Non Fatal AMI (Follow-Up)", "Ictus", "PCI"]
    for col in endpoint_date_cols:
        if col in df.columns:
            df[col + "_event"] = df[col].notna().astype(int)
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = (df[col] - df["Data prelievo"]).dt.days
            df[col] = df[col].fillna(0)

    # ---- Drop administrative columns ----------------------------------------
    drop_cols = [c for c in ADMIN_COLS if c in df.columns]
    # Keep "Data prelievo" until date arithmetic is done, then drop it
    drop_cols_no_date = [c for c in drop_cols if c != "Data prelievo"]
    df = df.drop(columns=drop_cols_no_date, errors="ignore")
    df = df.drop(columns=["Data prelievo"], errors="ignore")

    # ---- Column-stratified imputation (replaces aggressive dropna) ----------
    binary_cols = _infer_binary_cols(df)
    continuous_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in binary_cols
    ]

    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        for col in binary_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)

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


def prepare_cox_data(df_main):
    """Prepares data for Cox mortality model.

    NOTE: No scaling is applied here. Scaling must be done inside the CV fold
    loop (fit on train, transform on test) to avoid data leakage.

    Outcome-correlated columns dropped:
    - "Data of death": exact death date leaks the event time.
    - "Fatal MI or Sudden death", "CVD Death", "Accident", "Suicide", "UnKnown":
      cause-of-death sub-components that are direct components of the outcome.
    - "CABG _event", "Non Fatal AMI (Follow-Up)_event", "Ictus_event", "PCI_event":
      post-baseline event flags that are consequences of disease progression
      and would leak future outcome information.
    - "CABG ", "Non Fatal AMI (Follow-Up)", "Ictus", "PCI": corresponding dates.
    """
    df = df_main.copy()
    leakage_cols = MORTALITY_LEAKAGE_COLS + LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=["Data of death"], errors="ignore")
    return df


def prepare_cardiovascular_data(df_main):
    """Prepares data for cardiovascular competing risks.

    NOTE: No scaling is applied here. Scaling must be done inside the CV fold
    loop (fit on train, transform on test) to avoid data leakage.

    Competing events: 0 = censored, 1 = CVD death, 2 = non-CVD death.
    """
    df = df_main.copy()
    df["Other deaths"] = (df["Total mortality"] - df["CVD Death"]).clip(lower=0)
    df["death"] = 0
    df.loc[df["CVD Death"] == 1, "death"] = 1
    df.loc[df["Other deaths"] == 1, "death"] = 2

    drop_cols = (
        LEAKY_DATE_COLS + LEAKY_EVENT_FLAGS
        + ["Fatal MI or Sudden death", "UnKnown", "Total mortality",
           "Accident", "Suicide", "CVD Death", "Other deaths", "Data of death"]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def prepare_mi_data(df_main):
    """Prepares data for MI endpoint.

    NOTE: No scaling is applied here. Scaling must be done inside the CV fold
    loop (fit on train, transform on test) to avoid data leakage.

    Competing events: 0 = censored, 1 = MI event, 2 = other death.
    Duration column: "MI_date".
    """
    df = df_main.copy()
    df["MI_event"] = df[["Fatal MI or Sudden death", "Non Fatal AMI (Follow-Up)_event"]].max(axis=1)
    df["Other events"] = (df["Total mortality"] - df["MI_event"]).clip(lower=0)
    df["events"] = 0
    df.loc[df["MI_event"] == 1, "events"] = 1
    df.loc[df["Other events"] == 1, "events"] = 2

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
    return df


def prepare_stroke_data(df_main):
    """Prepares data for Stroke endpoint.

    NOTE: No scaling is applied here. Scaling must be done inside the CV fold
    loop (fit on train, transform on test) to avoid data leakage.

    Competing events: 0 = censored, 1 = stroke, 2 = other death.
    Duration column: "Ictus_date".
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
    return df
