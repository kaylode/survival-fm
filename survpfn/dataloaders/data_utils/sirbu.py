"""survpfn.data.preprocessing — data cleaning, imputation, and task preparation."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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



@lru_cache(maxsize=1)
def _load_sirbu_base() -> pd.DataFrame:
    """Load and clean the Sirbu dataset once, cache the result."""

    def load_and_merge_data(data_dir="Dataset Sirbu"):
        data_dir = Path(data_dir)

        # Read Excel files
        df_date = pd.read_excel(data_dir / "DataPrelievo.xlsx")
        df_update = pd.read_excel(data_dir / "Creatinina_AltriEsamiCorretti.xlsx")
        df_main = pd.read_excel(data_dir / "OrmoniTiroidei3Aprile2024.xlsx")

        # Merge shared columns logic from notebook
        # df1_aligned = df_main.set_index('Number')
        # df2_aligned = df_update.set_index('Number')
        # cols = ['Total cholesterol', 'HDL', 'LDL', 'Triglycerides']
        # The notebook showed how to find diffs. For actual loading, we just update df_main with df_update.

        # Set index
        df_main = df_main.set_index('Number')
        df_update = df_update.set_index('Number')

        # Keep index name safe
        df_main.index.name = 'Number'
        df_update.index.name = 'Number'

        # Columns to update
        cols_to_update = df_update.columns.intersection(df_main.columns)

        # Update shared columns
        df_main.update(df_update[cols_to_update])

        # Add new columns
        new_cols = df_update.columns.difference(df_main.columns)
        df_main = df_main.join(df_update[new_cols])

        # Reset index safely
        df_main = df_main.reset_index()

        # Add date column
        df_main = df_main.merge(
            df_date[['Number', 'Data prelievo']],
            on='Number',
            how='left'
        )

        return df_main

    df_raw = load_and_merge_data("Dataset Sirbu")


    return clean_and_impute(df_raw)


def prepare_allcause(df_main):
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


def load_sirbu_mortality() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — total mortality (binary event)."""
    df = prepare_allcause(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Follow Up Data", "Total mortality"

def load_sirbu_cv() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — cardiovascular mortality (competing risks, events in {0,1,2})."""
    df = prepare_cardiovascular_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Follow Up Data", "death"


def load_sirbu_mi() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — myocardial infarction (competing risks, events in {0,1,2})."""
    df = prepare_mi_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "MI_date", "events"

def load_sirbu_stroke() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — stroke (competing risks, events in {0,1,2})."""
    df = prepare_stroke_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Ictus_date", "events"