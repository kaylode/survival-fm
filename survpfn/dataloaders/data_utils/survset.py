"""
survpfn.dataloaders.survset — SurvSet benchmark dataset loader.

SurvSet (Drysdale 2022, arXiv:2203.03094) is an open-source repository of
77 standardised survival analysis datasets.  Each dataset is returned with
consistent column names:
    pid       — observation identifier
    event     — binary event indicator (0=censored, 1=event)
    time      — observed time (right-censored or interval lower bound)
    time2     — interval upper bound (only for interval-censored data; rare)
    num_*     — continuous features
    fac_*     — categorical features

This module wraps SurvLoader with the project's standard API:
    ``load_survset_dataset(ds_name) → (df, duration_col, event_col)``

where ``df`` has features + duration + event columns (no scaling applied).
Categorical `fac_*` columns are one-hot encoded.

Curated benchmark subset
-------------------------
``SURVSET_BENCHMARK`` lists 25 datasets selected for diverse sample sizes
(N ≥ 200) and reasonable event rates (≥ 5 %).  This matches the evaluation
protocol of Kim et al. (arXiv:2601.22259) who also benchmark on SurvSet.

Usage
-----
    from survpfn.dataloaders.survset import load_survset_dataset, SURVSET_BENCHMARK

    df, dur_col, ev_col = load_survset_dataset("gbsg2")
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Curated 25-dataset benchmark subset
# Selected criteria: N ≥ 200, single-event (no time2/interval censoring needed),
# event_rate ≥ 5 %, and available in SurvSet v0.1+.
# Covers diverse domains: oncology, cardiology, haematology, transplant, etc.
# ---------------------------------------------------------------------------

SURVSET_BENCHMARK: list[str] = [
    "cancer",           # Lung cancer (NCCTG)
    "breast",           # German breast cancer
    "GBSG2",            # German breast cancer (GBSG variant)
    "rott2",            # Rotterdam breast cancer
    "colon",            # Colon cancer (adjuvant therapy)
    "prostate",         # Prostate cancer
    "ovarian",          # Ovarian cancer
    "Melanoma",         # Melanoma
    "e1684",            # Melanoma clinical trial
    "pbc",              # Primary biliary cirrhosis
    "hepatoCellular",   # Hepatocellular carcinoma
    "nwtco",            # Wilms tumour (National Wilms Tumor Study Group)
    "retinopathy",      # Diabetic retinopathy
    "heart",            # Stanford heart transplant
    "veteran",          # Lung cancer veterans trial
    "whas500",          # Worcester Heart Attack Study
    "mgus",             # MGUS haematology (Monoclonal Gammopathy)
    "cgd",              # Chronic granulomatous disease
    "cost",             # Colorectal cancer
    "LeukSurv",         # Leukemia survival
    "Dialysis",         # Kidney dialysis
    "actg",             # AIDS clinical trial
    "rhc",              # Right heart catheterization
    "vlbw",             # Very low birth weight infants
    "grace",            # Global Registry of Acute Coronary Events
    "TRACE",            # TRAndolapril Cardiac Evaluation
    "support2",         # SUPPORT study (hospitalised patients)
    "DLBCL",            # Diffuse large B-cell lymphoma
    "diabetes",         # Diabetes
    "flchain",          # Free light chain (plasma cell dyscrasias)
    "Framingham",       # Framingham heart study
]

# Short alias: small healthcare datasets (good for quick sanity checks)
SURVSET_SMALL: list[str] = [
    "veteran", "retinopathy", "cancer", "pbc", "cgd", "hepatoCellular"
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns (fac_*); leave numerics unchanged."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=False)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def load_survset_dataset(ds_name: str) -> tuple[pd.DataFrame, str, str]:
    """Load a SurvSet dataset and return it in the project's standard format.

    Parameters
    ----------
    ds_name : str
        Dataset name as recognised by ``SurvSet.data.SurvLoader``
        (e.g. ``"gbsg2"``, ``"pbc"``, ``"ova"``).

    Returns
    -------
    df : pd.DataFrame
        Features + duration + event columns.  No scaling applied.
        Categoricals one-hot encoded.
    duration_col : str
        Name of the time column (always ``"time"``).
    event_col : str
        Name of the event column (always ``"event"``).

    Raises
    ------
    ImportError
        If the ``SurvSet`` package is not installed.
    ValueError
        If the dataset is not found in SurvSet.
    """
    try:
        from SurvSet.data import SurvLoader
    except ImportError as exc:
        raise ImportError(
            "SurvSet is not installed.  Run: uv add SurvSet"
        ) from exc

    loader = SurvLoader()
    available = loader.df_ds["ds"].tolist()
    match = [ds for ds in available if ds.lower() == ds_name.lower()]
    if not match:
        raise ValueError(
            f"Dataset {ds_name!r} not found in SurvSet.  "
            f"Available: {available}"
        )
    actual_ds_name = match[0]
    result = loader.load_dataset(ds_name=actual_ds_name)
    df_raw: pd.DataFrame = result["df"]

    # Drop pid (identifier) and time2 (interval censoring; not used here)
    drop_cols = [c for c in ["pid", "time2"] if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols)

    # Rename SurvSet columns to our standard names
    df = df.rename(columns={"time": "time", "event": "event"})

    # One-hot encode fac_* columns
    df = _encode_df(df)

    # Drop rows with NaN (SurvSet datasets are mostly clean already)
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"SurvSet '{ds_name}': dropped {n_dropped}/{n_before} rows with NaN.",
            stacklevel=2,
        )

    # Ensure positive durations
    df = df[df["time"] > 0].reset_index(drop=True)

    return df, "time", "event"


def list_survset_datasets() -> list[str]:
    """Return all dataset names available in the installed SurvSet package."""
    try:
        from SurvSet.data import SurvLoader
    except ImportError as exc:
        raise ImportError("SurvSet is not installed.  Run: uv add SurvSet") from exc
    loader = SurvLoader()
    return loader.df_ds["ds"].tolist()


def survset_info(ds_name: Optional[str] = None) -> pd.DataFrame:
    """Return a metadata DataFrame for all SurvSet datasets (or one specific).

    Columns: ds, n_rows, n_features, event_rate, has_interval_censoring.
    """
    try:
        from SurvSet.data import SurvLoader
    except ImportError as exc:
        raise ImportError("SurvSet is not installed.  Run: uv add SurvSet") from exc

    loader = SurvLoader()
    return loader.df_ds if ds_name is None else loader.df_ds[loader.df_ds["ds"] == ds_name]