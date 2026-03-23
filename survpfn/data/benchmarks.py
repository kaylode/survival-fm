"""
survpfn.data.benchmarks — External benchmark dataset loaders.

Supported datasets (all via pycox)
------------------------------------
* SUPPORT2   — Study to Understand Prognoses and Preferences for Outcomes and
               Risks of Treatments (8,873 critically-ill patients)
* METABRIC   — Molecular Taxonomy of Breast Cancer International Consortium
               (1,904 breast-cancer patients)
* GBSG       — German Breast Cancer Study Group
               (2,232 breast-cancer patients)

Also includes loaders for private / semi-private datasets:
* URRAH      — private URRAH cardiovascular cohort
* MIMIC-III  — PhysioNet MIMIC-III (raw CSV tables)

Each loader returns (X, y_time, y_event) as numpy arrays with basic
preprocessing applied (StandardScaler on continuous features; one-hot
encoding for categoricals).

Usage
-----
    from survpfn.data.benchmarks import load_support, load_metabric, load_gbsg, BENCHMARK_DATASETS

    X, y_time, y_event = load_support()

Dependencies
------------
    uv add pycox  # if not already installed
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_and_encode(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode object columns and StandardScale numeric columns.

    Parameters
    ----------
    df:
        Raw feature DataFrame (no outcome columns).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features_processed)
    """
    # Separate categorical and numeric columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    parts: list[np.ndarray] = []

    if num_cols:
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(df[num_cols].values.astype(float)))

    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=False)
        parts.append(dummies.values.astype(float))

    if not parts:
        return np.empty((len(df), 0), dtype=float)

    return np.hstack(parts)


def _drop_na_rows(
    X: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows where any feature or label is NaN."""
    valid = ~(
        np.isnan(X).any(axis=1)
        | np.isnan(y_time)
        | np.isnan(y_event)
    )
    return X[valid], y_time[valid], y_event[valid]


# ---------------------------------------------------------------------------
# Benchmark dataset loaders
# ---------------------------------------------------------------------------

def load_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the SUPPORT2 dataset.

    SUPPORT2 is a multi-centre study of hospitalised critically-ill patients.
    The standard survival task uses days-to-death as the outcome.

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Preprocessed feature matrix.
    y_time : np.ndarray, shape (n,)
        Days to death or censoring.
    y_event : np.ndarray, shape (n,)
        Event indicator (1 = died, 0 = censored).
    """
    try:
        from pycox.datasets import support
    except ImportError as exc:
        raise ImportError(
            "pycox is required for SUPPORT2.  Install with: uv add pycox"
        ) from exc

    df = support.read_df()
    y_time = df["duration"].values.astype(float)
    y_event = df["event"].values.astype(float)
    X_df = df.drop(columns=["duration", "event"])
    X = _scale_and_encode(X_df)
    X, y_time, y_event = _drop_na_rows(X, y_time, y_event)
    return X, y_time, y_event


def load_metabric() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the METABRIC dataset.

    METABRIC is a breast-cancer dataset with molecular and clinical covariates.

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Preprocessed feature matrix.
    y_time : np.ndarray, shape (n,)
        Duration (years).
    y_event : np.ndarray, shape (n,)
        Event indicator (1 = died, 0 = censored).
    """
    try:
        from pycox.datasets import metabric
    except ImportError as exc:
        raise ImportError(
            "pycox is required for METABRIC.  Install with: uv add pycox"
        ) from exc

    df = metabric.read_df()
    y_time = df["duration"].values.astype(float)
    y_event = df["event"].values.astype(float)
    X_df = df.drop(columns=["duration", "event"])
    X = _scale_and_encode(X_df)
    X, y_time, y_event = _drop_na_rows(X, y_time, y_event)
    return X, y_time, y_event


def load_gbsg() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the GBSG dataset.

    GBSG is a German breast-cancer randomised clinical trial dataset.

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Preprocessed feature matrix.
    y_time : np.ndarray, shape (n,)
        Duration (days).
    y_event : np.ndarray, shape (n,)
        Event indicator (1 = died/recurrence, 0 = censored).
    """
    try:
        from pycox.datasets import gbsg
    except ImportError as exc:
        raise ImportError(
            "pycox is required for GBSG.  Install with: uv add pycox"
        ) from exc

    df = gbsg.read_df()
    y_time = df["duration"].values.astype(float)
    y_event = df["event"].values.astype(float)
    X_df = df.drop(columns=["duration", "event"])
    X = _scale_and_encode(X_df)
    X, y_time, y_event = _drop_na_rows(X, y_time, y_event)
    return X, y_time, y_event


def load_sirbu() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the SIRBU cardiovascular dataset.

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Preprocessed feature matrix.
    y_time : np.ndarray, shape (n,)
        Follow-up duration (days).
    y_event : np.ndarray, shape (n,)
        Total mortality (1 = died, 0 = censored).
    """
    from survpfn.data.loader import load_and_merge_data
    from survpfn.data.preprocessing import clean_and_impute, prepare_cox_data

    # Load and merge raw Excel files
    df_raw = load_and_merge_data("Dataset Sirbu")
    
    # Impute missing values and clean dates
    df_clean = clean_and_impute(df_raw)
    
    # Drop outcome-correlated columns (leakage)
    df_task = prepare_cox_data(df_clean)
    
    y_time = df_task["Follow Up Data"].values.astype(float)
    y_event = df_task["Total mortality"].values.astype(float)
    X_df = df_task.drop(columns=["Follow Up Data", "Total mortality"])
    
    # Scale continuous and encode categorical
    X = _scale_and_encode(X_df)
    X, y_time, y_event = _drop_na_rows(X, y_time, y_event)
    
    return X, y_time, y_event


# ---------------------------------------------------------------------------
# Other dataset loaders (private / semi-private)
# ---------------------------------------------------------------------------

def load_urrah_dataset(filepath="Dataset Sirbu/URRAH_TG_conLegenda.xlsx"):
    """Load the URRAH cardiovascular dataset from a local Excel file."""
    df = pd.read_excel(filepath, sheet_name=None)
    df_value = df["Urrah_virdis"]
    df_legend = df["Legenda"]

    missing_percent = df_value.isnull().mean() * 100
    print("Missing percentage URRAH:")
    print(missing_percent.sort_values(ascending=False).head())

    df_value = df_value.drop(columns=['LVH', 'IMT', 'VES', 'IMA_BASE', "ALBURIA_MG_DL"])

    violations = df_value[~((df_value['SESSO'] == 0) & (df_value['SESSO0'] == 'DONNA')) & ~((df_value['SESSO'] == 1) & (df_value['SESSO0'] == 'UOMO'))]
    print("Rows that do not satisfy the rule:")
    print(violations)

    return df_value, df_legend


def load_mimic_dataset(data_dir, skip_tables=None):
    """Load MIMIC-III CSV tables from a directory.

    Parameters
    ----------
    data_dir:
        Path to directory containing MIMIC-III CSV files.
    skip_tables:
        Optional set/list of table names (stems) to skip.

    Returns
    -------
    dict mapping table_name -> pd.DataFrame
    """
    data_dir = Path(data_dir)
    tables = {}

    for file in data_dir.glob("*.csv"):
        table_name = file.stem.lower()
        if skip_tables and table_name in skip_tables:
            print(f"Skipping {table_name}")
            continue

        print(f"Loading {table_name}")
        tables[table_name] = pd.read_csv(file)

    return tables


def load_mimic():
    """Stub for a future survival-analysis-ready MIMIC-III pipeline.

    TODO: implement full MIMIC-III survival outcome extraction pipeline.
    This should produce a (df, duration_col, event_col) triple analogous
    to the pycox benchmark loaders. The raw table loading is available
    via :func:`load_mimic_dataset`.
    """
    raise NotImplementedError(
        "load_mimic() is a stub. "
        "Use load_mimic_dataset(data_dir) to load raw MIMIC-III CSV tables, "
        "then implement survival outcome extraction for your specific endpoint."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS: dict[str, Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "SUPPORT2": load_support,
    "METABRIC": load_metabric,
    "GBSG": load_gbsg,
    "SIRBU": load_sirbu,
}
