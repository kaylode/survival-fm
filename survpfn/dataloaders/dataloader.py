"""
survpfn.dataloaders.benchmarks — Benchmark dataset loaders.

All loaders return ``(df, duration_col, event_col)`` where ``df`` has:
- feature columns one-hot encoded (categoricals only, NOT scaled)
- raw numeric columns (no StandardScaler applied)

Scaling is the caller's responsibility and must happen inside the CV fold loop
(fit on train, transform on test) to avoid data leakage.

Supported datasets
------------------
Public (via pycox):
    SUPPORT2, METABRIC, GBSG

Private Sirbu tasks (requires local Excel files in "Dataset Sirbu/"):
    SIRBU_mortality — total mortality
    SIRBU_cv        — cardiovascular mortality (competing risks, binarized)
    SIRBU_mi        — myocardial infarction (competing risks, binarized)
    SIRBU_stroke    — stroke (competing risks, binarized)
    SIRBU           — alias for SIRBU_mortality
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; leave numerics unchanged.

    Returns a DataFrame with named columns (no scaling applied).
    Boolean columns are cast to float.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=False)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


def _pycox_to_df(pycox_module) -> tuple[pd.DataFrame, str, str]:
    """Generic loader for pycox benchmark datasets.

    Parameters
    ----------
    pycox_module:
        A pycox dataset module (e.g. ``pycox.datasets.gbsg``).

    Returns
    -------
    (df, "duration", "event") where df includes one-hot-encoded features
    and raw (unscaled) outcome columns.
    """
    raw = pycox_module.read_df()
    X_df = _encode_df(raw.drop(columns=["duration", "event"]))
    df_out = X_df.copy()
    df_out["duration"] = raw["duration"].values
    df_out["event"] = raw["event"].values
    df_out = df_out.dropna().reset_index(drop=True)
    return df_out, "duration", "event"


# ---------------------------------------------------------------------------
# Public benchmark loaders
# ---------------------------------------------------------------------------

def load_support() -> tuple[pd.DataFrame, str, str]:
    """SUPPORT2 — 8,873 critically-ill ICU patients."""
    try:
        from pycox.datasets import support
    except ImportError as e:
        raise ImportError("pycox required: uv add pycox") from e
    return _pycox_to_df(support)


def load_metabric() -> tuple[pd.DataFrame, str, str]:
    """METABRIC — 1,904 breast-cancer patients."""
    try:
        from pycox.datasets import metabric
    except ImportError as e:
        raise ImportError("pycox required: uv add pycox") from e
    return _pycox_to_df(metabric)


def load_gbsg() -> tuple[pd.DataFrame, str, str]:
    """GBSG — 2,232 German breast-cancer patients."""
    try:
        from pycox.datasets import gbsg
    except ImportError as e:
        raise ImportError("pycox required: uv add pycox") from e
    return _pycox_to_df(gbsg)


# ---------------------------------------------------------------------------
# Sirbu private dataset (multi-task)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_sirbu_base() -> pd.DataFrame:
    """Load and clean the Sirbu dataset once, cache the result."""
    from survpfn.dataloaders.preprocessing import clean_and_impute

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


def load_sirbu_mortality() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — total mortality (binary event)."""
    from survpfn.dataloaders.preprocessing import prepare_cox_data
    df = prepare_cox_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Follow Up Data", "Total mortality"


def load_sirbu_cv() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — cardiovascular mortality (competing risks, events in {0,1,2})."""
    from survpfn.dataloaders.preprocessing import prepare_cardiovascular_data
    df = prepare_cardiovascular_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Follow Up Data", "death"


def load_sirbu_mi() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — myocardial infarction (competing risks, events in {0,1,2})."""
    from survpfn.dataloaders.preprocessing import prepare_mi_data
    df = prepare_mi_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "MI_date", "events"


def load_sirbu_stroke() -> tuple[pd.DataFrame, str, str]:
    """Sirbu — stroke (competing risks, events in {0,1,2})."""
    from survpfn.dataloaders.preprocessing import prepare_stroke_data
    df = prepare_stroke_data(_load_sirbu_base())
    return df.dropna().reset_index(drop=True), "Ictus_date", "events"


# ---------------------------------------------------------------------------
# Private / semi-private datasets (not in main benchmark registry)
# ---------------------------------------------------------------------------

def load_urrah_dataset(filepath="Dataset Sirbu/URRAH_TG_conLegenda.xlsx"):
    """Load the URRAH cardiovascular dataset from a local Excel file."""
    df = pd.read_excel(filepath, sheet_name=None)
    df_value = df["Urrah_virdis"]
    df_legend = df["Legenda"]

    missing_percent = df_value.isnull().mean() * 100
    print("Missing percentage URRAH:")
    print(missing_percent.sort_values(ascending=False).head())

    df_value = df_value.drop(columns=["LVH", "IMT", "VES", "IMA_BASE", "ALBURIA_MG_DL"])
    return df_value, df_legend


def load_mimic_dataset(data_dir, skip_tables=None):
    """Load MIMIC-III CSV tables from a directory."""
    data_dir = Path(data_dir)
    tables = {}
    for file in data_dir.glob("*.csv"):
        name = file.stem.lower()
        if skip_tables and name in skip_tables:
            continue
        tables[name] = pd.read_csv(file)
    return tables


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS: dict[str, Callable[[], tuple[pd.DataFrame, str, str]]] = {
    # Public
    "SUPPORT2":        load_support,
    "METABRIC":        load_metabric,
    "GBSG":            load_gbsg,
    # Sirbu tasks
    "SIRBU":           load_sirbu_mortality,   # backward-compat alias
    "SIRBU_mortality": load_sirbu_mortality,
    "SIRBU_cv":        load_sirbu_cv,
    "SIRBU_mi":        load_sirbu_mi,
    "SIRBU_stroke":    load_sirbu_stroke,
}
