"""
survpfn.dataloaders.seer — SEER Breast Cancer Dataset loader.

Source: data/SEER Breast Cancer Dataset .csv (~4,024 patients)

Columns
-------
Age                       : numeric
Race                      : categorical (one-hot)
Marital Status            : categorical (one-hot)
T Stage                   : ordinal (T1=1 … T4=4)
N Stage                   : ordinal (N1=1 … N3=3)
6th Stage                 : categorical (one-hot)
Grade                     : ordinal (Grade I=1 … Grade IV=4)
A Stage                   : ordinal (Regional=0, Distant=1)
Tumor Size                : numeric
Estrogen Status           : binary (Positive=1, Negative=0)
Progesterone Status       : binary (Positive=1, Negative=0)
Regional Node Examined    : numeric
Reginol Node Positive     : numeric (typo in original CSV preserved)
Survival Months           : time column
Status                    : event (Dead=1, Alive=0)

Returns
-------
(df, "Survival Months", "Status")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

_ORDINAL_MAPS = {
    "Grade": {
        "Well differentiated; Grade I":          1,
        "Moderately differentiated; Grade II":   2,
        "Poorly differentiated; Grade III":      3,
        "Undifferentiated; anaplastic; Grade IV": 4,
    },
    "T Stage": {
        "T1": 1, "T2": 2, "T3": 3, "T4": 4,
    },
    "N Stage": {
        "N1": 1, "N2": 2, "N3": 3,
    },
    "A Stage": {
        "Regional": 0, "Distant": 1,
    },
    "Estrogen Status": {
        "Positive": 1, "Negative": 0,
    },
    "Progesterone Status": {
        "Positive": 1, "Negative": 0,
    },
}

_ONEHOT_COLS = ["Race", "Marital Status", "6th Stage"]

_NUMERIC_COLS = [
    "Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive"
]

_TIME_COL  = "Survival Months"
_EVENT_COL = "Status"


def load_seer_dataset(
    filepath: str | Path | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load and preprocess the SEER Breast Cancer dataset.

    Parameters
    ----------
    filepath:
        Path to the CSV file.  Defaults to
        ``data/SEER Breast Cancer Dataset .csv`` relative to the working
        directory (the standard location in this repo).

    Returns
    -------
    (df, "Survival Months", "Status")
        df has numeric + ordinal + one-hot encoded features, with
        ``Survival Months`` (int) and ``Status`` (0/1) appended.
        No StandardScaler is applied — callers must scale inside CV folds.
    """
    if filepath is None:
        filepath = Path("data") / "SEER Breast Cancer Dataset .csv"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"SEER CSV not found at '{filepath}'. "
            "Download from https://www.kaggle.com/datasets/reihanenamdari/breast-cancer "
            "and place at data/SEER Breast Cancer Dataset .csv"
        )

    # ── Load CSV ────────────────────────────────────────────────────────────
    raw = pd.read_csv(filepath)

    # Strip whitespace from column names (CSV has 'Race ', 'T Stage ', etc.)
    raw.columns = [c.strip() for c in raw.columns]

    # Drop the unnamed empty column (4th column in the CSV)
    unnamed = [c for c in raw.columns if c.startswith("Unnamed") or c == ""]
    if unnamed:
        raw = raw.drop(columns=unnamed)

    # ── Event encoding ───────────────────────────────────────────────────────
    raw[_EVENT_COL] = (raw[_EVENT_COL].str.strip() == "Dead").astype(float)

    # ── Ordinal encoding ────────────────────────────────────────────────────
    for col, mapping in _ORDINAL_MAPS.items():
        if col not in raw.columns:
            continue
        raw[col] = raw[col].map(mapping)
        # Rows with unseen categories → NaN; drop below

    # ── One-hot encoding ─────────────────────────────────────────────────────
    present_onehot = [c for c in _ONEHOT_COLS if c in raw.columns]
    if present_onehot:
        dummies = pd.get_dummies(raw[present_onehot], drop_first=False)
        # Cast bool → float
        bool_cols = dummies.select_dtypes(include=["bool"]).columns
        dummies[bool_cols] = dummies[bool_cols].astype(float)
        raw = pd.concat([raw.drop(columns=present_onehot), dummies], axis=1)

    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in _NUMERIC_COLS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw[_TIME_COL] = pd.to_numeric(raw[_TIME_COL], errors="coerce")

    # ── Drop rows with missing values ─────────────────────────────────────────
    df = raw.dropna().reset_index(drop=True)

    # Ensure all feature columns are float
    out_cols = {_TIME_COL, _EVENT_COL}
    feat_cols = [c for c in df.columns if c not in out_cols]
    df[feat_cols] = df[feat_cols].astype(float)
    df[_TIME_COL]  = df[_TIME_COL].astype(float)
    df[_EVENT_COL] = df[_EVENT_COL].astype(float)

    return df, _TIME_COL, _EVENT_COL