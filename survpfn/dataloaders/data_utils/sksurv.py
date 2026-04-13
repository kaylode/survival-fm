from __future__ import annotations

import pandas as pd
from typing import Callable
from survpfn.dataloaders.data_utils.utils import _encode_df

# ---------------------------------------------------------------------------
# Public benchmark loaders
# ---------------------------------------------------------------------------

def _sksurv_to_df(
    X: pd.DataFrame,
    y,
    duration_field: str,
    event_field: str,
) -> tuple[pd.DataFrame, str, str]:
    """Convert scikit-survival (X, y) pair to a flat DataFrame.

    Parameters
    ----------
    X              : feature DataFrame (from sksurv loader)
    y              : structured numpy array with event / duration fields
    duration_field : name of the duration field in y
    event_field    : name of the event field in y
    """
    df = _encode_df(X.copy())
    df["duration"] = y[duration_field].astype(float)
    df["event"]    = y[event_field].astype(float)
    df = df.dropna().reset_index(drop=True)
    return df, "duration", "event"

def load_whas500() -> tuple[pd.DataFrame, str, str]:
    """WHAS500 — 500 cardiovascular patients (Worcester Heart Attack Study)."""
    try:
        from sksurv.datasets import load_whas500
    except ImportError as e:
        raise ImportError("scikit-survival required: uv add scikit-survival") from e
    X, y = load_whas500()
    return _sksurv_to_df(X, y, duration_field="lenfol", event_field="fstat")


def load_veterans() -> tuple[pd.DataFrame, str, str]:
    """Veterans' Lung Cancer — 137 male veterans with advanced lung cancer."""
    try:
        from sksurv.datasets import load_veterans_lung_cancer
    except ImportError as e:
        raise ImportError("scikit-survival required: uv add scikit-survival") from e
    X, y = load_veterans_lung_cancer()
    return _sksurv_to_df(X, y, duration_field="Survival_in_days", event_field="Status")


def load_flchain() -> tuple[pd.DataFrame, str, str]:
    """FLCHAIN — 7,874 patients (serum free light chain study, haematology)."""
    try:
        from sksurv.datasets import load_flchain
    except ImportError as e:
        raise ImportError("scikit-survival required: uv add scikit-survival") from e
    X, y = load_flchain()
    return _sksurv_to_df(X, y, duration_field="futime", event_field="death")

