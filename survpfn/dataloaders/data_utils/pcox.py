from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from survpfn.dataloaders.data_utils.utils import _encode_df


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

