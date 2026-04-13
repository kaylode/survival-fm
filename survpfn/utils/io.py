from __future__ import annotations
import os
import pandas as pd
from pathlib import Path


def save_results(df: pd.DataFrame, path: str | Path, overwrite: bool = False) -> None:
    """Save results DataFrame to CSV, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load results CSV."""
    return pd.read_csv(path)
