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