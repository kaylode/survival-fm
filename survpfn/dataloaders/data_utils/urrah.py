import pandas as pd

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

