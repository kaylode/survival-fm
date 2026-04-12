import pandas as pd
from pathlib import Path

def load_data(dataset_name, data_dir="Dataset OrmoniTirodei"):
    """Load data for OrmoniTirodei or URRAH."""
    data_dir = Path(data_dir)
    match dataset_name:
        case "OrmoniTirodei":
            # Read Excel files
            df_date = pd.read_excel(data_dir / "DataPrelievo.xlsx")
            df_update = pd.read_excel(data_dir / "Creatinina_AltriEsamiCorretti.xlsx")
            df_main = pd.read_excel(data_dir / "OrmoniTiroidei3Aprile2024.xlsx")
            
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
        case "URRAH" | "HURRAH":
            # Read Excel files
            df_main = pd.read_excel(data_dir / "URRAH_TG_conLegenda.xlsx", sheet_name="Urrah_virdis")
            return df_main
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def load_urrah_dataset(filepath="data/Dataset Sirbu/URRAH_TG_conLegenda.xlsx"):
    """Load the URRAH cardiovascular dataset from a local Excel file."""
    df = pd.read_excel(filepath, sheet_name=None)
    df_value = df["Urrah_virdis"]
    df_legend = df["Legenda"]

    missing_percent = df_value.isnull().mean() * 100
    print("Missing percentage URRAH:")
    print(missing_percent.sort_values(ascending=False).head())

    df_value = df_value.drop(columns=["LVH", "IMT", "VES", "IMA_BASE", "ALBURIA_MG_DL"])

    # New update: SESSO check
    violations = df_value[~((df_value['SESSO'] == 0) & (df_value['SESSO0'] == 'DONNA')) & ~((df_value['SESSO'] == 1) & (df_value['SESSO0'] == 'UOMO'))]
    if not violations.empty:
        print("Rows that do not satisfy the rule (SESSO vs SESSO0 mismatch):")
        print(violations)
    
    return df_value, df_legend