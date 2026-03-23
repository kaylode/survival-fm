import pandas as pd
from pathlib import Path

def load_urrah_dataset(filepath="Dataset Sirbu/URRAH_TG_conLegenda.xlsx"):
    df = pd.read_excel(filepath, sheet_name=None)
    df_value = df["Urrah_virdis"]
    df_legend = df["Legenda"]

    missing_percent = df_value.isnull().mean() * 100
    print("Missing percentage URRAH:")
    print(missing_percent.sort_values(ascending=False).head())

    df_value = df_value.drop(columns=['LVH', 'IMT', 'VES', 'IMA_BASE',"ALBURIA_MG_DL"])

    violations = df_value[~((df_value['SESSO'] == 0) & (df_value['SESSO0'] == 'DONNA')) & ~((df_value['SESSO'] == 1) & (df_value['SESSO0'] == 'UOMO'))]
    print("Rows that do not satisfy the rule:")
    print(violations)
    
    return df_value, df_legend

def load_mimic_dataset(data_dir, skip_tables=None):
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

# mimic_path = "MIMIC/MIMIC/files/mimiciii/1.4"
# tables = load_mimic_dataset(mimic_path, skip_tables=["chartevents","labevents"])
