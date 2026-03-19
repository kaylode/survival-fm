import pandas as pd
from pathlib import Path

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
