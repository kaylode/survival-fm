import pandas as pd
import numpy as np

def clean_and_impute(df_main):
    # Fix outlier as done in notebook
    df_main.loc[8143, "Documented resting \nor exertional ischemia"] = 1
    
    # Drop where Number is NaN
    df_main = df_main.dropna(subset=['Number'])

    # Date parsing
    df_main['Data prelievo'] = pd.to_datetime(df_main['Data prelievo'])
    df_main['Follow Up Data'] = pd.to_datetime(df_main['Follow Up Data'])
    
    # Fix mortality where death is known
    # violations = df_main[((df_main["Total mortality"] == 0) & (df_main["Data of death"].notna())) | ((df_main["Total mortality"] != 0) & (df_main["Data of death"].isna()))]
    df_main.loc[7285, "Total mortality"] = 1
    df_main.loc[7285, "UnKnown"] = 1

    # Calculate days
    df_main["Follow Up Data"] = (
        df_main["Follow Up Data"] - df_main["Data prelievo"]
    ).dt.days

    df_main["Data of death"] = (
        pd.to_datetime(df_main["Data of death"]) - df_main["Data prelievo"]
    ).dt.days
    df_main["Data of death"] = df_main["Data of death"].fillna(0)

    # Calculate target events
    target = ["CABG " ,"Non Fatal AMI (Follow-Up)","Ictus","PCI"]
    for t in target:
        df_main[t+"_event"] = df_main[t].notna().astype(int)
        df_main[t] = (
            pd.to_datetime(df_main[t]) - df_main["Data prelievo"]
        ).dt.days
        df_main[t] = df_main[t].fillna(0)
    
    # Drop irrelevant
    df_main = df_main.drop(columns=["Collected by", "Cause of death","CardiopatiaCongenita"])
    
    # Drop all NaNs
    df_main = df_main.dropna()

    return df_main


def prepare_cox_data(df_main):
    """Prepares data for Cox mortality model."""
    binary_cols = [
        col for col in df_main.columns
        if set(df_main[col].dropna().unique()).issubset({0,1})
    ]

    df_mortality = df_main.copy()
    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo"]
    
    tmp = df_mortality[cols_to_keep].copy()
    df_mortality = df_mortality.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_mortality)
    
    df_mortality = pd.DataFrame(scaled_array, columns=df_mortality.columns, index=df_mortality.index)
    df_mortality = pd.concat([df_mortality, tmp], axis=1)

    df_mortality = df_mortality.drop(columns=[
        "Data of death", "Fatal MI or Sudden death", "UnKnown", 
        "Accident", "Suicide", "Number", "CVD Death", "Data prelievo"
    ])
    
    return df_mortality

def prepare_cardiovascular_data(df_main):
    """Prepares data for cardiovascular competing risks."""
    df_cardiovascular = df_main.copy()
    df_cardiovascular["Other deaths"] = df_cardiovascular["Total mortality"] - df_cardiovascular["CVD Death"]
    df_cardiovascular["death"] = 0
    df_cardiovascular.loc[df_cardiovascular["CVD Death"] == 1, "death"] = 1
    df_cardiovascular.loc[df_cardiovascular["Other deaths"] == 1, "death"] = 2

    binary_cols = [
        col for col in df_cardiovascular.columns
        if set(df_cardiovascular[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "death"]
    
    tmp = df_cardiovascular[cols_to_keep].copy()
    df_cardiovascular = df_cardiovascular.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_cardiovascular)

    df_cardiovascular = pd.DataFrame(scaled_array, columns=df_cardiovascular.columns, index=df_cardiovascular.index)
    df_cardiovascular = pd.concat([df_cardiovascular, tmp], axis=1)

    df_cardiovascular = df_cardiovascular.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Other deaths", "Data of death"
    ])

    return df_cardiovascular

def prepare_mi_data(df_main):
    """Prepares data for MI endpoint."""
    df_MI = df_main.copy()
    df_MI["MI_event"] = df_main[["Fatal MI or Sudden death", "Non Fatal AMI (Follow-Up)_event"]].max(axis=1)
    df_MI["Other events"] = df_MI["Total mortality"] - df_MI["MI_event"]
    df_MI["events"] = 0
    df_MI.loc[df_MI["MI_event"] == 1, "events"] = 1
    df_MI.loc[df_MI["Other events"] == 1, "events"] = 2

    binary_cols = [
        col for col in df_MI.columns
        if set(df_MI[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "events"]
    tmp = df_MI[cols_to_keep].copy()
    df_MI = df_MI.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_MI)

    df_MI = pd.DataFrame(scaled_array, columns=df_MI.columns, index=df_MI.index)
    df_MI = pd.concat([df_MI, tmp], axis=1)

    conditions = [
        df_main["Non Fatal AMI (Follow-Up)_event"] == 1,
        df_main["Fatal MI or Sudden death"] == 1
    ]
    choices = [
        df_main["Non Fatal AMI (Follow-Up)"],
        df_main["Data of death"]
    ]
    df_MI["MI_date"] = np.select(conditions, choices, default=df_main["Follow Up Data"])

    df_MI = df_MI.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Other events", "Data of death", "Non Fatal AMI (Follow-Up)_event", 
        "Non Fatal AMI (Follow-Up)", "Follow Up Data", "MI_event"
    ])
    
    return df_MI

def prepare_stroke_data(df_main):
    """Prepares data for Stroke endpoint."""
    df_stroke = df_main.copy()
    df_stroke["events"] = 0
    df_stroke.loc[df_stroke["Ictus_event"] == 1, "events"] = 1
    df_stroke.loc[df_stroke["Total mortality"] == 1, "events"] = 2

    binary_cols = [
        col for col in df_stroke.columns
        if set(df_stroke[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "events"]
    tmp = df_stroke[cols_to_keep].copy()
    df_stroke = df_stroke.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_stroke)

    df_stroke = pd.DataFrame(scaled_array, columns=df_stroke.columns, index=df_stroke.index)
    df_stroke = pd.concat([df_stroke, tmp], axis=1)

    df_stroke["Ictus_date"] = np.where( 
        df_stroke["Ictus_event"] == 1,
        df_stroke["Ictus"], 
        df_stroke["Follow Up Data"] 
    )

    df_stroke = df_stroke.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Data of death", "Non Fatal AMI (Follow-Up)_event", 
        "Non Fatal AMI (Follow-Up)", "Follow Up Data", "Ictus_event", "Ictus"
    ])

    return df_stroke
