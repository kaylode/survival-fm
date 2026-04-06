import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# CR Loss and Helpers (Extracted from deephit_cr.py)
# ---------------------------------------------------------------------------

def discretize_competing_times(durations, events, n_bins):
    event_times = durations[events > 0]
    probs = np.linspace(0, 100, n_bins + 1)[1:]
    bins = np.unique(np.percentile(event_times, probs))
    t_disc = np.digitize(durations, bins, right=True)
    t_disc = np.clip(t_disc, 0, len(bins) - 1)
    return bins, t_disc

class SurvivalDatasetDeepHit(Dataset):
    """
    Dataset for DeepHit competing risks handling discretization.
    """
    def __init__(self, x, t, e, num_durations=100):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.long)
        
        # Discretize durations
        self.cuts, t_disc = discretize_competing_times(t, e, num_durations)
        self.t_disc = torch.tensor(t_disc, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.t[index], self.e[index], self.t_disc[index]

# crisp-nam specific dataset loading logic
def _load_framingham_raw():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_parties", "crisp-nam", "datasets", "framingham.csv")
    data = pd.read_csv(file_path).groupby("RANDID").first()
    
    cat_cols = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'educ']
    cont_cols = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    x_cat = pd.DataFrame(cat_imputer.fit_transform(data[cat_cols]), columns=cat_cols, index=data.index)
    x_cat = pd.get_dummies(x_cat, drop_first=True)
    
    cont_imputer = SimpleImputer(strategy='mean')
    x_cont = cont_imputer.fit_transform(data[cont_cols])
    
    x = np.hstack([x_cat.values, x_cont])
    feature_names = np.concatenate([x_cat.columns.values, cont_cols])
    
    event = np.zeros(len(data), dtype=int)
    time = (data['TIMEDTH'] - data['TIME']).values.copy()
    
    cvd_mask = data['CVD'] == 1
    event[cvd_mask] = 1
    time_cvd = (data['TIMECVD'] - data['TIME']).values
    time[cvd_mask] = time_cvd[cvd_mask]
    
    death_mask = (data['DEATH'] == 1) & ~cvd_mask
    event[death_mask] = 2
    
    valid = ~np.isnan(time) & (time > 0)
    return x[valid], time[valid] + 1, event[valid], feature_names

def load_framingham() -> tuple[pd.DataFrame, str, str]:
    x, t, e, feats = _load_framingham_raw()
    df = pd.DataFrame(x, columns=feats)
    df["duration"] = t
    df["event"] = e
    return df, "duration", "event"

def _load_pbc2_raw():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_parties", "crisp-nam", "datasets", "pbc2.csv")
    data = pd.read_csv(file_path).drop(columns=['id', 'sno.', 'year', 'status2'])
    
    event_type = np.where(data['status'] == 'dead', 1, np.where(data['status'] == 'transplanted', 2, 0))
    cont_cols = ['age', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
    x_cont = data[cont_cols].replace('NA', np.nan).astype(float)
    x_cont_imputed = SimpleImputer(strategy='mean').fit_transform(x_cont)
    
    cat_cols = ['sex', 'drug', 'ascites', 'hepatomegaly', 'spiders', 'edema']
    x_cat = data[cat_cols].fillna('missing')
    x_cat_imputed = SimpleImputer(strategy='most_frequent').fit_transform(x_cat)
    x_cat_encoded = pd.get_dummies(pd.DataFrame(x_cat_imputed, columns=cat_cols), drop_first=True)
    x_cat_encoded = x_cat_encoded.loc[:, ~x_cat_encoded.columns.str.contains('_missing')]
    
    x = np.hstack([x_cat_encoded.values, x_cont_imputed])
    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    
    t = data['years'].astype(float).values * 365.25
    valid = ~np.isnan(t)
    return x[valid], t[valid], event_type[valid], feature_names

def load_pbc2() -> tuple[pd.DataFrame, str, str]:
    x, t, e, feats = _load_pbc2_raw()
    df = pd.DataFrame(x, columns=feats)
    df["duration"] = t
    df["event"] = e
    return df, "duration", "event"

def _load_support_cr_raw():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_parties", "crisp-nam", "datasets", "support2.csv")
    data = pd.read_csv(file_path)
    
    is_cancer = data['ca'].astype(str).str.lower().str.contains("meta") | data['dzgroup'].astype(str).str.lower().str.contains("cancer")
    event_type = np.where(data['death'] == 1, np.where(is_cancer, 1, 2), 0)
    
    cont_cols = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'scoma', 'aps', 'sps', 'adls', 'adlsc', 'charges', 'totcst', 'totmcst', 'avtisst']
    x_cont = data[cont_cols]
    
    x_cont_imputed = SimpleImputer(strategy='median').fit_transform(x_cont) if x_cont.isnull().values.any() else x_cont.values
    cat_cols = ['sex', 'income', 'race', 'dnr', 'dementia', 'diabetes']
    x_cat = data[cat_cols]
    x_cat_imputed = SimpleImputer(strategy='most_frequent').fit_transform(x_cat) if x_cat.isnull().values.any() else x_cat.values
    x_cat_encoded = pd.get_dummies(pd.DataFrame(x_cat_imputed, columns=cat_cols), drop_first=True)
    
    x = np.hstack([x_cat_encoded.values, x_cont_imputed])
    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    
    t = data['d.time'].values
    valid = ~np.isnan(t)
    return x[valid], (t[valid] + 1), event_type[valid], feature_names

def load_support_cr() -> tuple[pd.DataFrame, str, str]:
    x, t, e, feats = _load_support_cr_raw()
    df = pd.DataFrame(x, columns=feats)
    df["duration"] = t
    df["event"] = e
    return df, "duration", "event"

def _load_synthetic_cr_raw():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "third_parties", "crisp-nam", "datasets", "synthetic_comprisk.csv")
    df = pd.read_csv(file_path)
    T_obs = df["time"].values.astype(np.float32)
    e = df["label"].values.astype(np.float32)
    feature_columns = [col for col in df.columns if col.startswith("feature")]
    X = df[feature_columns].values.astype(np.float32)
    return X, T_obs, e, feature_columns

def load_synthetic_cr() -> tuple[pd.DataFrame, str, str]:
    x, t, e, feats = _load_synthetic_cr_raw()
    df = pd.DataFrame(x, columns=feats)
    df["duration"] = t
    df["event"] = e
    return df, "duration", "event"

