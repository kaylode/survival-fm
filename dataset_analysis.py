import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis


# Read Excel file
df_date = pd.read_excel("Dataset Sirbu/DataPrelievo.xlsx")

print(df_date.head())
print(df_date.info())

# Read Excel file
df_update = pd.read_excel("Dataset Sirbu/Creatinina_AltriEsamiCorretti.xlsx")

# Show first 5 rows
print(df_update.head())
print(df_update.info())

# Read Excel file
df_main = pd.read_excel("Dataset Sirbu/OrmoniTiroidei3Aprile2024.xlsx")

# Show first 5 rows
print(df_main.head())
print(df_main.info())

cols = ['Total cholesterol', 'HDL', 'LDL', 'Triglycerides']
df1_aligned = df_main.set_index('Number')
df2_aligned = df_update.set_index('Number')

common_index = df1_aligned.index.intersection(df2_aligned.index)

df1_aligned = df1_aligned.loc[common_index, cols]
df2_aligned = df2_aligned.loc[common_index, cols]

diff_mask = df1_aligned[cols] != df2_aligned[cols]

rows_with_diff = diff_mask.any(axis=1)

print(rows_with_diff[rows_with_diff])

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
df_update = df_update.reset_index()

df_main = df_main.merge(
    df_date[['Number', 'Data prelievo']],  
    on='Number',
    how='left' 
)

print(df_main.info())

print(df_main.describe())

print(df_main["Documented resting \nor exertional ischemia"].nlargest(5))
df_main.loc[8143, "Documented resting \nor exertional ischemia"] = 1
print(df_main["Documented resting \nor exertional ischemia"].nlargest(5))

df_main = df_main.dropna(subset=['Number'])

df_main['Data prelievo'] = pd.to_datetime(df_main['Data prelievo'])
df_main['Follow Up Data'] = pd.to_datetime(df_main['Follow Up Data'])

print((df_main['Data prelievo'] <= df_main['Follow Up Data']).all())

violations = df_main[
    ((df_main["Total mortality"] == 0) & (df_main["Data of death"].notna())) |
    ((df_main["Total mortality"] != 0) & (df_main["Data of death"].isna()))
]

print(violations["Total mortality"])
print(violations["Data of death"])

df_main.loc[7285, "Total mortality"] = 1
df_main.loc[7285, "UnKnown"] = 1

print(df_main.iloc[7285][["Total mortality", "UnKnown"]])

df_main["Follow Up Data"] = (
    df_main["Follow Up Data"] - df_main["Data prelievo"]
).dt.days

df_main["Data of death"] = (
    df_main["Data of death"] - df_main["Data prelievo"]
).dt.days
df_main["Data of death"] = df_main["Data of death"].fillna(0)


target = ["CABG " ,"Non Fatal AMI (Follow-Up)","Ictus","PCI"]

for t in target:
    df_main[t+"_event"] = df_main[t].notna().astype(int)

    df_main[t] = (
        df_main[t] - df_main["Data prelievo"]
    ).dt.days

    df_main[t] = df_main[t].fillna(0)

print(df_main.info())
print(df_main.head())



missing = df_main.isna().sum().sort_values(ascending=False)
missing_percent = (df_main.isna().mean()*100).sort_values(ascending=False)

missing_table = pd.concat([missing, missing_percent], axis=1)
missing_table.columns = ["Missing Values", "Percentage"]

print(missing_table.head(20))

df_main = df_main.drop(columns=["Collected by", "Cause of death","CardiopatiaCongenita"])
print(df_main.shape)
df_main = df_main.dropna()
print(df_main.shape)

numeric_df = df_main.select_dtypes(include=[np.number])

corr_matrix = numeric_df.corr()

plt.figure(figsize=(24,20))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

corr_threshold = 0.90

corr_matrix = numeric_df.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_features = [column for column in upper.columns if any(upper[column] > corr_threshold)]

print("Highly correlated features:")
print(high_corr_features)

binary_cols = [
    col for col in df_main.columns
    if set(df_main[col].dropna().unique()).issubset({0,1})
]

print(binary_cols)

# To avoid data leakage, we should not include features that are directly related to the target variable. 
# In this case, we will drop the columns that are related to mortality and cause of death, 
# as they would provide direct information about the outcome we are trying to predict.
df_mortality = df_main.copy()

cols_to_keep = binary_cols + ["Follow Up Data", "Data of death","Data prelievo"]

tmp = df_mortality[cols_to_keep].copy()

df_mortality = df_mortality.drop(columns=cols_to_keep)

# Standardize continuous features
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_mortality)

df_mortality = pd.DataFrame(scaled_array, columns=df_mortality.columns, index=df_mortality.index)

df_mortality = pd.concat([df_mortality, tmp], axis=1)

df_mortality = df_mortality.drop(columns=[   "Data of death",
                                        "Fatal MI or Sudden death",  
                                        "UnKnown", 
                                        "Accident",
                                        "Suicide",
                                        "Number",
                                        "CVD Death",
                                        "Data prelievo"])

print(df_mortality.info())

results = []
for c in df_mortality.columns: 
    if c not in ["Follow Up Data", "Total mortality"]:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(
            df_mortality[[c,'Follow Up Data','Total mortality']], 
            duration_col='Follow Up Data', 
            event_col='Total mortality')
        coef = cph.summary.loc[c, "coef"]
        se = cph.summary.loc[c, "se(coef)"]
        p = cph.summary.loc[c, "p"]
        results.append({"feature": c, "coef": coef, "se": se, "p": p})

importance = pd.DataFrame(results)
importance_temp = importance[importance["p"] < 0.05].copy()

importance["abs_logHR"] = importance["coef"].abs()
importance = importance[importance["p"] < 0.05]
importance = importance.sort_values("abs_logHR", ascending=False).head(20)

plt.figure(figsize=(8,10))
plt.barh(importance["feature"], importance["abs_logHR"])
plt.gca().invert_yaxis()
plt.xlabel("|log(HR)|")
plt.title("Univariate Cox Feature Importance")
plt.show()

# Calculate HR and 95% CI
importance_temp["HR"] = np.exp(importance_temp["coef"])
importance_temp["HR_lower"] = np.exp(importance_temp["coef"] - 1.96*importance_temp["se"])
importance_temp["HR_upper"] = np.exp(importance_temp["coef"] + 1.96*importance_temp["se"])

# Sort by HR
importance_temp = importance_temp.sort_values("HR")

# Plot forest plot
plt.figure(figsize=(8, len(importance_temp)*0.5))
plt.errorbar(
    importance_temp["HR"],
    range(len(importance_temp)),
    xerr=[importance_temp["HR"] - importance_temp["HR_lower"], importance_temp["HR_upper"] - importance_temp["HR"]],
    fmt='o',
    color='black',
    ecolor='gray',
    capsize=3
)
plt.yticks(range(len(importance_temp)), importance_temp["feature"])
plt.axvline(x=1, color='red', linestyle='--')  # no-effect line
plt.xlabel("Hazard Ratio (HR)")
plt.title("Univariate Cox Forest Plot")
plt.gca().invert_yaxis()
plt.show()

# Columns to use as covariates (exclude duration/event)
#covariates = [c for c in df_mortality.columns if c not in ['Data of death', 'Total mortality']]

# Fit multivariate Cox model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_mortality,#[covariates],# + ['Data of death', 'Total mortality']], 
        duration_col='Follow Up Data', 
        event_col='Total mortality')

print("Concordance Index:", cph.concordance_index_)

# Extract results
importance = cph.summary.reset_index()
importance.rename(columns={'index':'feature'}, inplace=True)

# Keep only significant features
importance = importance[importance["p"] < 0.05].copy()
importance_temp = importance.copy()

# Calculate |log(HR)|
importance["abs_logHR"] = importance["coef"].abs()

# Sort by |log(HR)| for plotting
importance = importance.sort_values("abs_logHR", ascending=False)  # top 20 if needed: .head(20)

# Plot |log(HR)|
plt.figure(figsize=(8,10))
plt.barh(importance["covariate"], importance["abs_logHR"])
plt.gca().invert_yaxis()
plt.xlabel("|log(HR)|")
plt.title("Multivariate Cox Feature Importance")
plt.show()

# Calculate HR and 95% CI
importance_temp["HR"] = np.exp(importance_temp["coef"])
importance_temp["HR_lower"] = np.exp(importance_temp["coef"] - 1.96*importance_temp["se(coef)"])
importance_temp["HR_upper"] = np.exp(importance_temp["coef"] + 1.96*importance_temp["se(coef)"])

# Sort by HR for plotting
importance_temp = importance_temp.sort_values("HR")

# Plot forest plot
plt.figure(figsize=(8, len(importance_temp)*0.5))
plt.errorbar(
    importance_temp["HR"],
    range(len(importance_temp)),
    xerr=[importance_temp["HR"] - importance_temp["HR_lower"], importance_temp["HR_upper"] - importance_temp["HR"]],
    fmt='o',
    color='black',
    ecolor='gray',
    capsize=3
)
plt.yticks(range(len(importance_temp)), importance_temp["covariate"])
plt.axvline(x=1, color='red', linestyle='--') 
plt.xlabel("Hazard Ratio (HR)")
plt.title("Multivariate Cox Forest Plot")
plt.gca().invert_yaxis()
plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# Assume 'T' = duration column, 'E' = event indicator
T = df_mortality["Follow Up Data"]         
E = df_mortality["Total mortality"]
X = df_mortality.drop(columns=["Follow Up Data", "Total mortality"]) 

# Standardize numerical features (keeping DataFrame)
#scaler = StandardScaler()
#X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split (still as DataFrames / Series)
X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
    X, T, E, test_size=0.2, random_state=42
)

# Convert DataFrames / Series to torch tensors for pycox
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)
E_train_tensor = torch.tensor(E_train.values, dtype=torch.float32)
T_test_tensor = torch.tensor(T_test.values, dtype=torch.float32)
E_test_tensor = torch.tensor(E_test.values, dtype=torch.float32)

import torchtuples as tt
from pycox.models import CoxPH
import torch
from torch import nn

# Network definition
in_features = X_train.shape[1]
num_nodes = [32, 32]  # hidden layers
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, 1, batch_norm=True, dropout=dropout, activation=nn.ReLU)

# Optimizer
optimizer = tt.optim.AdamWR(lr=0.01)

# DeepSurv model
model = CoxPH(net, tt.optim.AdamWR)

# Prepare dataset for pycox
#from pycox.preprocessing.label_transforms import LabTransCoxPH

# DeepSurv uses Cox PH loss, no need to transform labels
# But pycox allows easy handling of mini-batches

# Convert to torch tensors
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#T_train_tensor = torch.tensor(T_train, dtype=torch.float32)
#E_train_tensor = torch.tensor(E_train, dtype=torch.float32)

# Fit model
epochs = 100
batch_size = 128

model.fit(X_train_tensor, (T_train_tensor, E_train_tensor), batch_size, epochs, verbose=True)

# Concordance index
from pycox.evaluation import EvalSurv

# Predict log-risk for test set
risk_scores = model.predict(X_test_tensor)

# Convert to survival function
# For Cox, higher risk score = higher hazard
# EvalSurv can compute C-index
import numpy as np

# Use Kaplan-Meier baseline estimate
baseline_hazard = model.compute_baseline_hazards()
surv = model.predict_surv_df(X_test_tensor)

# Concordance index
from pycox.evaluation import EvalSurv
ev = EvalSurv(surv, T_test.values, E_test.values, censor_surv='km')
c_index = ev.concordance_td()
print("C-index:", c_index)

df_cardiovascular = df_main.copy()
df_cardiovascular["Other deaths"] = df_cardiovascular["Total mortality"] - df_cardiovascular["CVD Death"]
df_cardiovascular["death"] = 0
df_cardiovascular.loc[df_cardiovascular["CVD Death"] == 1, "death"] = 1
df_cardiovascular.loc[df_cardiovascular["Other deaths"] == 1, "death"] = 2

df_cardiovascular["death"].value_counts()

binary_cols = [
    col for col in df_cardiovascular.columns
    if set(df_cardiovascular[col].dropna().unique()).issubset({0,1})
]

print(binary_cols)

cols_to_keep = binary_cols + ["Follow Up Data", "Data of death","Data prelievo","death"]

tmp = df_cardiovascular[cols_to_keep].copy()

df_cardiovascular = df_cardiovascular.drop(columns=cols_to_keep)

scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_cardiovascular)

df_cardiovascular = pd.DataFrame(scaled_array, columns=df_cardiovascular.columns, index=df_cardiovascular.index)

df_cardiovascular = pd.concat([df_cardiovascular, tmp], axis=1)

df_cardiovascular = df_cardiovascular.drop(columns=[ "Fatal MI or Sudden death",  
                                            "UnKnown",
                                            "Total mortality", 
                                            "Accident",
                                            "Suicide",
                                            "Number",
                                            "Data prelievo",
                                            "CVD Death",
                                            "Other deaths",
                                            "Data of death"
                                             ])

print(df_cardiovascular.info())
print(df_cardiovascular["death"].value_counts())

df_cardiovascular["death"].value_counts()

df_temp = df_cardiovascular.copy()
df_temp["death"] = df_temp["death"] == 1

# initialize CoxPHFitter
cph_cv = CoxPHFitter(penalizer=0.1)

# fit
cph_cv.fit(df_temp, duration_col="Follow Up Data", event_col="death")

df_temp = df_cardiovascular.copy()
df_temp["death"] = df_temp["death"] == 2

# initialize CoxPHFitter
cph_other = CoxPHFitter(penalizer=0.1)

# fit
cph_other.fit(df_temp, duration_col="Follow Up Data", event_col="death")

print("Concordance index CV: ", cph_cv.concordance_index_)
print("Concordance index Other: ", cph_other.concordance_index_)

coef_cv = cph_cv.summary["exp(coef)"]
coef_other = cph_other.summary["exp(coef)"]
p_cv = cph_cv.summary["p"]
p_other = cph_other.summary["p"]

significant = (
    (p_cv < 0.05) |
    (p_other < 0.05)
)

comparison = pd.DataFrame({
    "CV death": coef_cv,
    "Other death": coef_other,
    "CV death p": p_cv,
    "Other death p": p_other
})

comparison = comparison[significant]

print(comparison)
comparison[["CV death","Other death"]].plot.bar(figsize=(10,6))

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycox.models import DeepHitSingle
import torchtuples as tt
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load and prepare data
# ------------------------------

# Columns: T = duration, E = event type (0 = censored, 1,2,... = competing risks)
T = df_cardiovascular["Follow Up Data"]
E = df_cardiovascular["death"]
X = df_cardiovascular.drop(columns=["Follow Up Data", "death"])

# Standardize features
#scaler = StandardScaler()
#X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
    X, T, E, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32)
E_train_tensor = torch.tensor(E_train.values, dtype=torch.int64)
T_test_tensor = torch.tensor(T_test.values, dtype=torch.float32)
E_test_tensor = torch.tensor(E_test.values, dtype=torch.int64)

# ------------------------------
# 2. Define DeepHit network
# ------------------------------

#from pycox.preprocessing.feature_transforms import StandardScaler
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

# Choose number of discrete time intervals
num_durations = 100  # can also try 200

labtrans = LabTransDiscreteTime(num_durations)

# Fit and transform train labels
y_train = labtrans.fit_transform(T_train.values, E_train.values)
y_test = labtrans.transform(T_test.values, E_test.values)

# Now the number of output nodes = labtrans.out_features
out_features = labtrans.out_features
print("Network output nodes needed:", out_features)  # should be 100

in_features = X_train.shape[1]
num_nodes = [64, 32]  # hidden layers
dropout = 0.1

net = tt.practical.MLPVanilla(
    in_features, num_nodes, out_features = out_features ,  # output nodes for discretized times
    batch_norm=True, dropout=dropout, activation=torch.nn.ReLU
)

# Optimizer
optimizer = tt.optim.AdamWR(lr=0.01)

# Define DeepHit model
model = DeepHitSingle(net, tt.optim.AdamWR, duration_index=labtrans.cuts)  # duration_index will be inferred automatically

# ------------------------------
# 3. Train model
# ------------------------------
epochs = 50
batch_size = 128

log = model.fit(
    X_train_tensor, 
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=True
)

# ------------------------------
# 4. Predict survival functions
# ------------------------------
surv = model.predict_surv_df(X_test_tensor)  # returns survival probability over discretized times

# ------------------------------
# 5. Evaluate model
# ------------------------------
# For competing risks, we evaluate CIF for each event type
ev = EvalSurv(surv, T_test.values, E_test.values, censor_surv='km')

# Concordance index (overall)
c_index = ev.concordance_td()
print("C-index:", c_index)

# ------------------------------
# 6. Plot example CIFs
# ------------------------------
times = surv.index.values
for i, event_type in enumerate([1, 2]):  # replace with your event types
    # Cumulative incidence function for event_type
    cif = 1 - surv.loc[:, ev.event_of_interest(event_type)]
    plt.step(times, cif.mean(axis=1), label=f"CIF Event {event_type}")

plt.xlabel("Time")
plt.ylabel("Cumulative Incidence")
plt.title("Cumulative Incidence Function (CIF)")
plt.legend()
plt.show()

#cph_cv.plot()

#cph_other.plot()

df_MI = df_main.copy()
df_MI["MI_event"] = df_main[["Fatal MI or Sudden death", "Non Fatal AMI (Follow-Up)_event"]].max(axis=1)
df_MI["Other events"] = df_MI["Total mortality"] - df_MI["MI_event"]
df_MI["events"] = 0
df_MI.loc[df_MI["MI_event"] == 1, "events"] = 1
df_MI.loc[df_MI["Other events"] == 1, "events"] = 2

df_MI["events"].value_counts()

binary_cols = [
    col for col in df_cardiovascular.columns
    if set(df_cardiovascular[col].dropna().unique()).issubset({0,1})
]

print(binary_cols)

cols_to_keep = binary_cols + ["Follow Up Data", "Data of death","Data prelievo", "events"]

tmp = df_MI[cols_to_keep].copy()

df_MI = df_MI.drop(columns=cols_to_keep)

print(df_MI.info())

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

df_MI["MI_date"] = np.select(
    conditions,
    choices,
    default=df_main["Follow Up Data"]
)

df_MI = df_MI.drop(columns=[    "Fatal MI or Sudden death",  
                                "UnKnown",
                                "Total mortality", 
                                "Accident",
                                "Suicide",
                                "Number",
                                "Data prelievo",
                                "CVD Death",
                                "Other events",
                                "Data of death",
                                "Non Fatal AMI (Follow-Up)_event",
                                "Non Fatal AMI (Follow-Up)",
                                "Follow Up Data",
                                "MI_event"
                                
                                             ])



print(df_MI.info())
print(df_MI["events"].value_counts())

df_temp = df_MI.copy()
df_temp["events"] = df_temp["events"] == 1

# initialize CoxPHFitter
cph_MI = CoxPHFitter(penalizer=0.1)

# fit
cph_MI.fit(df_temp, duration_col="MI_date", event_col="events")

df_temp = df_MI.copy()
df_temp["events"] = df_temp["events"] == 2

# initialize CoxPHFitter
cph_other = CoxPHFitter(penalizer=0.1)

# fit
cph_other.fit(df_temp, duration_col="MI_date", event_col="events")

print("Concordance index CV: ", cph_MI.concordance_index_)
print("Concordance index Other: ", cph_other.concordance_index_)

coef_MI = cph_MI.summary["exp(coef)"]
coef_other = cph_other.summary["exp(coef)"]
p_MI = cph_MI.summary["p"]
p_other = cph_other.summary["p"]

significant = (
    (p_MI < 0.05) |
    (p_other < 0.05)
)

comparison = pd.DataFrame({
    "MI event": coef_MI,
    "Other event": coef_other,
    "MI event p": p_MI,
    "Other event p": p_other
})

comparison = comparison[significant]

print(comparison)
comparison[["MI event","Other event"]].plot.bar(figsize=(10,6))

df_stroke = df_main.copy()
df_stroke["events"] = 0
df_stroke.loc[df_stroke["Ictus_event"] == 1, "events"] = 1
df_stroke.loc[df_stroke["Total mortality"] == 1, "events"] = 2

binary_cols = [
    col for col in df_cardiovascular.columns
    if set(df_cardiovascular[col].dropna().unique()).issubset({0,1})
]

print(binary_cols)

cols_to_keep = binary_cols + ["Follow Up Data", "Data of death","Data prelievo", "events"]

tmp = df_stroke[cols_to_keep].copy()

df_stroke = df_stroke.drop(columns=cols_to_keep)

scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_stroke)

df_stroke = pd.DataFrame(scaled_array, columns=df_stroke.columns, index=df_stroke.index)

df_stroke = pd.concat([df_stroke, tmp], axis=1)

df_stroke["Ictus_date"] = np.where( 
    df_stroke["Ictus_event"] == 1,
    df_stroke["Ictus"], 
    df_stroke["Follow Up Data"] )

df_stroke = df_stroke.drop(columns=[    "Fatal MI or Sudden death",  
                                "UnKnown",
                                "Total mortality", 
                                "Accident",
                                "Suicide",
                                "Number",
                                "Data prelievo",
                                "CVD Death",
                                "Data of death",
                                "Non Fatal AMI (Follow-Up)_event",
                                "Non Fatal AMI (Follow-Up)",
                                "Follow Up Data",
                                "Ictus_event",
                                "Ictus"])

print(df_stroke.info())
print(df_stroke["events"].value_counts())

df_temp = df_stroke.copy()
df_temp["events"] = df_temp["events"] == 1

# initialize CoxPHFitter
cph_stroke = CoxPHFitter(penalizer=0.1)

# fit
cph_stroke.fit(df_temp, duration_col="Ictus_date", event_col="events")

df_temp = df_stroke.copy()
df_temp["events"] = df_temp["events"] == 2

# initialize CoxPHFitter
cph_other = CoxPHFitter(penalizer=0.1)

# fit
cph_other.fit(df_temp, duration_col="Ictus_date", event_col="events")

print("Concordance index CV: ", cph_stroke.concordance_index_)
print("Concordance index Other: ", cph_other.concordance_index_)

coef_stroke = cph_stroke.summary["exp(coef)"]
coef_other = cph_other.summary["exp(coef)"]
p_stroke = cph_stroke.summary["p"]
p_other = cph_other.summary["p"]

significant = (
    (p_stroke < 0.05) |
    (p_other < 0.05)
)

comparison = pd.DataFrame({
    "Stroke event": coef_stroke,
    "Other event": coef_other,
    "Stroke event p": p_stroke,
    "Other event p": p_other
})

comparison = comparison[significant]

print(comparison)
comparison[["Stroke event","Other event"]].plot.bar(figsize=(10,6))

df = pd.read_excel("Dataset Sirbu/URRAH_TG_conLegenda.xlsx",sheet_name=None)

print(df.keys()) 
df_value = df["Urrah_virdis"]
df_legend = df["Legenda"]

print(df_value.info())

missing_percent = df_value.isnull().mean() * 100
print(missing_percent.sort_values(ascending=False))

df_value = df_value.drop(columns=['LVH', 'IMT', 'VES', 'IMA_BASE',"ALBURIA_MG_DL"])

violations = df_value[~((df_value['SESSO'] == 0) & (df_value['SESSO0'] == 'DONNA')) & ~((df_value['SESSO'] == 1) & (df_value['SESSO0'] == 'UOMO'))]

print("Rows that do not satisfy the rule:")
print(violations)

import pandas as pd
from pathlib import Path

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

mimic_path = "MIMIC/MIMIC/files/mimiciii/1.4"

tables = load_mimic_dataset(
    mimic_path,
    skip_tables=["chartevents","labevents"]
)

