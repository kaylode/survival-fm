import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def run_univariate_cox(df, duration_col, event_col, penalizer=0.1):
    results = []
    
    for c in df.columns: 
        if c not in [duration_col, event_col]:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(
                df[[c, duration_col, event_col]], 
                duration_col=duration_col, 
                event_col=event_col
            )
            coef = cph.summary.loc[c, "coef"]
            se = cph.summary.loc[c, "se(coef)"]
            p = cph.summary.loc[c, "p"]
            results.append({"feature": c, "coef": coef, "se": se, "p": p})

    importance = pd.DataFrame(results)
    importance["abs_logHR"] = importance["coef"].abs()
    
    importance_sig = importance[importance["p"] < 0.05].copy()
    importance_sig["HR"] = np.exp(importance_sig["coef"])
    importance_sig["HR_lower"] = np.exp(importance_sig["coef"] - 1.96 * importance_sig["se"])
    importance_sig["HR_upper"] = np.exp(importance_sig["coef"] + 1.96 * importance_sig["se"])

    # sort by hr
    importance_sig = importance_sig.sort_values("HR")

    # sort by abs_logHR for barplot
    importance_top20 = importance[importance["p"] < 0.05].sort_values("abs_logHR", ascending=False).head(20)

    return importance_sig, importance_top20

def run_multivariate_cox(df, duration_col, event_col, penalizer=0.1):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    
    print("Concordance Index:", cph.concordance_index_)

    importance = cph.summary.reset_index()
    importance.rename(columns={'index':'feature'}, inplace=True)

    importance_sig = importance[importance["p"] < 0.05].copy()
    importance_sig["HR"] = np.exp(importance_sig["coef"])
    importance_sig["HR_lower"] = np.exp(importance_sig["coef"] - 1.96 * importance_sig["se(coef)"])
    importance_sig["HR_upper"] = np.exp(importance_sig["coef"] + 1.96 * importance_sig["se(coef)"])
    importance_sig = importance_sig.sort_values("HR")

    importance["abs_logHR"] = importance["coef"].abs()
    importance_plot = importance[importance["p"] < 0.05].sort_values("abs_logHR", ascending=False)
    
    # fix the name convention so downstream plotting works
    if "covariate" in importance_plot.columns:
        importance_plot["feature"] = importance_plot["covariate"]
    if "covariate" in importance_sig.columns:
        importance_sig["feature"] = importance_sig["covariate"]
        
    return cph, importance_sig, importance_plot

def run_competing_risks_cox(df, event1_val, event2_val, duration_col="Follow Up Data", event_col="death", penalizer=0.1):
    # Model for event 1
    df_temp1 = df.copy()
    df_temp1[event_col] = df_temp1[event_col] == event1_val
    cph1 = CoxPHFitter(penalizer=penalizer)
    cph1.fit(df_temp1, duration_col=duration_col, event_col=event_col)

    # Model for event 2
    df_temp2 = df.copy()
    df_temp2[event_col] = df_temp2[event_col] == event2_val
    cph2 = CoxPHFitter(penalizer=penalizer)
    cph2.fit(df_temp2, duration_col=duration_col, event_col=event_col)

    print(f"Concordance index Event 1 ({event1_val}): ", cph1.concordance_index_)
    print(f"Concordance index Event 2 ({event2_val}): ", cph2.concordance_index_)

    coef1 = cph1.summary["exp(coef)"]
    coef2 = cph2.summary["exp(coef)"]
    p1 = cph1.summary["p"]
    p2 = cph2.summary["p"]

    significant = (p1 < 0.05) | (p2 < 0.05)

    comparison = pd.DataFrame({
        "Event 1 HR": coef1,
        "Event 2 HR": coef2,
        "Event 1 p": p1,
        "Event 2 p": p2
    })
    comparison = comparison[significant]

    return cph1, cph2, comparison
