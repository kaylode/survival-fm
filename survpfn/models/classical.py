"""survpfn.models.classical — Classical Cox and Kaplan-Meier survival models."""

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


def run_multivariate_cox(df_train, df_test, duration_col, event_col, penalizer=0.1):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_train, duration_col=duration_col, event_col=event_col)

    c_index = cph.score(df_test, scoring_method="concordance_index")
    print("Concordance Index (test):", c_index)

    importance = cph.summary.reset_index()
    importance.rename(columns={'index': 'feature'}, inplace=True)

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

    return cph, importance_sig, importance_plot, c_index


def run_competing_risks_cox(df_train, df_test, event1_val, event2_val, duration_col="Follow Up Data", event_col="death", penalizer=0.1):
    # Model for event 1
    df_temp1_train = df_train.copy()
    df_temp1_train[event_col] = df_temp1_train[event_col] == event1_val
    df_temp1_test = df_test.copy()
    df_temp1_test[event_col] = df_temp1_test[event_col] == event1_val
    cph1 = CoxPHFitter(penalizer=penalizer)
    cph1.fit(df_temp1_train, duration_col=duration_col, event_col=event_col)

    # Model for event 2
    df_temp2_train = df_train.copy()
    df_temp2_train[event_col] = df_temp2_train[event_col] == event2_val
    df_temp2_test = df_test.copy()
    df_temp2_test[event_col] = df_temp2_test[event_col] == event2_val
    cph2 = CoxPHFitter(penalizer=penalizer)
    cph2.fit(df_temp2_train, duration_col=duration_col, event_col=event_col)

    c_index1 = cph1.score(df_temp1_test, scoring_method="concordance_index")
    c_index2 = cph2.score(df_temp2_test, scoring_method="concordance_index")

    print(f"Concordance index Event 1 ({event1_val}) test: ", c_index1)
    print(f"Concordance index Event 2 ({event2_val}) test: ", c_index2)

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

    return cph1, cph2, comparison, c_index1, c_index2


from sksurv.nonparametric import kaplan_meier_estimator


def run_kaplan_meier(df_test, duration_col, event_col):
    """
    Computes the Kaplan-Meier estimator for the test set.
    Returns:
        time: unique time points
        survival_prob: survival probability at those time points
    """
    time, survival_prob = kaplan_meier_estimator(
        df_test[event_col].astype(bool),
        df_test[duration_col]
    )
    # Convert to a format consistent with model.predict_surv_df (a DataFrame)
    # KM is a population-level estimate, but we return it as a single "risk" profile
    km_df = pd.DataFrame(survival_prob, index=time, columns=["KP_Population"])
    return time, survival_prob, km_df
