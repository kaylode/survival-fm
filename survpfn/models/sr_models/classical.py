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
    # Drop constant columns (common source of Matrix is singular)
    non_const = [c for c in df_train.columns 
                 if c in {duration_col, event_col} or df_train[c].nunique() > 1]
    df_train_sub = df_train[non_const].copy()
    df_test_sub = df_test[[c for c in df_test.columns if c in non_const]].copy()

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df_train_sub, duration_col=duration_col, event_col=event_col)
    except Exception:
        # Retry with a larger penalizer if it fails
        cph = CoxPHFitter(penalizer=max(penalizer * 10, 1.0))
        cph.fit(df_train_sub, duration_col=duration_col, event_col=event_col)

    c_index = cph.score(df_test_sub, scoring_method="concordance_index")
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
