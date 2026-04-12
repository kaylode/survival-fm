"""survpfn.models.classical — Classical Cox and Kaplan-Meier survival models."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# ---------------------------------------------------------------------------
# Scalability constants (mirror tree.py)
# ---------------------------------------------------------------------------
_MAX_SURV_TIMES = 100     # max time-grid points for the returned surv matrix
_MAX_N_COX      = 50_000  # max training rows for the multivariate Cox fit


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


# ---------------------------------------------------------------------------
# Unified-API Cox wrapper (used by benchmark.py)
# ---------------------------------------------------------------------------

def train_cox(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    random_state: int = 42,
    **kwargs,
) -> tuple:
    """Train a regularised Cox PH model and return the unified survival API tuple.

    Uses ``sksurv.linear_model.CoxPHSurvivalAnalysis`` (scipy BFGS) which is
    5–20× faster than lifelines on large datasets.  For N > ``_MAX_N_COX``
    a stratified random subsample is used for fitting to avoid O(N²) blowup
    in the partial-likelihood gradient computation.

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
        risk_scores  : 1-D array (n_test,), higher = worse prognosis
        surv_probs   : 2-D array (n_test, n_times)
        surv_times   : 1-D array of time points (length ≤ _MAX_SURV_TIMES)
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    T_train = df_train[duration_col].values
    E_train = df_train[event_col].astype(bool).values
    X_train = df_train.drop(columns=[duration_col, event_col]).values.astype(np.float32)
    X_test  = df_test.drop(columns=[duration_col, event_col]).values.astype(np.float32)

    # Build structured array expected by sksurv
    y_train = np.array(
        [(e, t) for e, t in zip(E_train, T_train)],
        dtype=[("event", bool), ("time", float)],
    )

    # Stratified subsample for very large training sets
    if len(X_train) > _MAX_N_COX:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), _MAX_N_COX, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    model = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", max_iter=100)
    try:
        model.fit(X_train, y_train)
    except Exception:
        # Retry with stronger regularisation if convergence fails
        model = CoxPHSurvivalAnalysis(alpha=1.0, ties="efron", max_iter=200)
        model.fit(X_train, y_train)

    risk_scores = model.predict(X_test)

    surv_funcs = model.predict_survival_function(X_test)
    surv_times = model.unique_times_
    if len(surv_times) > _MAX_SURV_TIMES:
        idx = np.linspace(0, len(surv_times) - 1, _MAX_SURV_TIMES, dtype=int)
        surv_times = surv_times[idx]
    surv_probs = np.row_stack([fn(surv_times) for fn in surv_funcs])

    return model, risk_scores, surv_probs, surv_times


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
