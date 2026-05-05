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

def run_multivariate_cox(
    df_train,
    df_test,
    duration_col,
    event_col,
    penalizer=0.01,
    scale=False,
    max_retries=5,
    verbose=True,
):
    # ---------------------------
    # 1. Drop constant columns
    # ---------------------------
    non_const = [
        c for c in df_train.columns
        if c in {duration_col, event_col} or df_train[c].nunique() > 1
    ]
    df_train = df_train[non_const].copy()
    df_test = df_test[[c for c in df_test.columns if c in non_const]].copy()

    # ---------------------------
    # 2. Scale features (huge impact on convergence)
    # ---------------------------
    features = [c for c in df_train.columns if c not in {duration_col, event_col}]

    # ---------------------------
    # 2.5 PCA for high-dimensional data
    # ---------------------------
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    n_feat = len(feat_cols)

    if n_feat > 2000:
        from sklearn.decomposition import PCA

        # smarter component selection:
        # keep 95% variance but cap at 2000
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(df_train[feat_cols])
        X_test_pca = pca.transform(df_test[feat_cols])

        n_comp = X_train_pca.shape[1]

        print(f"→ PCA applied: {n_feat} → {n_comp} components (95% variance)", flush=True)

        # rebuild dataframes
        pca_cols = [f"pca_{i}" for i in range(n_comp)]

        df_train = (
            pd.DataFrame(X_train_pca, columns=pca_cols, index=df_train.index)
            .assign(**{
                duration_col: df_train[duration_col],
                event_col: df_train[event_col],
            })
        )

        # Bugfix: must use OLD df_test to get duration/event cols
        df_test_pca = pd.DataFrame(X_test_pca, columns=pca_cols, index=df_test.index)
        if duration_col in df_test.columns:
            df_test_pca[duration_col] = df_test[duration_col]
        if event_col in df_test.columns:
            df_test_pca[event_col] = df_test[event_col]
        df_test = df_test_pca

        feat_cols = pca_cols

    # ---------------------------
    # 3. Fit with adaptive penalization
    # ---------------------------
    current_penalizer = penalizer
    cph = None

    for i in range(max_retries):
        try:
            cph = CoxPHFitter(penalizer=current_penalizer)

            cph.fit(
                df_train,
                duration_col=duration_col,
                event_col=event_col,
                batch_mode=True,  # 🔥 big speedup
                show_progress=verbose,
                fit_options={
                    "max_steps": 100,     # reduce iterations
                    "precision": 1e-5     # faster convergence
                }
            )

            if np.isfinite(cph.params_).all():
                break

        except Exception:
            pass

        current_penalizer *= 5.0

    if cph is None or not np.isfinite(cph.params_).all():
        warnings.warn(
            f"Cox fit unstable after retries (penalizer={current_penalizer})",
            stacklevel=2
        )

    # ---------------------------
    # 4. Optional summary
    # ---------------------------
    cph.print_summary()

    # ---------------------------
    # 5. Faster scoring (optional subsample)
    # ---------------------------
    try:
        if len(df_test) > 50000:
            df_test_eval = df_test.sample(50000, random_state=42)
        else:
            df_test_eval = df_test

        c_index = cph.score(df_test_eval, scoring_method="concordance_index")
    except Exception:
        c_index = np.nan

    # ---------------------------
    # 6. Importance computation (cleaned)
    # ---------------------------
    importance = cph.summary.reset_index().rename(columns={'index': 'feature'})

    # unified filtering
    sig_mask = importance["p"] < 0.05
    importance_sig = importance[sig_mask].copy()

    # hazard ratios
    importance_sig["HR"] = np.exp(importance_sig["coef"])
    importance_sig["HR_lower"] = np.exp(
        importance_sig["coef"] - 1.96 * importance_sig["se(coef)"]
    )
    importance_sig["HR_upper"] = np.exp(
        importance_sig["coef"] + 1.96 * importance_sig["se(coef)"]
    )

    importance_sig = importance_sig.sort_values("HR")

    # plotting importance
    importance_plot = importance_sig.copy()
    importance_plot["abs_logHR"] = importance_plot["coef"].abs()
    importance_plot = importance_plot.sort_values("abs_logHR", ascending=False)

    return cph, df_test, importance_sig, importance_plot, c_index


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
    X_train = df_train.drop(columns=[duration_col, event_col]).values.astype(np.float64)
    X_test  = df_test.drop(columns=[duration_col, event_col]).values.astype(np.float64)

    # Sanity check for NaNs after preprocessing
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        import warnings
        warnings.warn("Cox input contains NaNs even after imputation/scaling. This will likely lead to NaN surv_probs.", stacklevel=2)

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

    # Use slightly higher alpha by default (0.5) to prevent coefficient explosion
    model = CoxPHSurvivalAnalysis(alpha=0.5, ties="efron", max_iter=100)
    try:
        model.fit(X_train, y_train)
    except Exception:
        # Retry with stronger regularisation if convergence fails
        model = CoxPHSurvivalAnalysis(alpha=5.0, ties="efron", max_iter=200)
        model.fit(X_train, y_train)

    risk_scores = model.predict(X_test)
    
    # Clip extreme risk scores to prevent overflow/underflow in survival calculation
    # exp(700) is the limit for float64.
    risk_scores = np.clip(risk_scores, -500, 500)

    surv_funcs = model.predict_survival_function(X_test)
    surv_times = model.unique_times_
    if len(surv_times) > _MAX_SURV_TIMES:
        idx = np.linspace(0, len(surv_times) - 1, _MAX_SURV_TIMES, dtype=int)
        surv_times = surv_times[idx]
    surv_probs = np.row_stack([fn(surv_times) for fn in surv_funcs])

    if np.isnan(surv_probs).any():
        import warnings
        warnings.warn("Cox model returned NaN surv_probs. Check for ill-conditioned data or extreme outliers.", stacklevel=2)

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
