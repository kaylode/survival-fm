"""survpfn.models.shared.calibration — Post-hoc survival calibration.

Design principle
----------------
Calibration is **separated from discrimination**.

* Raw survival curves are used for C-index, IBS, and risk stratification.
* Calibrated risks are computed only at specific horizons for calibration
  *reporting* (calibration curves, ECE, horizon Brier).

Three horizons produce poor full-curve calibration — the interpolated
curve distorts rank ordering and integrated metrics.  The correct workflow:

    1. Fit calibrators on a held-out validation split.
    2. For discrimination metrics: use raw ``predict_survival_df`` output.
    3. For calibration reporting: call ``predict_calibrated_horizon_risks``.

Full-curve calibration (``apply_isotonic_survival_calibrators``) is
preserved only when ≥ 9 horizons are fitted; fewer horizons returns the
raw curve unchanged.

Public API
----------
``choose_calibration_horizons``     — 9 quantile horizons by default
``fit_survival_calibrators``        — IPCW-weighted Platt or isotonic
``predict_calibrated_horizon_risks`` — calibrated risks at horizons only
``fit_isotonic_survival_calibrators`` — backward-compat alias
``apply_isotonic_survival_calibrators`` — guarded full-curve reconstruction
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Horizon selection
# ---------------------------------------------------------------------------

def choose_calibration_horizons(
    durations_train: np.ndarray,
    events_train: np.ndarray,
    quantiles: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> np.ndarray:
    """Return time horizons at which to calibrate.

    Defaults to 9 quantiles of the observed event times (Q10–Q90).
    More horizons are required for reliable full-curve reconstruction;
    9 is sufficient for horizon-level calibration reporting.

    Parameters
    ----------
    quantiles : tuple
        Quantiles of event times to use as calibration horizons.
        Pass ``(0.25, 0.50, 0.75)`` to reproduce the old 3-horizon behaviour.
    """
    event_times = np.asarray(durations_train)[np.asarray(events_train) > 0]
    if len(event_times) == 0:
        raise ValueError("No observed events in training data.")
    return np.unique(np.quantile(event_times, quantiles))


# ---------------------------------------------------------------------------
# IPCW helpers
# ---------------------------------------------------------------------------

def _km_censoring_survival(
    durations: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier estimate of the censoring survival function G(t).

    Returns (times_km, G_km) where G_km[i] = P(censoring time > times_km[i]).
    The "event" for the KM is censoring (original event = 0).
    """
    durations = np.asarray(durations, dtype=float)
    c_events = (1 - (np.asarray(events) > 0).astype(int))  # censor indicator

    order = np.argsort(durations)
    T_sorted = durations[order]
    C_sorted = c_events[order]

    n = len(T_sorted)
    G = 1.0
    at_risk = n
    times_km, G_km = [], []

    for i, (t, c) in enumerate(zip(T_sorted, C_sorted)):
        # emit G before processing ties at a new time
        if i > 0 and T_sorted[i] != T_sorted[i - 1]:
            times_km.append(T_sorted[i - 1])
            G_km.append(G)
        if c == 1:
            G *= 1.0 - 1.0 / max(at_risk, 1)
        at_risk -= 1

    times_km.append(T_sorted[-1])
    G_km.append(G)
    return np.array(times_km, dtype=float), np.array(G_km, dtype=float)


def _ipcw_weights(
    durations: np.ndarray,
    events: np.ndarray,
    tau: float,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """IPCW sample weights for horizon *tau*.

    Parameters
    ----------
    durations, events : array-like
    tau : float
        Calibration horizon.
    eps : float
        Minimum G(t) to prevent division by zero.

    Returns
    -------
    observed_mask : bool array, (n,)
        True for patients whose τ-status is observable:
        event occurred by τ  OR  still at risk at τ.
    weights : float array, (n,)
        IPCW weight 1/G(min(T_i, τ)) for observable patients; 0 elsewhere.
    """
    durations = np.asarray(durations, dtype=float)
    is_event = (np.asarray(events) > 0)

    times_km, G_km = _km_censoring_survival(durations, events)

    def G_at(t: float) -> float:
        idx = int(np.searchsorted(times_km, t, side="right")) - 1
        idx = max(0, min(idx, len(G_km) - 1))
        return G_km[idx]

    t_eval = np.minimum(durations, tau)
    G_vals = np.maximum(np.array([G_at(t) for t in t_eval]), eps)

    observed_mask = (is_event & (durations <= tau)) | (durations > tau)
    weights = np.where(observed_mask, 1.0 / G_vals, 0.0)
    return observed_mask, weights


# ---------------------------------------------------------------------------
# Fitting calibrators
# ---------------------------------------------------------------------------

# Internal storage format: Dict[float, Tuple[str, fitted_obj]]
# where str ∈ {"platt", "isotonic"}

def fit_survival_calibrators(
    surv_val: pd.DataFrame,
    durations_val: np.ndarray,
    events_val: np.ndarray,
    horizons: np.ndarray,
    method: str = "platt",
    use_ipcw: bool = True,
) -> Dict[float, Tuple[str, object]]:
    """Fit per-horizon calibrators on a validation set.

    Parameters
    ----------
    surv_val : pd.DataFrame
        Survival probabilities on the validation set (rows=times, cols=patients).
    durations_val, events_val : array-like
        Validation durations and event indicators.
    horizons : array-like
        Time horizons at which to fit a calibrator.
    method : {"platt", "isotonic"}
        ``"platt"`` (default) — logistic regression on logit-transformed predicted
        risk.  Smoother than isotonic; less prone to overfitting on small
        validation sets.
        ``"isotonic"`` — isotonic regression; can overfit, use with ≥ 200 val
        samples per horizon.
    use_ipcw : bool
        If True, weight samples by 1/G(min(T, τ)) using the KM censoring
        survival.  Reduces calibration bias under heavy or informative censoring.

    Returns
    -------
    dict : float → ("platt" | "isotonic", fitted_object)
    """
    if method not in {"platt", "isotonic"}:
        raise ValueError(f"Unknown calibration method {method!r}. Use 'platt' or 'isotonic'.")

    durations_val = np.asarray(durations_val, dtype=float)
    events_val = np.asarray(events_val)
    horizons = np.asarray(horizons, dtype=float)
    is_event = (events_val > 0)

    calibrators: Dict[float, Tuple[str, object]] = {}

    for tau in horizons:
        # Predicted risk at tau from raw survival curve
        idx = int(np.searchsorted(surv_val.index.values, tau, side="left"))
        idx = min(idx, len(surv_val.index) - 1)
        pred_risk = 1.0 - surv_val.iloc[idx].values.astype(float)

        # Binary label: did event occur by tau?
        y_tau = (is_event & (durations_val <= tau)).astype(float)

        if use_ipcw:
            observed_mask, weights = _ipcw_weights(durations_val, events_val, tau)
        else:
            observed_mask = (is_event & (durations_val <= tau)) | (durations_val > tau)
            weights = None

        x_cal = pred_risk[observed_mask]
        y_cal = y_tau[observed_mask]
        w_cal = weights[observed_mask] if weights is not None else None

        if len(np.unique(y_cal)) < 2:
            continue  # need both classes to fit

        if method == "platt":
            eps = 1e-6
            x_logit = logit(np.clip(x_cal, eps, 1 - eps)).reshape(-1, 1)
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(x_logit, y_cal, sample_weight=w_cal)
            calibrators[float(tau)] = ("platt", cal)

        else:  # isotonic
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(x_cal, y_cal, sample_weight=w_cal)
            calibrators[float(tau)] = ("isotonic", iso)

    return calibrators


# ---------------------------------------------------------------------------
# Applying calibrators
# ---------------------------------------------------------------------------

def predict_calibrated_horizon_risks(
    surv_df: pd.DataFrame,
    calibrators: Dict[float, Tuple[str, object]],
) -> pd.DataFrame:
    """Return calibrated event risks at each calibration horizon.

    **This is the preferred function for calibration reporting.**  It returns
    risks at the calibrated horizons *only* — it does NOT reconstruct or modify
    the full survival curve, so C-index and IBS computed from raw curves are
    unaffected.

    Parameters
    ----------
    surv_df : pd.DataFrame
        Raw survival probabilities (rows=times, cols=patients).
    calibrators : dict
        Output of ``fit_survival_calibrators`` or ``fit_isotonic_survival_calibrators``.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_horizons, n_patients)``.  Index = calibration time horizons.
        Values = calibrated P(event by τ | x) ∈ [0, 1].
    """
    if not calibrators:
        return pd.DataFrame()

    times = surv_df.index.values.astype(float)
    eps = 1e-6
    rows: Dict[float, np.ndarray] = {}

    for tau, cal_obj in sorted(calibrators.items()):
        idx = int(np.searchsorted(times, tau, side="left"))
        idx = min(idx, len(times) - 1)
        pred_risk = (1.0 - surv_df.iloc[idx].values).astype(float)

        method, cal = cal_obj
        if method == "platt":
            x_logit = logit(np.clip(pred_risk, eps, 1 - eps)).reshape(-1, 1)
            risk_cal = cal.predict_proba(x_logit)[:, 1]
        elif method == "isotonic":
            risk_cal = cal.predict(pred_risk)
        else:
            risk_cal = pred_risk  # no-op fallback

        rows[tau] = np.clip(risk_cal, 0.0, 1.0)

    # DataFrame: index=horizons, columns=patients
    return pd.DataFrame(rows, index=surv_df.columns).T


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def fit_isotonic_survival_calibrators(
    surv_val: pd.DataFrame,
    durations_val: np.ndarray,
    events_val: np.ndarray,
    horizons: np.ndarray,
    use_ipcw: bool = True,
) -> Dict[float, Tuple[str, object]]:
    """Backward-compatible alias for ``fit_survival_calibrators(..., method='isotonic')``.

    Prefer ``fit_survival_calibrators(..., method='platt')`` for small validation
    sets where isotonic tends to overfit.
    """
    return fit_survival_calibrators(
        surv_val, durations_val, events_val, horizons,
        method="isotonic", use_ipcw=use_ipcw,
    )


def apply_isotonic_survival_calibrators(
    surv_df: pd.DataFrame,
    calibrators: Dict[float, object],
) -> pd.DataFrame:
    """Full-curve calibration via interpolation over calibrated horizons.

    .. warning::
        Full-curve calibration from few horizons distorts rank ordering,
        C-index, and IBS.  This function returns the *raw* curve unchanged
        when fewer than 9 calibration horizons are fitted.

        **For calibration reporting**, use ``predict_calibrated_horizon_risks``
        instead — it returns horizon-level risks without touching the curve.

    With ≥ 9 horizons, performs linear interpolation of calibrated risks back
    onto the full time grid (monotonicity enforced).
    """
    n_horizons = len(calibrators)

    if n_horizons < 9:
        warnings.warn(
            f"apply_isotonic_survival_calibrators: only {n_horizons} horizon(s) "
            f"fitted — full-curve reconstruction from so few horizons distorts "
            f"C-index and IBS.  Returning raw survival curve unchanged.\n"
            f"Use predict_calibrated_horizon_risks() for horizon-level calibration "
            f"reporting, and reserve raw curves for discrimination metrics.",
            UserWarning,
            stacklevel=2,
        )
        return surv_df

    # ≥ 9 horizons: safe to interpolate back to full grid.
    times = surv_df.index.values.astype(float)
    R = 1.0 - surv_df.values.copy()  # (n_times, n_patients)

    horizon_risks = predict_calibrated_horizon_risks(surv_df, calibrators)
    cal_horizons = horizon_risks.index.values.astype(float)
    R_cal_points = horizon_risks.values  # (n_horizons, n_patients)

    # Boundary: t=t_0 → risk=0
    h_ext = np.concatenate([[times[0]], cal_horizons])
    r_ext = np.vstack([np.zeros((1, R.shape[1])), R_cal_points])

    R_cal = np.zeros_like(R)
    for j in range(R.shape[1]):
        R_cal[:, j] = np.interp(times, h_ext, r_ext[:, j])

    # Enforce valid cumulative risk (non-decreasing) and survival (non-increasing)
    R_cal = np.maximum.accumulate(np.clip(R_cal, 0.0, 1.0), axis=0)
    S_cal = np.minimum.accumulate(np.clip(1.0 - R_cal, 0.0, 1.0), axis=0)
    return pd.DataFrame(S_cal, index=surv_df.index, columns=surv_df.columns)
