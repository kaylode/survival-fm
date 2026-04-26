"""survpfn.eval.metrics — single-risk survival model evaluation metrics."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc
)
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os

from .utils import _get_survival_array, _safe_time_grid

# ── Color codes ──────────────────────────────────────────────────────────────
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[96m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def d_calibration(
    events: np.ndarray,
    times: np.ndarray,
    surv_probs: np.ndarray,
    surv_times: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute D-calibration (distribution calibration) for survival models.

    D-calibration measures whether the predicted survival probability at the
    event time is uniformly distributed over [0, 1] for uncensored subjects.
    A well-calibrated model produces a uniform histogram; the returned value
    is the mean squared deviation from uniformity (lower is better).

    Algorithm (Haider et al., 2020)
    --------------------------------
    For each uncensored subject i:
    1. Interpolate S_i(t_i) — the predicted survival probability at the
       observed event time.
    2. Collect all S_i(t_i) values.
    3. Check that the resulting distribution is approximately Uniform(0, 1)
       by computing the chi-squared statistic against a uniform histogram.

    Parameters
    ----------
    events:
        Boolean array of event indicators.
    times:
        Time-to-event or censoring times (same length as events).
    surv_probs:
        Predicted survival matrix (n_samples, n_surv_times).
    surv_times:
        Time points for the survival matrix columns.
    n_bins:
        Number of histogram bins for the uniformity test.

    Returns
    -------
    float
        Mean squared deviation from the uniform density (normalised by bin
        count).  0 = perfectly calibrated.
    """
    uncensored_mask = events.astype(bool)
    if uncensored_mask.sum() < 2:
        return float("nan")

    event_times = times[uncensored_mask]
    event_surv = surv_probs[uncensored_mask]

    # Predicted survival at each subject's own event time
    s_at_t = np.array([
        float(np.interp(t, surv_times, s))
        for t, s in zip(event_times, event_surv)
    ])

    # Histogram over [0, 1]
    counts, _ = np.histogram(s_at_t, bins=n_bins, range=(0.0, 1.0))
    expected = len(s_at_t) / n_bins
    deviation = float(np.mean((counts - expected) ** 2) / (expected ** 2))
    return deviation


def evaluate_sr(
    y_train,
    y_test,
    duration_col,
    event_col,
    risk_scores,
    surv_probs=None,
    surv_times=None,
    time_horizons=None,
    n_time_points: int = 50,
):
    """
    Evaluate single risk survival metrics.
    Args:
        y_train: Training labels df (contains duration and event).
        y_test: Test labels df.
        duration_col: Name of time column.
        event_col: Name of event column.
        risk_scores: Array of risk scores (e.g. hazard from Cox or -surv from others).
        surv_probs: Matrix of survival probabilities shape (n_samples, n_times).
        surv_times: Array of times corresponding to columns of surv_probs.
        time_horizons: Specific times to evaluate time-dependent AUC and IBS.
        n_time_points: Number of evenly spaced points for the IBS integral.
    Returns:
        Dictionary of metrics {"C-index": ..., "IBS": ..., "AUC_mean": ..., "D-cal": ...}
    """
    y_train_sksurv = _get_survival_array(y_train, duration_col, event_col)
    y_test_sksurv = _get_survival_array(y_test, duration_col, event_col)

    metrics = {}

    # 1. Concordance Index
    # 1a. Harrell's C-index
    try:
        c_index, _, _, _, _ = concordance_index_censored(
            y_test_sksurv["event"], y_test_sksurv["time"], risk_scores
        )
        metrics["C-index"] = c_index
    except Exception as e:
        warnings.warn(f"C-index computation failed: {e}", stacklevel=2)
        metrics["C-index"] = np.nan

    # 1b. Time-dependent concordance index (Antolini)
    if surv_probs is not None and surv_times is not None:
        try:
            from pycox.evaluation import EvalSurv
            surv_df = pd.DataFrame(surv_probs.T, index=surv_times)
            eval_pycox = EvalSurv(surv_df, y_test_sksurv["time"], y_test_sksurv["event"])
            metrics["C_td"] = eval_pycox.concordance_td('antolini')
        except Exception as e:
            warnings.warn(f"Time-dependent C-index failed: {e}", stacklevel=2)
            metrics["C_td"] = np.nan


    # Filter test subjects for IPCW bounds
    from .utils import _filter_ipcw_test
    y_test_ipcw, _all_outs = _filter_ipcw_test(y_train_sksurv, y_test_sksurv, risk_scores, surv_probs)
    y_test_filtered = y_test_ipcw
    risk_scores_filtered = _all_outs[0]
    surv_probs_filtered = _all_outs[1]

    if len(y_test_filtered) == 0:
        return metrics

    # Only create grid and horizons after filtering
    grid = _safe_time_grid(y_train_sksurv, y_test_filtered, n_points=n_time_points)
    if len(grid) == 0:
        return metrics

    if time_horizons is None:
        mask = y_test_filtered["event"]
        if sum(mask) > 0:
            time_horizons = np.percentile(y_test_filtered["time"][mask], [25, 50, 75])
        else:
            time_horizons = grid[[len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4]]

    lo, hi = float(grid[0]), float(grid[-1])
    valid_times = np.array([t for t in time_horizons if lo < t < hi])

    # 2. Integrated Brier Score
    # try:
    surv_at_grid = np.zeros((surv_probs_filtered.shape[0], len(grid)))
    for i in range(surv_probs_filtered.shape[0]):
        surv_at_grid[i, :] = np.interp(grid, surv_times, surv_probs_filtered[i, :])

    ibs = integrated_brier_score(
        y_train_sksurv, y_test_filtered, surv_at_grid, grid
    )
    metrics["IBS"] = ibs

    if len(valid_times) > 0:
        try:
            # 3. Time-dependent AUC
            auc, mean_auc = cumulative_dynamic_auc(
                y_train_sksurv, y_test_filtered, risk_scores_filtered, valid_times
            )
            metrics["AUC_mean"] = mean_auc
            for t, a in zip(valid_times, auc):
                metrics[f"AUC_t={t:.1f}"] = a
        except Exception as e:
            warnings.warn(f"Time-dependent AUC computation failed: {e}", stacklevel=2)
            metrics["AUC_mean"] = np.nan

    # 4. D-calibration (reliability)
    try:
        metrics["D-cal"] = d_calibration(
            y_test_sksurv["event"], y_test_sksurv["time"],
            surv_probs, surv_times, n_bins=10
        )
    except Exception as e:
        warnings.warn(f"D-calibration computation failed: {e}", stacklevel=2)
        metrics["D-cal"] = float("nan")

    return metrics


def brier_score_per_time(
    y_train,
    y_test,
    duration_col: str,
    event_col: str,
    surv_probs: np.ndarray,
    surv_times: np.ndarray,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time-point Brier score (not integrated).

    Returns
    -------
    times : np.ndarray, shape (n_points,)
    brier : np.ndarray, shape (n_points,)
    """
    from sksurv.metrics import brier_score

    y_train_sk = _get_survival_array(y_train, duration_col, event_col)
    y_test_sk  = _get_survival_array(y_test,  duration_col, event_col)

    grid = _safe_time_grid(y_train_sk, y_test_sk, n_points=n_points)
    if len(grid) == 0:
        return np.array([]), np.array([])

    surv_at_grid = np.zeros((surv_probs.shape[0], len(grid)))
    for i in range(surv_probs.shape[0]):
        surv_at_grid[i] = np.interp(grid, surv_times, surv_probs[i])

    try:
        _, bs = brier_score(y_train_sk, y_test_sk, surv_at_grid, grid)
    except Exception:
        bs = np.full(len(grid), np.nan)

    return grid, bs


from SurvivalEVAL.Evaluator import SurvivalEvaluator
from pycox.evaluation import EvalSurv

def evaluate_sr_survival_eval(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    surv_probs: np.ndarray,
    surv_times: np.ndarray,
    n_time_points: int = 100,
    output_dir: Optional[str] = None,
    risk_scores = None,
) -> dict:
    """
    Evaluate single risk survival metrics using the SurvivalEVAL package.
    Provides Concordance, IBS, D-Calibration p-value, and MAE (Margin/Pseudo-obs).
    """
    T_train = y_train[duration_col].values.astype(float)
    E_train = y_train[event_col].values.astype(int)
    T_test  = y_test[duration_col].values.astype(float)
    E_test  = y_test[event_col].values.astype(int)

    # Initialise SurvivalEvaluator
    try:
        evaluator = SurvivalEvaluator(
            pred_survs=surv_probs,
            time_coordinates=surv_times,
            event_times=T_test,
            event_indicators=E_test,
            train_event_times=T_train,
            train_event_indicators=E_train
        )
        
        # 1. Survival Curves Plotting
        if output_dir:
            try:
                sample_indices = [0, min(10, len(T_test)-1), min(100, len(T_test)-1)]
                fig, ax = evaluator.plot_survival_curves(sample_indices)
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, "survival_curves.pdf"))
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Survival curve plotting failed: {e}")

        metrics = {}
        # 2. Concordance (Harrell's C-index)
        c_index, _, _ = evaluator.concordance()

        # Time-dependent concordance index (Antolini)
        try:
            surv_df = pd.DataFrame(surv_probs.T, index=surv_times)
            eval_pycox = EvalSurv(surv_df, T_test, E_test)
            metrics["C_td"] = eval_pycox.concordance_td('antolini')
        except Exception as e:
            warnings.warn(f"Time-dependent concordance index failed: {e}")
            metrics["C_td"] = np.nan

        # If nan, compute with sksurv
        if np.isnan(c_index):
            y_train_sksurv = _get_survival_array(y_train, duration_col, event_col)
            y_test_sksurv = _get_survival_array(y_test, duration_col, event_col)
            c_index, _, _, _, _ = concordance_index_censored(
                y_test_sksurv["event"], y_test_sksurv["time"], risk_scores
        )
        metrics["C-index"] = c_index

        # 3. Integrated Brier Score
        # Draw figure only when we have a directory to save it to (BUG-M6).
        # Always close any returned figure immediately after saving to prevent
        # matplotlib figure accumulation across folds.
        draw_ibs_fig = output_dir is not None
        ibs_res = evaluator.integrated_brier_score(num_points=n_time_points, draw_figure=draw_ibs_fig)
        if isinstance(ibs_res, tuple):
            ibs, fig_data = ibs_res
            # fig_data may be (fig, ax) or just a Figure — handle both
            fig_obj = fig_data[0] if isinstance(fig_data, (tuple, list)) else fig_data
            if output_dir:
                try:
                    fig_obj.savefig(os.path.join(output_dir, "ibs_figure.pdf"))
                except Exception as _e:
                    warnings.warn(f"IBS figure save failed: {_e}")
            plt.close(fig_obj)   # always close regardless of whether we saved
        else:
            ibs = ibs_res
        metrics["IBS"] = ibs

        # 4. AUC and Brier Score at Percentiles (q25, q50, q75)
        try:
            mask = E_test.astype(bool)
            if mask.sum() > 0:
                quantiles = [25, 50, 75]
                time_horizons = np.percentile(T_test[mask], quantiles)
                
                aucs = []
                briers = []
                td_cis = []
                
                for q, t in zip(quantiles, time_horizons):
                    # 1. SurvivalEVAL.auc(target_time)
                    auc_val = evaluator.auc(target_time=t)
                    metrics[f"D-AUC_q{q}"] = auc_val
                    aucs.append(auc_val)

                    # 2. SurvivalEVAL.brier_score(target_time)
                    bs_val = evaluator.brier_score(target_time=t)
                    metrics[f"BS_q{q}"] = bs_val
                    briers.append(bs_val)
                    
                    # 3. Truncated C-index at horizon t
                    # We treat events after t as censored at t
                    T_trunc = np.minimum(T_test, t)
                    E_trunc = E_test * (T_test <= t)
                    # For sksurv C-index, we need a risk score. 
                    # We approximate risk score at time t as 1 - S(t)
                    risk_t = 1.0 - np.array([np.interp(t, surv_times, s) for s in surv_probs])
                    try:
                        ci_t, _, _, _, _ = concordance_index_censored(E_trunc.astype(bool), T_trunc, risk_t)
                        metrics[f"TD-CI_q{q}"] = ci_t
                        td_cis.append(ci_t)
                    except:
                        metrics[f"TD-CI_q{q}"] = np.nan

                    # 4. ECE at horizon t  (pd.qcut equal-frequency binning)
                    # Patients censored before τ have unknown status → exclude them.
                    is_event_flag = E_test.astype(bool)
                    y_tau     = (is_event_flag & (T_test <= t)).astype(float)
                    observed  = (is_event_flag & (T_test <= t)) | (T_test > t)

                    pred_obs = risk_t[observed]
                    y_obs    = y_tau[observed]

                    if len(y_obs) >= 10 and y_obs.sum() > 0:
                        try:
                            _df = pd.DataFrame({"pred": pred_obs, "y": y_obs})
                            _df["bin"] = pd.qcut(_df["pred"], q=10, duplicates="drop")
                            _grp = _df.groupby("bin", observed=True).agg(
                                mean_pred=("pred", "mean"),
                                obs_rate=("y", "mean"),
                                n=("y", "size"),
                            )
                            _n = _grp["n"].sum()
                            ece_val = float(
                                np.sum(_grp["n"] / _n * np.abs(_grp["obs_rate"] - _grp["mean_pred"]))
                            ) if _n > 0 else np.nan
                            metrics[f"ECE_q{q}"] = ece_val
                        except Exception:
                            metrics[f"ECE_q{q}"] = np.nan
                    else:
                        metrics[f"ECE_q{q}"] = np.nan

                metrics["AUC_mean"] = np.nanmean(aucs) if aucs else np.nan
                metrics["ECE_mean"] = np.nanmean([
                    metrics.get(f"ECE_q{q}", np.nan) for q in quantiles
                ])
            else:
                metrics["AUC_mean"] = np.nan
        except Exception as e:
            warnings.warn(f"Time-dependent metrics at quantiles failed: {e}")
            metrics["AUC_mean"] = np.nan

        # 5. D-Calibration (returns p-value and binary details)
        try:
            p_val, dcal_details = evaluator.d_calibration(return_details=True)
            metrics["D-cal_pval"] = p_val
        except Exception:
            metrics["D-cal_pval"] = np.nan

        # 6. MAE (Mean Absolute Error)
        try:
            metrics["MAE-Margin"] = evaluator.mae(method='Margin', weighted=True)
            metrics["MAE-PO"] = evaluator.mae(method='Pseudo_obs', weighted=True)
        except Exception:
            metrics["MAE-Margin"] = np.nan
            metrics["MAE-PO"] = np.nan

        return metrics
        
    except Exception as e:
        print(f"      {C_YELLOW}→ FAILED  {e}{C_RESET}")
        breakpoint()
        return {"C-index": np.nan, "IBS": np.nan}
    finally:
        # Ensure all figures are closed to prevent memory leaks
        plt.close('all')