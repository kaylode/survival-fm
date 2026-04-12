"""survpfn.eval.metrics — competing-risks survival model evaluation metrics."""

import warnings
from typing import List, Any

import torch
import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc,
)

from .utils import _safe_time_grid


# --- LOSS FUNCTIONS (from crisp-nam/utils/loss.py) ---

def weighted_negative_log_likelihood_loss(risk_scores, times, events,
                                          num_competing_risks, event_weights=None,
                                          sample_weights=None, eps=1e-8) -> float:
    """
    Computes the weighted negative log-likelihood loss for competing risks Cox model.
    """
    device = times.device
    batch_size = times.shape[0]
    
    loss = torch.tensor(0.0, device=device)
    
    if event_weights is None:
        event_weights = torch.ones(num_competing_risks, device=device)
    if sample_weights is None:
        sample_weights = torch.ones(batch_size, device=device)
    
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss
    
    for k in range(1, num_competing_risks + 1):
        event_mask = (events == k)
        n_events_k = event_mask.sum().item()
        
        if n_events_k == 0:
            continue
            
        risk_k = risk_scores[k-1].squeeze()
        event_weight = event_weights[k-1]
        
        for i in range(batch_size):
            if event_mask[i]:
                risk_set = (times >= times[i])
                risk_set_scores = risk_k[risk_set]
                log_risk_sum = torch.logsumexp(risk_set_scores, dim=0)
                
                individual_loss = log_risk_sum - risk_k[i]
                weighted_individual_loss = individual_loss * event_weight * sample_weights[i]
                loss += weighted_individual_loss
    
    return loss / max(n_events, 1)

def negative_log_likelihood_loss(risk_scores, times, events,
                                 num_competing_risks, eps=1e-8):
    """
    Computes the negative log-likelihood loss for competing risks Cox model.
    """
    device = times.device
    batch_size = times.shape[0]
    
    loss = torch.tensor(0.0, device=device)
    
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss
    
    for k in range(1, num_competing_risks + 1):
        event_mask = (events == k)
        n_events_k = event_mask.sum().item()
        
        if n_events_k == 0:
            continue
            
        risk_k = risk_scores[k-1].squeeze()
        
        for i in range(batch_size):
            if event_mask[i]:
                risk_set = (times >= times[i])
                risk_set_scores = risk_k[risk_set]
                log_risk_sum = torch.logsumexp(risk_set_scores, dim=0)
                loss += log_risk_sum - risk_k[i]
    
    return loss / max(n_events, 1)

def compute_l2_penalty(model, include_bias=False) -> float:
    """
    Compute L2 regularization penalty on model parameters
    """
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not include_bias and 'bias' in name:
                continue
            l2_reg += torch.sum(param ** 2)
    return float(l2_reg)


# --- METRICS & RISK FUNCTIONS (from crisp-nam/utils/risk_cif.py) ---

def compute_baseline_cif(times:np.ndarray,
                         events:np.ndarray,
                         eval_times:List[Any],
                         event_type:int) -> np.ndarray:
    """
    Compute baseline cumulative incidence function for a specific event type
    """
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_events = events[sort_idx]
    
    n_samples = len(times)
    baseline_cif = np.zeros(len(eval_times))
    
    for i, t in enumerate(eval_times):
        cif_t = 0.0
        event_count = np.sum((sorted_events == event_type) & (sorted_times <= t))
        if event_count > 0:
            cif_t = event_count / n_samples
        baseline_cif[i] = cif_t
        
    return baseline_cif

def predict_cif(model:torch.nn.Module,
                x:np.ndarray,
                baseline_cif:np.ndarray,
                times:np.ndarray,
                event_of_interest:int) -> np.ndarray:
    """
    Predict cumulative incidence function for a specific competing risk.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)  # list of length num_risks
        f_j_x = logits[event_of_interest].squeeze(1).cpu().numpy()  # (n_samples,)

    baseline_cif = np.asarray(baseline_cif).reshape(1, -1)  # (1, T)
    risk_scores = np.exp(f_j_x).reshape(-1, 1)               # (N, 1)
    
    cif_pred = 1.0 - np.power(1.0 - baseline_cif, risk_scores)  # shape (N, T)
    
    return cif_pred

def predict_risk(model:torch.nn.Module,
                 x_input:np.ndarray,
                 device:str = 'cpu'):
    """
    Predicts relative risk scores for each competing risk.
    """
    model.eval()
    
    if isinstance(x_input, np.ndarray):
        x_tensor = torch.from_numpy(x_input).float().to(device)
    else:
        x_tensor = x_input.to(device).float()
    
    with torch.no_grad():
        risk_outputs, _ = model(x_tensor)
        risks = torch.cat(risk_outputs, dim=1)

    return risks.cpu().numpy()  

def predict_absolute_risk(model:torch.nn.Module,
                          x_input:np.ndarray,
                          baseline_cifs:List[Any],
                          times:List[Any],
                          device:str = 'cpu') -> np.ndarray:
    """
    Predict absolute risk (CIF) for each competing event by given time points.
    """
    rel_risks = predict_risk(model, x_input, device)
    n_samples, num_events = rel_risks.shape
    n_times = len(times)

    abs_risks = np.zeros((n_samples, num_events, n_times))

    for k in range(num_events):
        base_cif = np.clip(baseline_cifs[k], 1e-10, 0.9999)
        for i in range(n_samples):
            abs_risks[i, k, :] = 1 - np.power(1 - base_cif, np.exp(rel_risks[i, k]))
    
    return abs_risks


def evaluate_cr(
    y_train,
    y_test,
    duration_col: str,
    event_col: str,
    cif_per_cause: list,
    surv_times: np.ndarray,
    n_time_points: int = 50,
    risk_scores = None,
) -> dict:
    """Evaluate a competing-risks model with per-cause and macro-averaged metrics.

    For each cause *k* (1-indexed):

    * **Cause-specific C-index** — concordance where competing events are
      treated as censored.  This is the standard Fine-Gray / subdistribution
      concordance index.
    * **Cause-specific IBS** — IPCW-weighted integrated Brier score for
      CIF_k, using cause-*k* binary event labels and the corresponding KM
      censoring estimator.  Passing ``1 − CIF_k`` as the "survival" estimate
      reproduces the ``riskRegression::Score(..., cause=k)`` approach in R.
    * **Cause-specific time-dependent AUC** — evaluated at the 25th, 50th,
      and 75th percentiles of cause-*k* event times in the test set.

    The headline metrics ``C-index``, ``IBS``, and ``AUC_mean`` are the
    **macro-average** over all causes (i.e. mean over k), which is the
    standard reporting convention for competing-risks benchmarks.

    Parameters
    ----------
    y_train, y_test : DataFrames with ``duration_col`` and ``event_col``.
    cif_per_cause   : list of (n_test, T) float arrays, one per cause.
                      ``cif_per_cause[0]`` = CIF for cause 1,
                      ``cif_per_cause[1]`` = CIF for cause 2, etc.
    surv_times      : (T,) time grid matching the CIF columns.
    n_time_points   : grid resolution for IBS/AUC integration.

    Returns
    -------
    dict with keys ``C-index_cause{k}``, ``IBS_cause{k}``, ``AUC_cause{k}``
    for each k, plus macro-averaged ``C-index``, ``IBS``, ``AUC_mean``.
    """
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    T_train = y_train[duration_col].values.astype(float)
    E_train = y_train[event_col].values.astype(int)
    T_test  = y_test[duration_col].values.astype(float)
    E_test  = y_test[event_col].values.astype(int)

    metrics: dict = {}
    c_indices, ibs_vals, auc_vals = [], [], []

    for k_idx, cif_k in enumerate(cif_per_cause):
        k = k_idx + 1   # 1-indexed cause label

        # ── Cause-k structured arrays (competing events → censored) ──────────
        y_train_k = np.array(
            [(bool(e == k), t) for e, t in zip(E_train, T_train)],
            dtype=[("event", "?"), ("time", "<f8")],
        )
        y_test_k = np.array(
            [(bool(e == k), t) for e, t in zip(E_test, T_test)],
            dtype=[("event", "?"), ("time", "<f8")],
        )

        # ── Risk score for cause k = integrated CIF_k ────────────────────────
        span   = surv_times[-1] - surv_times[0] + 1e-8
        risk_k = _trapz(cif_k, surv_times, axis=1) / span

        # ── Cause-k C-index ───────────────────────────────────────────────────
        try:
            c_k, _, _, _, _ = concordance_index_censored(
                y_test_k["event"], y_test_k["time"], risk_k
            )
            metrics[f"C-index_cause{k}"] = float(c_k)
            c_indices.append(c_k)
        except Exception as e:
            warnings.warn(f"Cause-{k} C-index failed: {e}", stacklevel=2)
            metrics[f"C-index_cause{k}"] = float("nan")

        # ── Cause-k IBS and AUC ───────────────────────────────────────────────
        from .utils import _filter_ipcw_test
        # _filter_ipcw_test returns (y_filtered, cif_filtered, risk_filtered)
        y_test_k_filtered, cif_k_filtered, risk_k_filtered = _filter_ipcw_test(
            y_train_k, y_test_k, cif_k, risk_k
        )

        if len(y_test_k_filtered) == 0:
            metrics[f"IBS_cause{k}"] = float("nan")
            metrics[f"AUC_cause{k}"] = float("nan")
            continue

        grid = _safe_time_grid(y_train_k, y_test_k_filtered, n_points=n_time_points)
        if len(grid) == 0:
            metrics[f"IBS_cause{k}"] = float("nan")
            metrics[f"AUC_cause{k}"] = float("nan")
            continue

        # 1. Cause-k IBS
        try:
            # Interpolate 1 − CIF_k onto the safe grid
            surv_k_grid = np.zeros((len(cif_k_filtered), len(grid)))
            for i in range(len(cif_k_filtered)):
                surv_k_grid[i] = 1.0 - np.interp(grid, surv_times, cif_k_filtered[i])

            ibs_k = integrated_brier_score(y_train_k, y_test_k_filtered, surv_k_grid, grid)
            metrics[f"IBS_cause{k}"] = float(ibs_k)
            ibs_vals.append(ibs_k)
        except Exception as e:
            warnings.warn(f"Cause-{k} IBS failed: {e}", stacklevel=2)
            metrics[f"IBS_cause{k}"] = float("nan")

        # 2. Cause-k AUC
        lo, hi = float(grid[0]), float(grid[-1])
        mask_k = y_test_k_filtered["event"]
        if mask_k.sum() >= 2:
            time_horizons = np.percentile(y_test_k_filtered["time"][mask_k], [25, 50, 75])
            valid_times   = np.array([t for t in time_horizons if lo < t < hi])
            if len(valid_times) > 0:
                try:
                    _, mean_auc_k = cumulative_dynamic_auc(
                        y_train_k, y_test_k_filtered, risk_k_filtered, valid_times
                    )
                    metrics[f"AUC_cause{k}"] = float(mean_auc_k)
                    auc_vals.append(mean_auc_k)
                except Exception as e:
                    warnings.warn(f"Cause-{k} AUC failed: {e}", stacklevel=2)
                    metrics[f"AUC_cause{k}"] = float("nan")
            else:
                metrics[f"AUC_cause{k}"] = float("nan")
        else:
            metrics[f"AUC_cause{k}"] = float("nan")

    # ── Macro-averaged headline metrics ───────────────────────────────────────
    metrics["C-index"] = float(np.nanmean(c_indices)) if c_indices else float("nan")
    metrics["IBS"]     = float(np.nanmean(ibs_vals))  if ibs_vals  else float("nan")
    metrics["AUC_mean"]= float(np.nanmean(auc_vals))  if auc_vals  else float("nan")
    metrics["n_causes"]= len(cif_per_cause)

    return metrics
