"""
survpfn.models.shared.preprocessing — Feature and label preprocessing for FM models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from survpfn.models.shared.binning import get_cuts, resolve_num_durations


# ---------------------------------------------------------------------------
# Time-binning helpers
# ---------------------------------------------------------------------------

def discretize_competing_times(durations, events, n_bins):
    """Discretize times for competing risks (quantile-based on any event)."""
    event_times = durations[events > 0]
    if len(event_times) == 0:
        event_times = durations
    # Use linspace on event times to ensure n_bins intervals
    max_t = event_times.max()
    bins = np.linspace(0, max_t, n_bins + 1)[1:]
    t_disc = np.digitize(durations, bins, right=True)
    t_disc = np.clip(t_disc, 0, n_bins - 1)
    return bins, t_disc


# ---------------------------------------------------------------------------
# SurvivalTimeBinEncoder
# ---------------------------------------------------------------------------

class SurvivalTimeBinEncoder:
    """Compute KM-derived time-bin features from training survival data.

    The 6-dim feature vector per bin encodes complementary views of the
    population-level survival landscape:

    Idx  Name                  Interpretation
    ---  ----                  --------------
     0   t_k / T_max           Continuous time position (0 -> 1)
     1   k / (K-1)             Discrete bin index (positional)
     2   KM_survival(t_k)      How many patients typically survive this long
     3   KM_discrete_hazard    Local instantaneous risk at t_k
     4   event_density_k       Fraction of all events concentrated in this bin
     5   at_risk_fraction_k    n_at_risk(t_k) / n_total (estimate reliability)

    Parameters
    ----------
    n_bins : int
        Number of time bins (quantile-based bin boundaries on event times).
    """

    N_TIME_FEATURES: int = 6

    def __init__(self, n_bins: int = -1, scheme:str="quantile", **hybrid_kw) -> None:
        self.n_bins = n_bins
        self._bin_times: Optional[np.ndarray] = None  # (K,) quantile cuts
        self._bin_feats: Optional[np.ndarray] = None  # (K, 6) raw features
        self.scheme = scheme
        self.hybrid_kw = hybrid_kw

    def fit(self, T: np.ndarray, E: np.ndarray) -> "SurvivalTimeBinEncoder":
        """Compute bin boundaries and KM statistics from training data.
        Ensures t=0 is included as the first bin coordinate.
        """
        T = np.asarray(T, dtype=float)
        E = (np.asarray(E) > 0).astype(int)

        n_bins = resolve_num_durations(T, E, self.n_bins)
        cuts = get_cuts(T, E, n_bins, scheme=self.scheme, **self.hybrid_kw)
        self._bin_times = np.unique(np.concatenate([[0.0], cuts])).astype(float)
        
        K = len(self._bin_times)
        T_max = float(T.max()) + 1e-8
        n_total = len(T)
        n_events = max(int(E.sum()), 1)

        # Kaplan-Meier survival S(t_k)
        km_S = np.ones(K, dtype=float)
        # S(0) should be 1.0 (already initialized)
        S_prev = 1.0
        for i, t_k in enumerate(self._bin_times):
            if t_k == 0:
                continue
            n_risk = int((T >= t_k).sum())
            d_k = int(((T == t_k) & (E == 1)).sum())
            S_new = S_prev * (1.0 - d_k / n_risk) if n_risk > 0 and d_k > 0 else S_prev
            km_S[i] = S_new
            S_prev = S_new

        # Discrete hazard h_k = 1 - S(t_k)/S(t_{k-1})
        km_h = np.zeros(K, dtype=float)
        for i in range(K):
            S_prev_i = km_S[i - 1] if i > 0 else 1.0
            km_h[i] = 1.0 - km_S[i] / S_prev_i if S_prev_i > 1e-12 else 0.0

        # Event density: fraction of all events in each bin
        event_density = np.zeros(K, dtype=float)
        for i in range(K):
            if i == 0:
                event_density[i] = 0.0 # No events at exactly t=0 (usually)
                continue
            lo = self._bin_times[i-1]
            hi = self._bin_times[i]
            cnt = ((T > lo) & (T <= hi) & (E == 1)).sum()
            event_density[i] = cnt / n_events

        # At-risk fraction n_risk(t_k) / n_total
        at_risk_frac = np.array(
            [(T >= t_k).sum() / n_total for t_k in self._bin_times],
            dtype=float,
        )

        # Assemble (K, 6) feature matrix
        self._bin_feats = np.stack([
            self._bin_times / T_max,                  # 0: normalized time
            np.arange(K) / max(K - 1, 1),             # 1: bin index
            km_S,                                      # 2: KM survival
            km_h,                                      # 3: discrete hazard
            event_density,                             # 4: event density
            at_risk_frac,                              # 5: at-risk fraction
        ], axis=1).astype(np.float32)

        return self

    @property
    def bin_times(self) -> np.ndarray:
        if self._bin_times is None:
            raise RuntimeError("Call fit() first.")
        return self._bin_times.copy()

    @property
    def bin_feats(self) -> np.ndarray:
        if self._bin_feats is None:
            raise RuntimeError("Call fit() first.")
        return self._bin_feats.copy()

    def get_labels_and_masks(
        self, T: np.ndarray, E: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin binary labels and validity masks for masked BCE training."""
        T = np.asarray(T, dtype=float)
        E = (np.asarray(E) > 0).astype(int)
        K = len(self._bin_times)
        n = len(T)

        labels = np.zeros((n, K), dtype=np.float32)
        masks  = np.zeros((n, K), dtype=bool)

        for k, t_k in enumerate(self._bin_times):
            mask_k  = (E == 1) | (T > t_k)
            label_k = (E == 1) & (T <= t_k)
            masks[:, k]  = mask_k
            labels[:, k] = label_k.astype(np.float32)

        return labels, masks


# ---------------------------------------------------------------------------
# Data Expansion
# ---------------------------------------------------------------------------

def expand_survival_data(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    bin_times: np.ndarray,
    bin_feats: np.ndarray,
    sampling_ratio: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised temporal expansion: N patients × K bins -> flat binary dataset.
    
    Valid rows: event patient OR survived past this bin.

    Optional sampling_ratio: if provided, keeps all y=1 rows and samples 
    sampling_ratio * n_pos y=0 rows to reduce dataset size and handle imbalance.
    """
    N, D = X.shape
    K = len(bin_times)

    # Repeat: patient-major order (p0b0, p0b1, ... p0bK, p1b0, ...)
    X_rep = np.repeat(X, K, axis=0)
    T_rep = np.repeat(T, K)
    E_rep = np.repeat(E, K)
    feat_rep = np.tile(bin_feats, (N, 1))
    t_rep = np.tile(bin_times, N)

    mask = (E_rep == 1) | (T_rep > t_rep)

    X_filt = X_rep[mask]
    feat_filt = feat_rep[mask]
    T_filt = T_rep[mask]
    t_filt = t_rep[mask]
    E_filt = E_rep[mask]

    X_exp = np.concatenate([X_filt, feat_filt], axis=1)
    y_exp = np.where(T_filt <= t_filt, E_filt, 0).astype(np.int64)

    # ── Downsampling Logic ──
    if sampling_ratio is not None:
        pos_idx = np.where(y_exp == 1)[0]
        neg_idx = np.where(y_exp == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg_wanted = int(n_pos * sampling_ratio)
        
        if n_neg_wanted < len(neg_idx):
            rng = np.random.default_rng(random_state)
            sampled_neg_idx = rng.choice(neg_idx, size=n_neg_wanted, replace=False)
            
            # Combine and sort to maintain some semblance of order (optional but avoids total chaos)
            keep_idx = np.concatenate([pos_idx, sampled_neg_idx])
            keep_idx.sort()
            
            X_exp = X_exp[keep_idx]
            y_exp = y_exp[keep_idx]

    return X_exp, y_exp


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

class FMDataPrep:
    """Unified feature preprocessing for all FM joint/embedding models."""

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.pca_V: Optional[torch.Tensor] = None

    def fit_transform(
        self,
        x: np.ndarray,
        max_features: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit scaler (and optionally PCA) on ``x``, return scaled features."""
        x_scaled = self.scaler.fit_transform(x).astype(np.float32)
        x_pt = torch.from_numpy(x_scaled) 

        if max_features is not None and x.shape[1] > max_features:
            _, _, V = torch.pca_lowrank(x_pt, q=max_features)
            self.pca_V = V
        else:
            self.pca_V = None

        if device is not None:
            # Re-apply transform to get the projection result as a tensor on device
            return self.transform(x, device=device)
        return x_scaled

    def transform(self, x: np.ndarray, device: Optional[str] = None) -> Union[np.ndarray, torch.Tensor]:
        """Apply fitted scaler and optional PCA."""
        x_scaled = self.scaler.transform(x).astype(np.float32)
        x_pt = torch.from_numpy(x_scaled)
        if self.pca_V is not None:
            x_pt = x_pt @ self.pca_V
        
        if device is not None:
            return x_pt.to(device)
        return x_pt.numpy()


# ---------------------------------------------------------------------------
# Label / target preprocessing
# ---------------------------------------------------------------------------

@dataclass
class TargetTensors:
    """Container for all target tensors needed by the training loop."""
    dur_disc: torch.Tensor          # discretised / labtrans-transformed duration (for loss)
    dur_cont: torch.Tensor          # original continuous duration (for CR ranking loss)
    events: torch.Tensor            # event indicator (float for Cox, long for discrete)
    interval_frac: Optional[torch.Tensor]  # only for PCHazard
    bin_times: Optional[np.ndarray]       # CR bin edges (stored for CIF prediction)


class FMTargetPrep:
    """Unified target preprocessing for survival and competing risks models.
    
    Ensures that binning, label transforms, and discretization are fitted 
    on training data and applied consistently to validation/test data.
    """
    def __init__(
        self,
        task_type: str,
        head_type: str,
        num_durations: int,
        labtrans = None,
    ) -> None:
        self.task_type = task_type
        self.head_type = head_type
        self.num_durations = num_durations
        self.labtrans = labtrans
        self.bin_times: Optional[np.ndarray] = None
        self._fitted = False

    def fit_transform(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        device: str = "cpu"
    ) -> Tuple[TargetTensors, int]:
        """Fit preprocessing on durations/events and return targets."""
        self._fitted = True
        return self._prepare(durations, events, device=device, fit=True)

    def transform(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        device: str = "cpu"
    ) -> TargetTensors:
        """Apply fitted preprocessing to new durations/events."""
        if not self._fitted:
            raise RuntimeError("FMTargetPrep must be fitted before calling transform().")
        tgt, _ = self._prepare(durations, events, device=device, fit=False)
        return tgt

    def _prepare(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        device: str,
        fit: bool = False
    ) -> Tuple[TargetTensors, int]:
        """Internal helper for fit/transform."""
        if self.task_type == "cr":
            if fit:
                self.bin_times, t_disc = discretize_competing_times(durations, events, self.num_durations)
            else:
                # Use fitted bins
                t_disc = np.digitize(durations, self.bin_times, right=True)
                t_disc = np.clip(t_disc, 0, len(self.bin_times) - 1)
            
            effective_n_dur = len(self.bin_times)
            return TargetTensors(
                dur_disc=torch.from_numpy(t_disc).to(device, torch.long),
                dur_cont=torch.from_numpy(durations.copy()).to(device, torch.float32),
                events=torch.from_numpy(events.copy()).to(device, torch.long),
                interval_frac=None,
                bin_times=self.bin_times,
            ), effective_n_dur

        elif self.labtrans is not None:
            if fit:
                # Add tiny jitter to break ties (only for fitting labtrans)
                durations_jit = durations + np.random.uniform(0, 1e-7, size=durations.shape)
                targets = self.labtrans.fit_transform(durations_jit, events)
            else:
                targets = self.labtrans.transform(durations, events)
                
            frac = (
                torch.from_numpy(targets[2]).float().to(device)
                if len(targets) > 2 else None
            )
            dur_disc = torch.from_numpy(targets[0]).to(device)
            ev = torch.from_numpy(targets[1]).to(device)
            
            return TargetTensors(
                dur_disc=dur_disc,
                dur_cont=torch.from_numpy(durations.copy()).float().to(device),
                events=ev,
                interval_frac=frac,
                bin_times=None,
            ), self.labtrans.out_features

        else:
            # Cox: continuous targets (no fitting needed)
            dur = torch.from_numpy(durations.copy()).float().to(device)
            ev = torch.from_numpy(events.copy()).float().to(device)
            return TargetTensors(
                dur_disc=dur,
                dur_cont=dur,
                events=ev,
                interval_frac=None,
                bin_times=None,
            ), self.num_durations


def prepare_targets(
    durations: np.ndarray,
    events: np.ndarray,
    *,
    task_type: str,
    labtrans,
    num_durations: int,
    device: str,
    **kwargs
) -> Tuple[TargetTensors, int]:
    """Legacy wrapper for backward compatibility. Note: This still fits every time."""
    prep = FMTargetPrep(task_type, kwargs.get("head_type", "deephit"), num_durations, labtrans)
    return prep.fit_transform(durations, events, device=device)
