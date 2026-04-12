"""
survpfn.models.shared.binning — Dataset-adaptive bin-count selection and
hybrid time-discretisation for discrete-time survival models.

Design
------
Three binning schemes are provided:

  quantile   (default / current)
    Bin boundaries = quantiles of observed event times.
    Each bin has ≈ equal number of events.
    Guarantees no empty bins.  Best for small datasets and zero-shot ICL.

  uniform
    Equally-spaced bins across [t_min, t_max] of event times.
    Clinically interpretable, but creates severe label imbalance when
    events are early-skewed.  Useful for calibration reporting.

  hybrid     (recommended for fine-tuned heads)
    Dense quantile bins over the "early" period (high event density),
    coarser uniform bins over the "late" period.
    Best of both worlds: stable supervision + clinical interpretability.

    Parameters
    ----------
    early_event_quantile : float  (default 0.5)
        Quantile of event times used as the early/late split point.
        0.5 = split at the median event time.
    early_bin_frac : float  (default 0.7)
        Fraction of n_bins allocated to the early period.

Adaptive bin count
------------------
``adaptive_num_durations(T, E)`` finds the largest K in a candidate set
such that quantile binning produces ≥ ``min_events_per_bin`` events in
every bin.  This is purely data-driven and requires no tuning.

From the benchmark analysis (xai/bin_analysis.py):

  Dataset          N events    Recommended K
  VETERANS            128         20
  WHAS500             215         20
  SEER               ~600         30
  GBSG               1267        100
  METABRIC           1103        100
  FLCHAIN            1962        100
  SUPPORT2           6036        100
  SIRBU_*          2400-2600     100

Public API
----------
  adaptive_num_durations(T, E, ...)   -> int
  get_cuts(T, E, n_bins, scheme)      -> np.ndarray  (bin edges)
  resolve_num_durations(T, E, req)    -> int          (-1 / 0 → auto)
"""

from __future__ import annotations

import numpy as np
from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Candidate bin counts tried by adaptive selection (ascending order).
CANDIDATE_BINS: list[int] = [10, 20, 30, 50, 100]

#: Default minimum events per bin to accept a bin count.
DEFAULT_MIN_EPB: int = 5


# ---------------------------------------------------------------------------
# Core binning functions
# ---------------------------------------------------------------------------

def quantile_cuts(T_events: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based cut points on observed event times.

    Returns n_bins+1 boundaries (including min and max) after deduplication.
    The actual number of bins may be smaller than ``n_bins`` if many event
    times are identical (ties collapse quantiles).
    """
    T_events = np.asarray(T_events, dtype=float)
    # Add tiny jitter to break ties (only during quantile calculation)
    jitter = np.random.uniform(0, 1e-8, size=T_events.shape)
    probs = np.linspace(0, 100, n_bins + 1)
    cuts = np.unique(np.percentile(T_events + jitter, probs))
    return cuts


def uniform_cuts(T_events: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniformly-spaced cut points over [min(T_events), max(T_events)]."""
    T_events = np.asarray(T_events, dtype=float)
    return np.linspace(T_events.min(), T_events.max(), n_bins + 1)


def hybrid_cuts(
    T_events: np.ndarray,
    n_bins: int,
    early_event_quantile: float = 0.5,
    early_bin_frac: float = 0.7,
) -> np.ndarray:
    """Dense-early / coarse-late hybrid cut points.

    The time axis is split at ``early_event_quantile`` (default: median event
    time).  ``early_bin_frac`` of the total bins are placed in the early
    region using quantile spacing; the remaining bins cover the late region
    with uniform spacing.

    Parameters
    ----------
    T_events : np.ndarray
        Observed (uncensored) event times.
    n_bins : int
        Total number of bins requested.
    early_event_quantile : float
        Quantile of event times used as the early/late split (0 < q < 1).
    early_bin_frac : float
        Fraction of bins to allocate to the early period (0 < f < 1).

    Returns
    -------
    np.ndarray
        Sorted, deduplicated cut-point array of length ≤ n_bins + 1.
    """
    T_events = np.asarray(T_events, dtype=float)

    t_split = float(np.percentile(T_events, early_event_quantile * 100))

    n_early = max(1, int(round(n_bins * early_bin_frac)))
    n_late = max(1, n_bins - n_early)

    early_mask = T_events <= t_split
    late_mask = T_events > t_split

    T_early = T_events[early_mask]
    T_late = T_events[late_mask]

    # Early region: quantile bins (dense, event-balanced)
    if len(T_early) >= n_early:
        probs_early = np.linspace(0, 100, n_early + 1)
        cuts_early = np.unique(np.percentile(T_early, probs_early))
    else:
        cuts_early = np.unique(np.linspace(T_events.min(), t_split, n_early + 1))

    # Late region: uniform bins (coarse, clinically interpretable)
    if len(T_late) >= 2:
        cuts_late = np.linspace(t_split, T_late.max(), n_late + 1)
    else:
        cuts_late = np.array([t_split, T_events.max()])

    cuts = np.unique(np.concatenate([cuts_early, cuts_late]))
    return cuts


# ---------------------------------------------------------------------------
# Adaptive bin count selection
# ---------------------------------------------------------------------------

def adaptive_num_durations(
    T: np.ndarray,
    E: np.ndarray,
    min_events_per_bin: int = DEFAULT_MIN_EPB,
    max_bins: int = 100,
    candidates: list[int] | None = None,
) -> int:
    """Choose the largest bin count where every quantile bin has ≥ min_events_per_bin events.

    This is a data-driven upper bound: it scans ``candidates`` in ascending
    order and returns the largest one that satisfies the sparsity constraint.
    If no candidate satisfies it, the smallest candidate is returned.

    Parameters
    ----------
    T, E : np.ndarray
        Training durations and event indicators (E=0 censored, E>0 event).
    min_events_per_bin : int
        Minimum acceptable events per bin (default 5).
    max_bins : int
        Hard upper limit regardless of data size (default 100).
    candidates : list[int] or None
        Bin counts to try.  Defaults to [10, 20, 30, 50, 100].

    Returns
    -------
    int
        Recommended number of bins.
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E)
    T_events = T[E > 0]

    if candidates is None:
        candidates = CANDIDATE_BINS


    candidates = sorted(k for k in candidates if k <= max_bins)
    if not candidates:
        return 10

    if len(T_events) == 0:
        return candidates[0]

    best_actual = candidates[0]
    for k in candidates:
        cuts = quantile_cuts(T_events, k)
        actual_bins = len(cuts) - 1
        if actual_bins < 1:
            continue
        epb = _events_per_bin(T, E, cuts)
        if epb.min() >= min_events_per_bin and (epb == 0).sum() == 0:
            best_actual = actual_bins  # still satisfies → keep going

    print(f"  Adaptive binning: {len(T_events)} events -> selected {best_actual} bins", flush=True)
    return best_actual


def _events_per_bin(T: np.ndarray, E: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Count events in each interval defined by ``cuts``."""
    n_bins = len(cuts) - 1
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        lo, hi = cuts[k], cuts[k + 1]
        counts[k] = int(((T > lo) & (T <= hi) & (E > 0)).sum())
    return counts


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def get_cuts(
    T: np.ndarray,
    E: np.ndarray,
    n_bins: int,
    scheme: Literal["quantile", "uniform", "hybrid"] = "quantile",
    **hybrid_kw,
) -> np.ndarray:
    """Return bin cut-points using the requested scheme.

    Parameters
    ----------
    T, E : np.ndarray
        Training durations and event indicators.
    n_bins : int
        Number of bins (actual count may be lower after deduplication).
    scheme : {"quantile", "uniform", "hybrid"}
        Discretisation strategy.
    **hybrid_kw
        Forwarded to :func:`hybrid_cuts` (``early_event_quantile``,
        ``early_bin_frac``).

    Returns
    -------
    np.ndarray
        Cut-point boundaries (length ≤ n_bins + 1).
    """
    T_events = np.asarray(T, dtype=float)[np.asarray(E) > 0]
    if len(T_events) == 0:
        T_events = np.asarray(T, dtype=float)

    if scheme == "quantile":
        return quantile_cuts(T_events, n_bins)
    elif scheme == "uniform":
        return uniform_cuts(T_events, n_bins)
    elif scheme == "hybrid":
        return hybrid_cuts(T_events, n_bins, **hybrid_kw)
    else:
        raise ValueError(f"Unknown binning scheme: {scheme!r}. Choose quantile | uniform | hybrid.")


def resolve_num_durations(
    T: np.ndarray,
    E: np.ndarray,
    requested: int,
    min_events_per_bin: int = DEFAULT_MIN_EPB,
    max_bins: int = 100,
) -> int:
    """Return actual bin count, resolving the -1 / 0 "auto" sentinel.

    Parameters
    ----------
    T, E : np.ndarray
        Training durations and event indicators.
    requested : int
        Value from ``FMConfig.num_durations``.  -1 or 0 triggers adaptive
        selection; any positive value is returned as-is.
    min_events_per_bin, max_bins
        Forwarded to :func:`adaptive_num_durations` when auto is requested.

    Returns
    -------
    int
        Resolved number of bins.
    """
    if requested > 0:
        res = requested
    else:
        res = adaptive_num_durations(
            T, E,
            min_events_per_bin=min_events_per_bin,
            max_bins=max_bins,
        )
    print(f"Resolved number of bins: {res}", flush=True)
    return res


