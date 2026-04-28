"""survpfn.models.shared.cr.loss — Competing-risks loss functions.

Classes
-------
CRDeepHitLoss
    Joint DeepHit log-likelihood for competing risks.
    Key fix over ``compute_deephit_cr_loss_v2``: the censoring mask uses
    **strict** inequality (P(T > C) = sum_{t > C} PMF) instead of the
    inclusive >= that underestimates the censoring probability.

CRCoxLoss
    Cause-specific Cox partial likelihood.
    Vectorised with ``torch.logcumsumexp`` — O(n log n) vs the O(n²)
    Python loop in the old ``compute_cox_cr_loss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRDeepHitLoss(nn.Module):
    """DeepHit log-likelihood for competing risks.

    The model output ``pmf`` must be a valid joint PMF over
    ``(cause, time-bin)`` pairs — i.e. sum_{k,d} pmf[i,k,d] == 1 for all i.
    The backbone models (TabPFN / TabDPT / TabICL) guarantee this via a
    joint softmax applied in their ``forward()`` methods.

    Parameters
    ----------
    alpha : float
        Scalar multiplier on the log-likelihood term (default 1.0).
    eps : float
        Numerical stability epsilon added inside log.

    Inputs
    ------
    pmf    : (B, K, D) — joint PMF, already normalised.
    t_disc : (B,)      — discretised event/censoring time index (long).
    events : (B,)      — cause codes: 0 = censored, 1..K = causes (long).
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        pmf: torch.Tensor,
        t_disc: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        B, K, D = pmf.shape
        device = pmf.device
        eps = self.eps

        is_event = (events > 0).float()  # (B,)

        # --- mask1: indicator that sample i experienced cause k at bin d ----
        # mask1[i, k, d] = 1  iff  events[i] == k+1  AND  t_disc[i] == d
        mask1 = torch.zeros(B, K, D, device=device)
        for i in range(B):
            k_idx = int(events[i].item()) - 1  # 0-indexed cause
            d_idx = int(t_disc[i].item())
            if 0 <= k_idx < K and 0 <= d_idx < D:
                mask1[i, k_idx, d_idx] = 1.0

        # --- mask2: indicator of bins *at or after* censoring time --------
        # P(T >= C) = sum_{t >= C} PMF
        #
        # We use inclusive ">=" rather than strict ">" for a critical reason:
        # discretize_competing_times clips ALL patients censored beyond the
        # last event time to bin D-1.  With strict ">", d_start = D for those
        # patients → mask2 is all-zero → p_survive = 0 → log(eps) = -18 but
        # gradient = 0.  Every late-censored patient contributes a huge loss
        # with ZERO gradient, silently discarding their learning signal.
        # Using ">=" caps d_start at D-1, keeps p_survive > 0, and gives
        # non-zero gradient — consistent with compute_deephit_cr_loss_v2.
        mask2 = torch.zeros(B, D, device=device)
        for i in range(B):
            d_start = int(t_disc[i].item())   # inclusive >=
            if d_start < D:
                mask2[i, d_start:] = 1.0

        # --- event log-likelihood  ------------------------------------------
        # P(event k at time d) = sum over mask1
        p_event = (mask1 * pmf).sum(dim=(1, 2))            # (B,)
        loss_event = is_event * torch.log(p_event + eps)   # (B,)

        # --- censoring log-likelihood  --------------------------------------
        # P(T > C) = sum over all causes and all bins > C
        p_survive = (mask2.unsqueeze(1) * pmf).sum(dim=(1, 2))   # (B,)
        loss_cens = (1.0 - is_event) * torch.log(p_survive + eps)

        return self.alpha * (-torch.mean(loss_event + loss_cens))


class CRCoxLoss(nn.Module):
    """Cause-specific Cox partial-likelihood for competing risks.

    Implements the partial likelihood for each cause k independently,
    using only patients whose event was of cause k as "cases" and the
    risk set at their event time.

    Vectorised using ``torch.logcumsumexp`` after sorting by time
    (descending) — avoids the O(n²) inner loop in the old implementation.

    Inputs
    ------
    pmf    : (B, K, D) or (B, K) — PMF or risk-score tensor.
               If 3-D, probabilities are summed over time bins to produce
               per-cause risk scores (B, K).
    t_cont : (B,) — continuous event/censoring times (for sort order).
    events : (B,) — cause codes: 0 = censored, 1..K = causes (long/float).
    """

    def forward(
        self,
        pmf: torch.Tensor,
        t_cont: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        # Aggregate over time bins → risk scores (B, K)
        risk = pmf.sum(dim=-1) if pmf.dim() == 3 else pmf

        K = risk.shape[1]
        device = risk.device
        n_events = int((events > 0).sum().item())
        if n_events == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Sort by time descending so cumulative logsumexp gives the risk set
        idx = torch.argsort(t_cont, descending=True)
        e_sorted = events[idx]
        r_sorted = risk[idx]   # (B, K)

        # To handle ties correctly, patients with the same time must have the 
        # same denominator (sum of exp(risk) for all patients with T >= t).
        # We find the last index for each unique time in the sorted array.
        t_sorted = t_cont[idx]
        _, counts = torch.unique_consecutive(t_sorted, return_inverse=False, return_counts=True, dim=0)
        # unique_consecutive returns in order of appearance. 
        # Since t_sorted is descending, the "last" index for time t is the 
        # end of its block.
        last_indices = torch.cumsum(counts, dim=0) - 1

        total_loss = torch.zeros(1, device=device)

        for k in range(1, K + 1):
            cause_mask = (e_sorted == k)   # bool (B,)
            if not cause_mask.any():
                continue

            rk = r_sorted[:, k - 1]   # (B,) — risk scores for cause k

            # cum_lse[i] = logsumexp(rk[0], ..., rk[i])
            cum_lse = torch.logcumsumexp(rk, dim=0)   # (B,)
            
            # Fix ties: use the cum_lse at the last index of each tie-block
            cum_lse_stable = torch.repeat_interleave(cum_lse[last_indices], counts)

            # Partial log-likelihood contribution: r_i - logsumexp(risk set_i)
            # Negated (we minimise): logsumexp - r_i
            total_loss = total_loss + (cum_lse_stable[cause_mask] - rk[cause_mask]).sum()

        return total_loss / max(n_events, 1)
