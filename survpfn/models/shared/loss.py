"""
survpfn.models.shared.loss — Loss functions for survival models.
"""

import torch
import numpy as np
from survpfn.models.shared.preprocessing import discretize_competing_times


def compute_deephit_cr_loss(out, t, e, td, num_events, num_bins, device, alpha=1.0, beta=1.0):
    # Log-likelihood loss
    batch_size = out.size(0)
    
    # mask1: [batch, events, bins]
    mask1 = torch.zeros(batch_size, num_events, num_bins, device=device)
    for i in range(batch_size):
        if e[i] > 0:
            mask1[i, e[i]-1, td[i]] = 1
            
    # mask2: [batch, bins]
    mask2 = torch.zeros(batch_size, num_bins, device=device)
    for i in range(batch_size):
        mask2[i, td[i]:] = 1
        
    I_1 = (e > 0).float().view(-1, 1)
    
    # Likelihood for events
    tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    loss1_event = I_1 * torch.log(tmp1 + 1e-8)
    
    # Likelihood for censored (probability of surviving past td)
    # Sum of all event probabilities across all times > td
    tmp2 = torch.sum(torch.sum(mask2.unsqueeze(1) * out, dim=2), dim=1, keepdim=True)
    loss1_censored = (1.0 - I_1) * torch.log(1.0 - tmp2 + 1e-8)
    
    loss1 = -torch.mean(loss1_event + loss1_censored)
    
    # Simplified ranking loss (omitted for brevity in this generic wrapper, can be re-added)
    return alpha * loss1

def compute_deephit_cr_loss_v2(out, t, e, td, num_events, num_bins, device, alpha=1.0):
    """
    Corrected DeepHit CR loss (using proper survival probability for censoring).
    """
    batch_size = out.size(0)
    
    # mask1: probability of the specific event at the specific time
    mask1 = torch.zeros(batch_size, num_events, num_bins, device=device)
    for i in range(batch_size):
        if e[i] > 0:
            mask1[i, e[i]-1, td[i]] = 1
            
    # mask2: probability of surviving past censoring time (at or after td)
    mask2 = torch.zeros(batch_size, num_bins, device=device)
    for i in range(batch_size):
        mask2[i, td[i]:] = 1
        
    I_1 = (e > 0).float().view(-1, 1)
    
    # Likelihood for events: P(T=td, E=e)
    tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    loss1_event = I_1 * torch.log(tmp1 + 1e-8)
    
    # Likelihood for censored: P(T >= td) = sum_{k} sum_{t >= td} P(k, t)
    tmp2 = torch.sum(torch.sum(mask2.unsqueeze(1) * out, dim=2), dim=1, keepdim=True)
    loss1_censored = (1.0 - I_1) * torch.log(tmp2 + 1e-8)
    
    loss1 = -torch.mean(loss1_event + loss1_censored)
    return alpha * loss1

def compute_cox_cr_loss(out, t, e, num_events, device):
    """
    Cause-specific Cox loss for Competing Risks (continuous-time partial likelihood).
    Inspired by crisp-nam implementation.
    'out' can be (batch, num_events, num_bins) -> we aggregate bins to get risk scores,
    or ideally 'out' should be (batch, num_events) for this loss.
    """
    if out.dim() == 3:
        # If discrete output, approximate risk score by summing probabilities (or using logits before softmax)
        # Here we just take the sum of probabilities as a proxy for 'risk' if we have to.
        # But better to use this with a model that outputs (batch, num_events).
        risk_scores = torch.sum(out, dim=2) # (batch, num_events)
    else:
        risk_scores = out # (batch, num_events)
        
    # Sort for efficient risk set calculation
    idx = torch.argsort(t, descending=True)
    t_sorted = t[idx]
    e_sorted = e[idx]
    r_sorted = risk_scores[idx]
    
    loss = torch.tensor(0.0, device=device)
    n_events = (e > 0).sum().item()
    if n_events == 0:
        return loss
        
    for k in range(1, num_events + 1):
        event_mask_k = (e_sorted == k)
        if not event_mask_k.any():
            continue
            
        risk_k = r_sorted[:, k-1]
        
        # logsumexp over risk set (all j <= i in sorted list)
        for i in range(len(t_sorted)):
            if event_mask_k[i]:
                log_risk_sum = torch.logsumexp(risk_k[:i+1], dim=0)
                loss += log_risk_sum - risk_k[i]
                
    return loss / max(n_events, 1)
