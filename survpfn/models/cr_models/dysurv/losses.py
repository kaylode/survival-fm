import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

def nll_competing_risks(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Multinomial log-likelihood for discrete-time competing risks.
    
    Args:
        phi: Tensor of shape (batch, num_causes, num_time_bins)
        idx_durations: Tensor of shape (batch,) with time bin indices
        events: Tensor of shape (batch,) with event types (0 for censored, 1..K for causes)
        reduction: 'mean' or 'sum'
        
    Returns:
        Loss tensor
    """
    batch_size, num_causes, num_time_bins = phi.shape
    
    # Add a zero logit for the "no event" category at each time step
    # logits shape: (batch, num_causes + 1, num_time_bins)
    # The first index (0) will correspond to "no event"
    logits = torch.cat([torch.zeros(batch_size, 1, num_time_bins, device=phi.device), phi], dim=1)
    
    # Log-softmax over the causes (+ no event) at each time step
    # shape: (batch, num_causes + 1, num_time_bins)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Probability of failing from cause k at time t: log_probs[:, k, t]
    # Probability of surviving at time t: log_probs[:, 0, t]
    
    events = events.long()
    idx_durations = idx_durations.long()
    
    # Loss for survival up to time t-1
    # sum_{l < t} log P(T > l | T >= l)
    # We use cumsum to get survival probabilities up to each time point
    log_surv = log_probs[:, 0, :].cumsum(dim=1)
    
    # For individuals who fail at time t (event > 0):
    # loss = log P(T=t, K=event) = log P(T=t, K=event | T >= t) + log P(T > t-1)
    
    # Extract log P(T=t, K=event | T >= t)
    # We use gather to pick the correct cause at the correct time
    # events is 1..K, so we use it directly as index in log_probs (where 1..K are the causes)
    event_logits = log_probs.gather(1, events.view(-1, 1, 1).expand(-1, 1, num_time_bins))
    # event_logits shape: (batch, 1, num_time_bins)
    event_log_p_at_t = event_logits.squeeze(1).gather(1, idx_durations.view(-1, 1)).view(-1)
    
    # Extract log P(T > t-1)
    # If t=0, log P(T > -1) = 0
    # Otherwise, it's log_surv[:, t-1]
    shifted_log_surv = F.pad(log_surv[:, :-1], (1, 0), value=0.0)
    log_surv_at_t_minus_1 = shifted_log_surv.gather(1, idx_durations.view(-1, 1)).view(-1)
    
    # For censored individuals (event == 0):
    # loss = log P(T > t) = log_surv[:, t]
    log_surv_at_t = log_surv.gather(1, idx_durations.view(-1, 1)).view(-1)
    
    # Combine losses
    is_event = (events > 0).float()
    loss_event = event_log_p_at_t + log_surv_at_t_minus_1
    loss_censored = log_surv_at_t
    
    loss = -(is_event * loss_event + (1 - is_event) * loss_censored)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class DySurvCRLoss(nn.Module):
    def __init__(self, alphas: list = [1.0, 0.1, 0.1], **kwargs):
        super().__init__()
        self.alphas = alphas

    def forward(self, *args):
        if len(args) == 2:
            outputs, targets = args
        elif len(args) == 6:
            outputs = args[:4]
            targets = args[4:]
        else:
            raise TypeError(f"DySurvCRLoss.forward expects 2 or 6 positional arguments, got {len(args)}")

        decoded, phi, mu, logvar = outputs
        target_surv, target_ae = targets
        
        # target_surv: (idx_durations, events)
        idx_durations, events = target_surv
        
        # Survival Loss
        loss_surv = nll_competing_risks(phi, idx_durations, events)

        # AutoEncoder Loss
        if target_ae.dim() == 2 and decoded.dim() == 3:
            target_ae = target_ae.unsqueeze(1)
        loss_ae = F.mse_loss(decoded, target_ae)

        # KL-divergence Loss
        loss_kd = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return self.alphas[0] * loss_surv + self.alphas[1] * loss_ae + self.alphas[2] * loss_kd
