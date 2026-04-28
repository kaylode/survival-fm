import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from survpfn.dataloaders.competing import SurvivalDatasetDeepHit

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

def create_masks_gpu(e, t_disc, num_Event, num_Category, device):
    """
    Creates masks for DeepHit loss:
    - mask1: [batch, num_Event, num_Category] - Indicator for observed event and time.
    - mask2: [batch, num_Event, num_Category] - Indicator for all events at or after censoring time.
    - mask3: [batch, num_Category] - Indicator for times at or before event time (for ranking).
    """
    batch_size = e.size(0)
    mask1 = torch.zeros(batch_size, num_Event, num_Category, device=device)
    mask2 = torch.zeros(batch_size, num_Event, num_Category, device=device)
    mask3 = torch.zeros(batch_size, num_Category, device=device)
    
    for i in range(batch_size):
        ti = int(t_disc[i].item())
        if e[i] > 0:
            ki = int(e[i].item()) - 1
            mask1[i, ki, ti] = 1
        
        # For likelihood of censored data: P(T >= t_cens)
        # Sum over all causes k and all times tau >= ti
        mask2[i, :, ti:] = 1
        
        # For ranking loss: F_k(t_i) = P(K=k, T <= t_i)
        # Prefix sum up to ti
        mask3[i, :ti+1] = 1
        
    return mask1, mask2, mask3

class DeepHitCRPattern(nn.Module):
    def __init__(self, x_dim, num_Event, num_Category,
                 h_dim_shared=128, h_dim_CS=64, num_layers_shared=2, num_layers_CS=2,
                 dropout_rate=0.4):
        super().__init__()
        self.num_Event = num_Event
        self.num_Category = num_Category
        self.h_dim_CS = h_dim_CS
        self.dropout_rate = dropout_rate
        
        # Shared subnetwork
        layers_shared = []
        prev_dim = x_dim
        for _ in range(num_layers_shared):
            layers_shared.extend([
                nn.Linear(prev_dim, h_dim_shared),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim_shared),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim_shared
        self.shared_net = nn.Sequential(*layers_shared)
        
        # Cause-specific subnetworks
        self.cs_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim_shared, h_dim_CS),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim_CS),
                nn.Dropout(dropout_rate),
                nn.Linear(h_dim_CS, h_dim_CS),
                nn.ReLU()
            ) for _ in range(num_Event)
        ])
        
        # Final output layer
        self.output_layer = nn.Linear(num_Event * h_dim_CS, num_Event * num_Category)
        
        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        shared_out = self.shared_net(x)
        
        cs_outputs = [net(shared_out) for net in self.cs_nets]
        stacked_out = torch.stack(cs_outputs, dim=1) # [B, K, h_dim_CS]
        reshaped_out = stacked_out.view(x.size(0), -1) # [B, K * h_dim_CS]
        
        logits = self.output_layer(reshaped_out)
        # Joint PMF over all events and categories
        out = F.softmax(logits.view(-1, self.num_Event * self.num_Category), dim=1)
        out = out.view(-1, self.num_Event, self.num_Category)
        return out, None

    def log_likelihood_loss(self, out, k, mask1, mask2):
        # Contribution for observed events
        I_1 = (k > 0).float().view(-1, 1)
        # P(K=k_obs, T=t_obs)
        tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + 1e-8)
        
        # Contribution for censored data: P(T >= t_cens)
        # mask2 sums over all k and tau >= t_cens
        tmp2 = torch.sum(torch.sum(mask2 * out, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + 1e-8)
        
        return -torch.mean(tmp1 + tmp2)

    def ranking_loss(self, out, t, k, mask_pref):
        """
        Ranking loss term: Penalizes pairs (i, j) where i had event e at t_i,
        t_i < t_j, but F_e(t_i | x_i) <= F_e(t_i | x_j).
        """
        sigma1 = 0.1
        eta = []
        one_vector = torch.ones_like(t, dtype=torch.float32)
        
        for e in range(self.num_Event):
            # Indicator that subject i had event e+1
            I_2 = (k == (e + 1)).float()
            I_2_diag = torch.diag(I_2.squeeze())
            
            # CIF for cause e: F_e(t | x) = sum_{tau <= t} P(K=e, T=tau | x)
            # We need matrix C where C_ij = F_e(t_i | x_j)
            # A = out[:, e, :] (B x L)
            # M = mask_pref (B x L) where M_it = 1 if t <= t_i
            # C_ij = sum_t M_it * A_jt = (M * A^T)_ij
            C = torch.matmul(mask_pref, out[:, e, :].transpose(0, 1))
            
            # diag_C[i] = F_e(t_i | x_i)
            diag_C = torch.diag(C).unsqueeze(0)
            
            # R_ij = F_e(t_i | x_i) - F_e(t_i | x_j)
            R = diag_C.transpose(0, 1) - C
            
            # Time difference matrix T: T_ij = 1 if t_i < t_j
            T = F.relu(torch.sign(torch.matmul(one_vector, t.transpose(0, 1)) - 
                                  torch.matmul(t, one_vector.transpose(0, 1))))
            # T_ij = 1 if i had event e AND t_i < t_j
            T = torch.matmul(I_2_diag, T)
            
            # Loss for this event
            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)
            
        eta = torch.stack(eta, dim=1)
        # Average over events and samples
        return torch.mean(eta)

    def calibration_loss(self, out, k, mask_pref):
        """
        MSE Calibration loss on CIF (optional term often used in DeepHit).
        """
        loss_cal = 0.0
        for e in range(self.num_Event):
            I_2 = (k == (e + 1)).float().squeeze()
            # F_e(t_i | x_i)
            cif_ti = torch.sum(out[:, e, :] * mask_pref, dim=1)
            loss_cal += torch.mean((cif_ti - I_2)**2)
        return loss_cal / self.num_Event

    def compute_loss(self, out, t, k, mask1, mask2, mask_pref, alpha=1.0, beta=1.0, gamma=0.1):
        loss1 = self.log_likelihood_loss(out, k, mask1, mask2)
        loss2 = self.ranking_loss(out, t, k, mask_pref)
        loss3 = self.calibration_loss(out, k, mask_pref)
        
        return alpha * loss1 + beta * loss2 + gamma * loss3

def _predict_absolute_risk_deephit(model, x_test, cuts, times, device="cpu"):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds, _ = model(x_test_tensor)
    preds_np = preds.cpu().numpy()
    
    n_samples, n_events, n_categories = preds_np.shape
    n_times = len(times)
    abs_risks = np.zeros((n_samples, n_events, n_times))
    
    # Pre-calculate CIF (cumulative sum over categories)
    cif = np.cumsum(preds_np, axis=2) # [B, K, L]
    
    for t_idx, time_val in enumerate(times):
        # Find which bin this time_val belongs to
        # cuts[i] is the right edge of bin i
        bin_idx = np.searchsorted(cuts, time_val)
        
        if bin_idx == 0:
            # Before the first bin
            if time_val <= 0:
                abs_risks[:, :, t_idx] = 0
            else:
                # Interpolate between 0 and cuts[0]
                frac = time_val / (cuts[0] + 1e-8)
                abs_risks[:, :, t_idx] = frac * cif[:, :, 0]
        elif bin_idx >= n_categories:
            # After the last bin
            abs_risks[:, :, t_idx] = cif[:, :, -1]
        else:
            # Interpolate between bin_idx-1 and bin_idx
            t_prev = cuts[bin_idx - 1]
            t_curr = cuts[bin_idx]
            frac = (time_val - t_prev) / (t_curr - t_prev + 1e-8)
            val_prev = cif[:, :, bin_idx - 1]
            val_curr = cif[:, :, bin_idx]
            abs_risks[:, :, t_idx] = val_prev + frac * (val_curr - val_prev)
            
    return [abs_risks[:, k, :] for k in range(n_events)]

def _train_deephit_cr_once(
    X_train, T_train, E_train, X_test,
    epochs, batch_size, lr, device, num_durations,
):
    """Train one DeepHit-CR model and return (model, cif_per_cause, grid)."""
    t_max = float(np.max(T_train))
    num_Event = int(np.max(E_train))

    dataset = SurvivalDatasetDeepHit(X_train, T_train, E_train, num_durations)
    cuts = dataset.cuts # Important for mapping time to bins
    current_bs = batch_size
    model = None
    
    while True:
        try:
            loader = DataLoader(dataset, batch_size=current_bs, shuffle=True)
            model = DeepHitCRPattern(
                x_dim=X_train.shape[1], num_Event=num_Event,
                num_Category=len(cuts), # Use actual number of bins from cuts
            ).to(device)
            # Use weight decay for L2 regularization
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            
            for epoch in range(epochs):
                model.train()
                for x, t, e, t_disc in loader:
                    x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
                    mask1, mask2, mask_pref = create_masks_gpu(e, t_disc, num_Event, len(cuts), device)
                    
                    out, _ = model(x)
                    loss = model.compute_loss(out, t.view(-1, 1), e.view(-1, 1), mask1, mask2, mask_pref)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            break
        except RuntimeError as err:
            if "CUDA out of memory" in str(err) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"      {C_YELLOW}\u2192 CUDA OOM (DeepHitCR)! Reducing batch_size to {current_bs}{C_RESET}")
            else:
                raise

    grid = np.linspace(float(np.min(T_train)) + 1e-6, t_max - 1e-6, 50)
    cif_per_cause = _predict_absolute_risk_deephit(model, X_test, cuts, grid, device)
    return model, cif_per_cause, grid


def train_deephit_cr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    num_durations: int = 50, # Increased default durations for better resolution
    **kwargs
) -> tuple:
    feats = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train = df_train[feats].values.astype(np.float32)
    T_train = df_train[duration_col].values.astype(np.float32)
    E_train = df_train[event_col].values.astype(np.int64)
    X_test  = df_test[feats].values.astype(np.float32)

    # ── Optional Optuna HPO ───────────────────────────────────────────────────
    if tune and n_trials > 1:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                _lr   = trial.suggest_float("lr",            1e-4, 5e-3, log=True)
                _bs   = trial.suggest_categorical("batch_size", [64, 128, 256])
                _ep   = trial.suggest_int("epochs",           50,  150, step=25)
                _nd   = trial.suggest_int("num_durations",    30,  100, step=10)
                _,  cif_trial, grid_trial = _train_deephit_cr_once(
                    X_train, T_train, E_train, X_test,
                    epochs=_ep, batch_size=_bs, lr=_lr, device=device,
                    num_durations=_nd,
                )
                # Proxy objective: mean CIF variance across test samples (discrimination proxy)
                cif0 = cif_trial[0]
                return float(np.mean(np.std(cif0, axis=0)))

            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            epochs        = best.get("epochs",        epochs)
            batch_size    = best.get("batch_size",    batch_size)
            lr            = best.get("lr",            lr)
            num_durations = best.get("num_durations", num_durations)
            print(f"      {C_YELLOW}[DeepHitCR] Best HPO: {best}{C_RESET}")
        except Exception as hpo_err:
            print(f"      {C_YELLOW}[DeepHitCR] HPO failed ({hpo_err}), using defaults{C_RESET}")

    # ── Final training ────────────────────────────────────────────────────────
    model, cif_per_cause, grid = _train_deephit_cr_once(
        X_train, T_train, E_train, X_test,
        epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, num_durations=num_durations,
    )

    # Final risks for all causes
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    span = grid[-1] - grid[0] + 1e-8
    risks = [_trapz(cif, grid, axis=1) / span for cif in cif_per_cause]

    return model, risks, cif_per_cause, grid
