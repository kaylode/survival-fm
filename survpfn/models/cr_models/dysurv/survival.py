import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

from .model import DySurvCR
from .losses import DySurvCRLoss
from survpfn.dataloaders.competing import SurvivalDatasetDeepHit

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

def train_dysurv_cr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    num_durations: int = 20,
    encoded_features: int = 16,
    device: str = "cpu",
    **kwargs
) -> tuple:
    """
    Train DySurvCR and return benchmark-compatible outputs.
    """
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train = df_train[feat_cols].values.astype(np.float32)
    T_train = df_train[duration_col].values.astype(np.float32)
    E_train = df_train[event_col].values.astype(np.int64)
    X_test  = df_test[feat_cols].values.astype(np.float32)
    
    num_causes = int(np.max(E_train))
    
    # Internal train/val split
    X_tr, X_val, T_tr, T_val, E_tr, E_val = train_test_split(
        X_train, T_train, E_train, test_size=val_fraction, random_state=42, stratify=(E_train > 0)
    )
    
    train_ds = SurvivalDatasetDeepHit(X_tr, T_tr, E_tr, num_durations=num_durations)
    val_ds   = SurvivalDatasetDeepHit(X_val, T_val, E_val, num_durations=num_durations)
    
    # Use cuts from train dataset
    cuts = train_ds.cuts
    
    # We need to manually discretize val_ds using train_ds.cuts if we want consistency
    # But SurvivalDatasetDeepHit calculates its own bins. 
    # Let's override it for consistency.
    def discretize_with_cuts(durations, cuts):
        t_disc = np.digitize(durations, cuts, right=True)
        t_disc = np.clip(t_disc, 0, len(cuts) - 1)
        return torch.tensor(t_disc, dtype=torch.long)
    
    val_ds.t_disc = discretize_with_cuts(T_val, cuts)
    val_ds.cuts = cuts
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = DySurvCR(
        in_features=len(feat_cols),
        encoded_features=encoded_features,
        out_features=len(cuts),
        num_causes=num_causes
    ).to(device)
    
    criterion = DySurvCRLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, t, e, t_disc in train_loader:
            x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
            optimizer.zero_grad()
            
            # DySurvCR forward returns (decoded, phi, mu, logvar)
            outputs = model(x)
            
            # criterion expects (outputs, targets)
            # targets: ((idx_durations, events), target_ae)
            targets = ((t_disc, e), x)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, t, e, t_disc in val_loader:
                x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
                outputs = model(x)
                targets = ((t_disc, e), x)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state:
        model.load_state_dict(best_state)
    
    # Prediction on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        # phi: (batch, num_causes, num_time_bins)
        phi = model.predict(X_test_tensor)
        
        # Calculate CIF per cause
        # P(K=k, T=t | T >= t) = exp(phi_k(t)) / (1 + sum_j exp(phi_j(t)))
        # We use softmax over (0, phi_1, ..., phi_K)
        batch_size, num_causes, num_bins = phi.shape
        logits = torch.cat([torch.zeros(batch_size, 1, num_bins, device=device), phi], dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        hazard_causes = probs[:, 1:, :] # (batch, num_causes, num_bins)
        hazard_no_event = probs[:, 0, :] # (batch, num_bins)
        
        # S(t) = product_{l <= t} P(T > l | T >= l)
        surv = hazard_no_event.cumprod(dim=1) # (batch, num_bins)
        
        # P(T=t, K=k) = hazard_k(t) * S(t-1)
        surv_shifted = F.pad(surv[:, :-1], (1, 0), value=1.0)
        pdf_causes = hazard_causes * surv_shifted.unsqueeze(1) # (batch, num_causes, num_bins)
        
        # CIF_k(t) = sum_{l <= t} P(T=l, K=k)
        cif_causes = pdf_causes.cumsum(dim=2).cpu().numpy()
        
    # Interpolate CIFs to a fine grid for benchmarking
    grid = np.linspace(0, cuts.max(), 100)
    cif_per_cause = []
    times = np.concatenate([[0], cuts])
    
    for k in range(num_causes):
        cif_k = cif_causes[:, k, :]
        # Add CIF=0 at time=0
        cif_k_ext = np.concatenate([np.zeros((len(X_test), 1)), cif_k], axis=1)
        
        f = interp1d(times, cif_k_ext, kind='linear', axis=1, fill_value="extrapolate")
        cif_interp = np.clip(f(grid), 0.0, 1.0)
        cif_per_cause.append(cif_interp)
        
    # Benchmark expects: (model, risk_scores, cif_per_cause, grid)
    # risk_scores for cause 1 (or average risk)
    risk_cause1 = cif_per_cause[0][:, -1] # Risk at max time
    
    return model, risk_cause1, cif_per_cause, grid

import torch.nn.functional as F
