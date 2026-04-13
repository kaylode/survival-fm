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

class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), batch_norm=False, dropout_rate=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = activation
    
    def forward(self, x):
        x = self.fc(x)
        if self.batch_norm: x = self.batch_norm(x)
        if self.activation: x = self.activation(x)
        if self.dropout: x = self.dropout(x)
        return x

class FCNet(nn.Module):
    def __init__(self, in_dim, num_layers, h_dim, activation=nn.ReLU(), dropout_rate=0.0):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for i in range(num_layers):
            layers.append(FCLayer(prev_dim, h_dim, activation=activation, dropout_rate=dropout_rate))
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def create_fc_mask1_gpu(e, t_disc, num_Event, num_Category, device):
    batch_size = e.size(0)
    mask1 = torch.zeros(batch_size, num_Event, num_Category, device=device)
    for i in range(batch_size):
        if e[i] > 0:
            event_idx = int(e[i].item()) - 1
            t_idx = int(t_disc[i].item())
            mask1[i, event_idx, t_idx] = 1
    return mask1

def create_fc_mask2_gpu(t_disc, num_Category, device):
    batch_size = t_disc.size(0)
    mask2 = torch.zeros(batch_size, num_Category, device=device)
    for i in range(batch_size):
        t_idx = int(t_disc[i].item())
        mask2[i, t_idx:] = 1
    return mask2

class DeepHitCRPattern(nn.Module):
    def __init__(self, x_dim, num_Event, num_Category,
                 h_dim_shared=128, h_dim_CS=64, num_layers_shared=1, num_layers_CS=2,
                 dropout_rate=0.1):
        super().__init__()
        self.num_Event = num_Event
        self.num_Category = num_Category
        self.h_dim_CS = h_dim_CS
        self.dropout_rate = dropout_rate
        
        self.shared_net = FCNet(x_dim, num_layers_shared, h_dim_shared, activation=nn.ReLU(), dropout_rate=dropout_rate)
        self.cs_nets = nn.ModuleList([
            FCNet(x_dim + h_dim_shared, num_layers_CS, h_dim_CS, activation=nn.ReLU(), dropout_rate=dropout_rate)
            for _ in range(num_Event)
        ])
        self.output_layer = nn.Linear(num_Event * h_dim_CS, num_Event * num_Category)

    def forward(self, x):
        shared_out = self.shared_net(x)
        h = torch.cat([x, shared_out], dim=1)
        
        cs_outputs = [net(h) for net in self.cs_nets]
        stacked_out = torch.stack(cs_outputs, dim=1)
        reshaped_out = stacked_out.view(-1, self.num_Event * self.h_dim_CS)
        
        logits = self.output_layer(F.dropout(reshaped_out, self.dropout_rate, self.training))
        out = F.softmax(logits.view(-1, self.num_Event * self.num_Category), dim=1)
        out = out.view(-1, self.num_Event, self.num_Category)
        return out, None

    def log_likelihood_loss(self, out, t, k, mask1, mask2):
        if not isinstance(k, torch.Tensor): k = torch.tensor(k, device=out.device)
        I_1 = (k > 0).float().view(-1, 1)
        tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + 1e-8)
        tmp2 = torch.sum(torch.sum(mask2.unsqueeze(1) * out, dim=2), dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + 1e-8)
        return -torch.mean(tmp1 + tmp2)

    def ranking_loss(self, out, t, k, mask2):
        sigma1 = 0.1
        eta = []
        if not isinstance(k, torch.Tensor): k = torch.tensor(k, device=out.device)
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=out.device)
        one_vector = torch.ones_like(t)
        
        for e in range(self.num_Event):
            I_2 = (k == e+1).float()
            I_2_diag = torch.diag(I_2.squeeze())
            tmp_e = out[:, e, :]
            
            R = torch.matmul(tmp_e, mask2.transpose(0, 1))
            diag_R = torch.diag(R).unsqueeze(1)
            R = torch.matmul(one_vector, diag_R.transpose(0, 1)) - R
            R = R.transpose(0, 1)
            
            T = F.relu(torch.sign(torch.matmul(one_vector, t.transpose(0, 1)) - 
                                  torch.matmul(t, one_vector.transpose(0, 1))))
            T = torch.matmul(I_2_diag, T)
            
            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)
            
        eta = torch.stack(eta, dim=1)
        eta = torch.mean(eta.reshape(-1, self.num_Event), dim=1, keepdim=True)
        return torch.sum(eta)

    def compute_loss(self, out, t, k, mask1, mask2, alpha=1.0, beta=1.0):
        loss1 = self.log_likelihood_loss(out, t, k, mask1, mask2)
        loss2 = self.ranking_loss(out, t, k, mask2)
        return alpha * loss1 + beta * loss2

def _predict_absolute_risk_deephit(model, x_test, t_max, n_categories, times, device="cpu"):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds, _ = model(x_test_tensor)
    preds_np = preds.cpu().numpy()
    
    n_samples, n_events, _ = preds_np.shape
    n_times = len(times)
    abs_risks = np.zeros((n_samples, n_events, n_times))
    
    for t_idx, time_val in enumerate(times):
        scaled_t = (time_val / t_max) * (n_categories - 1)
        lower_bin = int(np.floor(scaled_t))
        upper_bin = min(int(np.ceil(scaled_t)), n_categories-1)
        
        if lower_bin == upper_bin or upper_bin >= n_categories:
            for k in range(n_events):
                abs_risks[:, k, t_idx] = np.sum(preds_np[:, k, :lower_bin+1], axis=1)
        else:
            weight_upper = scaled_t - lower_bin
            weight_lower = 1 - weight_upper
            for k in range(n_events):
                cum_prob_lower = np.sum(preds_np[:, k, :lower_bin+1], axis=1)
                cum_prob_upper = np.sum(preds_np[:, k, :upper_bin+1], axis=1)
                abs_risks[:, k, t_idx] = weight_lower * cum_prob_lower + weight_upper * cum_prob_upper
    
    return [abs_risks[:, k, :] for k in range(n_events)]

def train_deephit_cr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    tune: bool = False,
    n_trials: int = 10,
    out_dir: str = "results",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    num_durations: int = 20,
    **kwargs
) -> tuple:
    feats = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train = df_train[feats].values.astype(np.float32)
    T_train = df_train[duration_col].values.astype(np.float32)
    E_train = df_train[event_col].values.astype(np.int64)
    
    X_test  = df_test[feats].values.astype(np.float32)
    t_max = np.max(T_train)
    num_Event = int(np.max(E_train))
    
    dataset = SurvivalDatasetDeepHit(X_train, T_train, E_train, num_durations)
    success = False
    current_bs = batch_size
    while not success:
        try:
            loader = DataLoader(dataset, batch_size=current_bs, shuffle=True)
            model = DeepHitCRPattern(x_dim=X_train.shape[1], num_Event=num_Event, num_Category=num_durations).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                for x, t, e, t_disc in loader:
                    x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
                    mask1 = create_fc_mask1_gpu(e, t_disc, num_Event, num_durations, device)
                    mask2 = create_fc_mask2_gpu(t_disc, num_durations, device)
                    out, _ = model(x)
                    loss = model.compute_loss(out, t.view(-1, 1), e.view(-1, 1), mask1, mask2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            success = True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"      {C_YELLOW}→ CUDA OOM (DeepHitCR)! Reducing batch_size to {current_bs} and restarting{C_RESET}")
            else:
                raise e
            
    # Inference wrapper mapping
    grid = np.linspace(np.min(T_train) + 1e-6, np.max(T_train) - 1e-6, 50)
    cif_per_cause = _predict_absolute_risk_deephit(model, X_test, t_max, num_durations, grid, device)
    
    # Simple risk estimation (cause 1 risk integrated over time for compatibility)
    span = grid[-1] - grid[0] + 1e-8
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    risk_cause1 = _trapz(cif_per_cause[0], grid, axis=1) / span
    
    return model, risk_cause1, cif_per_cause, grid
