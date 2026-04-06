"""
survpfn.models.shared.heads — Unified survival head layer for FM-embedding pipelines.

Supports both Single-Risk (Cox, DeepHit, PCHazard, MTLR) and 
Competing-Risks (DeepHit CR).
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, DeepHitSingle, MTLR, PCHazard

from survpfn.utils.config import load_model_config, apply_tuning_params
from survpfn.models.shared.preprocessing import get_time_bins

# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------

class MLPVanilla(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_nodes: list,
        out_features: int,
        batch_norm: bool = True,
        dropout: Optional[float] = None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias: bool = True,
    ):
        super().__init__()
        nodes = [in_features] + list(num_nodes)
        layers: list[nn.Module] = []

        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

        out_layer = nn.Linear(nodes[-1], out_features, bias=output_bias)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity="relu")
        if output_bias:
            nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# Competing Risks Head (DeepHit CR)
# ---------------------------------------------------------------------------

class DeepHitCRHead(nn.Module):
    def __init__(self, in_dim, num_events, num_categories,
                 num_nodes_shared=[128], num_nodes_cs=[64, 64],
                 dropout_rate=0.1):
        super().__init__()
        self.num_events = num_events
        self.num_categories = num_categories
        
        # Shared network (1 layer)
        self.shared_net = nn.Sequential(
            nn.Linear(in_dim, num_nodes_shared[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cause-specific networks (1 layer + final output layer = 2 layers total for each cause)
        self.cs_nets = nn.ModuleList()
        for _ in range(num_events):
            cs_layers = nn.Sequential(
                nn.Linear(num_nodes_shared[0], num_nodes_cs[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.cs_nets.append(cs_layers)
        
        self.output_layer = nn.Linear(num_events * num_nodes_cs[0], num_events * num_categories)

    def forward(self, x):
        shared_out = self.shared_net(x)
        cs_outputs = [net(shared_out) for net in self.cs_nets]
        stacked_out = torch.stack(cs_outputs, dim=1)
        # Final linear expects (batch, num_events * hidden)
        reshaped_out = stacked_out.view(x.size(0), -1)
        
        logits = self.output_layer(reshaped_out)
        # Softmax over (num_events * num_categories)
        out = F.softmax(logits, dim=1)
        out = out.view(-1, self.num_events, self.num_categories)
        return out

# ---------------------------------------------------------------------------
# Unified survival head wrapper
# ---------------------------------------------------------------------------

class EmbeddingSurvHead:
    """
    Unified head for Single-Risk and Competing-Risks.
    """
    def __init__(
        self,
        embedding_dim: int,
        head_type: str = "cox", # cox, deephit, pchazard, mtlr, deephit_cr
        num_events: int = 1,
        num_durations: int = 100,
        num_nodes: list = [64, 64],
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_norm: bool = False,
        device: str = "cpu",
        cr_loss_type: str = "deephit"
    ):
        self.embedding_dim = embedding_dim
        self.head_type = head_type.lower()
        self.num_events = num_events
        self.num_durations = num_durations
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.lr = lr
        self.batch_norm = batch_norm
        self.device = device
        self.cr_loss_type = cr_loss_type
        
        self.model = None
        self.labtrans = None
        self._x_train = None
        self._y_train = None
        self.duration_index = None

    def fit(self, embeddings, durations, events, epochs=100, batch_size=256, verbose=True):
        x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        if self.num_events > 1 or self.head_type == "deephit_cr":
            self._fit_cr(embeddings, durations, events, epochs, batch_size, verbose)
        else:
            self._fit_sr(embeddings, durations, events, epochs, batch_size, verbose)
        return self

    def _fit_sr(self, embeddings, durations, events, epochs, batch_size, verbose):
        x = embeddings.astype(np.float32)
        if self.head_type == "cox":
            n_out = 1
            self.labtrans = None
            y_train = (durations.astype(np.float32), events.astype(np.float32))
        elif self.head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            self.labtrans = LabTransDiscreteTime(self.num_durations, scheme='quantiles')
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = self.labtrans.out_features
        elif self.head_type == "pchazard":
            self.labtrans = PCHazard.label_transform(self.num_durations, scheme='quantiles')
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = self.labtrans.out_features
        elif self.head_type == "mtlr":
            self.labtrans = MTLR.label_transform(self.num_durations, scheme='quantiles')
            y_train = self.labtrans.fit_transform(durations, events)
            n_out = self.labtrans.out_features
        else:
            raise ValueError(f"Unknown SR head type: {self.head_type}")

        if self.labtrans:
            self.duration_index = self.labtrans.cuts

        net = MLPVanilla(self.embedding_dim, self.num_nodes, n_out, self.batch_norm, self.dropout).to(self.device)
        opt = tt.optim.Adam(lr=self.lr)

        if self.head_type == "cox":
            self.model = CoxPH(net, opt)
        elif self.head_type == "deephit":
            self.model = DeepHitSingle(net, opt, duration_index=self.duration_index)
        elif self.head_type == "pchazard":
            self.model = PCHazard(net, opt, duration_index=self.duration_index)
        elif self.head_type == "mtlr":
            self.model = MTLR(net, opt, duration_index=self.duration_index)

        self.model.fit(x, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        if self.head_type == "cox":
            self.model.compute_baseline_hazards()

    def _fit_cr(self, embeddings, durations, events, epochs, batch_size, verbose):
        # DeepHit CR implementation
        # Discretize times
        self._bin_times, t_disc = discretize_competing_times(durations, events, self.num_durations)
        self.num_durations = len(self._bin_times)
        
        x_tensor = torch.tensor(embeddings, dtype=torch.float32)
        t_tensor = torch.tensor(durations, dtype=torch.float32)
        e_tensor = torch.tensor(events, dtype=torch.long)
        td_tensor = torch.tensor(t_disc, dtype=torch.long)
        
        dataset = TensorDataset(x_tensor, t_tensor, e_tensor, td_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = DeepHitCRHead(self.embedding_dim, self.num_events, self.num_durations,
                                    num_nodes_shared=[self.num_nodes[0]],
                                    num_nodes_cs=self.num_nodes[1:],
                                    dropout_rate=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(epochs):
            self.model.train()
            for x_b, t_b, e_b, td_b in loader:
                x_b, t_b, e_b, td_b = x_b.to(self.device), t_b.to(self.device), e_b.to(self.device), td_b.to(self.device)
                out = self.model(x_b)
                
                if self.cr_loss_type == "deephit":
                    loss = compute_deephit_cr_loss(out, t_b, e_b, td_b, self.num_events, self.num_durations, self.device)
                elif self.cr_loss_type == "deephit_v2":
                    loss = compute_deephit_cr_loss_v2(out, t_b, e_b, td_b, self.num_events, self.num_durations, self.device)
                elif self.cr_loss_type == "cox":
                    loss = compute_cox_cr_loss(out, t_b, e_b, self.num_events, self.device)
                else:
                    raise ValueError(f"Unknown cr_loss_type: {self.cr_loss_type}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_survival(self, embeddings) -> pd.DataFrame:
        if self.num_events > 1:
            return self._predict_cr(embeddings)
        
        x = embeddings.astype(np.float32)
        if self.head_type == "cox":
            return self.model.predict_surv_df(x)
        if hasattr(self.model, "interpolate"):
            return self.model.interpolate(10).predict_surv_df(x)
        return self.model.predict_surv_df(x)

    def _predict_cr(self, embeddings) -> List[pd.DataFrame]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            preds = self.model(x).cpu().numpy() # (batch, num_events, num_bins)
                   # Convert to CIDFs
        cifs = []
        for k in range(self.num_events):
            # CDF for event k: sum probs of event k up to time t
            cif_k = np.cumsum(preds[:, k, :], axis=1) # (batch, num_bins)
            df = pd.DataFrame(cif_k.T, index=self._bin_times)
            cifs.append(df)
        return cifs

# ---------------------------------------------------------------------------
# CR Loss and Helpers (Extracted from deephit_cr.py)
# ---------------------------------------------------------------------------

def discretize_competing_times(durations, events, n_bins):
    event_times = durations[events > 0]
    if len(event_times) == 0:
        event_times = durations
    # Use linspace on event times to ensure n_bins intervals
    max_t = event_times.max()
    bins = np.linspace(0, max_t, n_bins + 1)[1:]
    t_disc = np.digitize(durations, bins, right=True)
    t_disc = np.clip(t_disc, 0, n_bins - 1)
    return bins, t_disc

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

# ---------------------------------------------------------------------------
# Training & Tuning
# ---------------------------------------------------------------------------

def train_fm_embedding_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    embedding_fn: Callable,
    emb_kwargs: Optional[dict] = None,
    head_type: str = "cox",
    num_durations: int = 100,
    tune: bool = False,
    n_trials: int = 20,
    out_dir: str = "results",
    study_id: Optional[str] = None,
    fm_name: str = "fm",
    task_type: str = "sr", # "sr" or "cr"
    cr_loss_type: str = "deephit",
    **kwargs
) -> tuple:
    emb_kwargs = emb_kwargs or {}
    feat_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
    X_train = df_train[feat_cols].values.astype(np.float32)
    t_train = df_train[duration_col].values
    e_train = df_train[event_col].values
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_bin = (e_train > 0).astype(np.float32)

    train_emb, test_emb = embedding_fn(X_train, y_bin, X_test, **emb_kwargs)
    emb_dim = train_emb.shape[1]

    # Load config from JSON
    config_name = "embedding_head" if task_type == "sr" else "deephit_cr"
    config = load_model_config(config_name)
    params = config["default"]

    num_events = int(e_train.max()) if task_type == "cr" else 1

    if tune:
        import optuna
        # ... Optuna logic using config["tuning"] ...
        pass

    model = EmbeddingSurvHead(
        embedding_dim=emb_dim,
        head_type=head_type if task_type == "sr" else "deephit_cr",
        num_events=num_events,
        num_durations=num_durations,
        num_nodes=params.get("num_nodes", [64, 64]),
        dropout=params.get("dropout", 0.1),
        lr=params.get("lr", 1e-3),
        device="cuda" if torch.cuda.is_available() else "cpu",
        cr_loss_type=cr_loss_type
    )
    model.fit(train_emb, t_train, e_train, epochs=params.get("epochs", 100))

    if task_type == "cr":
        cifs = model.predict_survival(test_emb)
        # return model and CIFs
        # For compatibility with benchmark, we might need a single risk score
        risk = 1.0 - cifs[0].iloc[-1].values # Example: risk of cause 1
        return model, risk, [c.values.T for c in cifs], cifs[0].index.values
    else:
        surv_df = model.predict_survival(test_emb)
        times = surv_df.index.values
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = _trapz(surv_df.values.T, times)
        risk = -auc
        surv_probs = np.clip(surv_df.values.T, 0, 1)
        return model, risk, surv_probs, times
