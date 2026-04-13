import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        if isinstance(x, tuple):
            tensor, _ = x
        else:
            tensor = x
        # Reshape shape (batch, hidden)
        if tensor.dim() == 2:
            return tensor
        return tensor.mean(dim=1)

class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fc2 = nn.Linear(3 * self.hidden_size, 5 * self.hidden_size)
        self.fc3 = nn.Linear(5 * self.hidden_size, 3 * self.hidden_size)
        self.fc4 = nn.Linear(3 * self.hidden_size, output_size)
        
    def forward(self, x, y):
        x = torch.cat((x, y.reshape(-1, 1)), dim=1)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.LSTM1(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        out = self.fc4(x)
        return out

class DySurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features, seq_len=1):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, in_features, batch_first=True)
        self.extract = extract_tensor()
        self.fc11 = nn.Linear(in_features, 3 * in_features)
        self.fc12 = nn.Linear(3 * in_features, 5 * in_features)
        self.fc13 = nn.Linear(5 * in_features, 3 * in_features)
        self.fc14 = nn.Linear(3 * in_features, encoded_features)
        self.fc24 = nn.Linear(3 * in_features, encoded_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.output_all = True

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 3 * in_features), nn.ReLU(), 
            nn.Linear(3 * in_features, 5 * in_features), nn.ReLU(), 
            nn.Linear(5 * in_features, 3 * in_features), nn.ReLU(), 
            nn.Linear(3 * in_features, out_features),
        )
        
        self.decoder2 = Decoder(seq_len, encoded_features + 1, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encoder(self, x):
        # x: (batch, seq_len, in_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Pass through LSTM        
        x, _ = self.lstm1(x)
        x = self.extract(x)

        x = self.relu(self.fc11(x)) 
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z
    
    def forward(self, x):
        # x: (batch, seq_len, in_features)
        mu, logvar = self.encoder(x.float())
        z = self.reparameterize(mu, logvar)
        phi = self.surv_net(z)

        if self.output_all:
            # During training and validation, we need all components for DySurvLoss
            y_dummy = torch.zeros(x.shape[0], 1, device=x.device)
            decoded = self.decoder2(z, y_dummy)
            return decoded, phi, mu, logvar
        
        return phi

    def predict(self, x):
        mu, _ = self.encoder(x.float())
        # Use mu instead of reparameterizing for deterministic prediction
        return self.surv_net(mu)

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                         reduction: str = 'mean', training: bool = True,
                         pos_weight: float | None = None) -> Tensor:
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`.")
    
    if events.dtype is torch.bool:
        events = events.float()
    
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    pw = torch.tensor([pos_weight if pos_weight is not None else 1.0], device=phi.device)

    if training:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, pos_weight=pw, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')

    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class DySurvLoss(nn.Module):
    def __init__(self, alphas: list = [1.0, 0.1, 0.1], pos_weight: float | None = None, **kwargs):
        super().__init__()
        self.alphas = alphas
        self.pos_weight = pos_weight

    def forward(self, *args):
        """
        Handles both packed and unpacked arguments from torchtuples.
        Expected: (decoded, phi, mu, logvar), (target_loghaz, target_ae)
        """
        if len(args) == 2:
            outputs, targets = args
        elif len(args) == 6:
            outputs = args[:4]
            targets = args[4:]
        else:
            raise TypeError(f"DySurvLoss.forward expects 2 or 6 positional arguments, got {len(args)}")

        decoded, phi, mu, logvar = outputs
        target_loghaz, target_ae = targets
        
        # target_loghaz: (idx_durations, events)
        idx_durations, events = target_loghaz
        
        # Survival Loss
        loss_surv = nll_logistic_hazard(phi, idx_durations, events, pos_weight=self.pos_weight)

        # AutoEncoder Loss
        # target_ae might need to be unsqueezed if its (batch, features) but decoded is (batch, seq, features)
        if target_ae.dim() == 2 and decoded.dim() == 3:
            target_ae = target_ae.unsqueeze(1)
        loss_ae = F.mse_loss(decoded, target_ae)

        # KL-divergence Loss
        loss_kd = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return self.alphas[0] * loss_surv + self.alphas[1] * loss_ae + self.alphas[2] * loss_kd
