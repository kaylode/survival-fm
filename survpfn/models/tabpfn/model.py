"""

Classes / functions
-------------------
* TabPFNSurvModel     — PyTorch nn.Module with TabPFN backbone + survival head
                        (jointly-trained variant; supports cox/deephit/pchazard/mtlr)
* TabPFNSurvPH        — High-level wrapper with fit / predict_survival
"""

import os
import pathlib
from survpfn.models.tabpfn.backbone.utils import load_model_workflow
from sklearn.model_selection import train_test_split
from pycox.models import CoxPH, PCHazard, MTLR, DeepHitSingle
from pycox.evaluation import EvalSurv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Union, List, Optional
import torchtuples as tt
import pandas as pd

# ---------------------------------------------------------------------------
# TabPFN Survival Model (Backbone + Survival Head)
# ---------------------------------------------------------------------------

class TabPFNSurvModel(nn.Module):
    def __init__(
        self,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        device: str = "cuda:0",
        freeze_tabpfn: bool = True,
        dtype: torch.dtype = torch.float32,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
    ):
        """
        TabPFNSurvModel: Uses a pre-trained TabPFN model and adds a survival head.
        Uses a forward hook to capture transformer embeddings for the survival task.
        """
        super().__init__()

        # Load TabPFN
        base_path = pathlib.Path(__file__).parent.parent
        model_tuple, self.config, _ = load_model_workflow(
            0, 42, add_name='',
            base_path=base_path, device=device,
            only_inference=True
        )
        self.tabpfn = model_tuple[2]
        self.to(dtype)
        self.device = device
        self.dtype = dtype

        # Freezing logic
        for param in self.tabpfn.parameters():
            param.requires_grad = not freeze_tabpfn

        self.ninp = self.tabpfn.ninp
        self.num_classes = 10
        self.num_expected_features = self.config.get('num_features', 100)

        self.task_type = task_type
        self.num_events = num_events
        
        # --- (1) Input Adapter (used BEFORE TabPFN) ---
        if use_adapter:
            if input_dim is None:
                raise ValueError("input_dim must be provided if use_adapter is True")
            # All MLP should be 2-layer and no BatchNorm
            self.input_adapter = nn.Sequential(
                nn.Linear(input_dim, self.num_expected_features),
                nn.ReLU(),
                nn.Linear(self.num_expected_features, self.num_expected_features)
            )
        else:
            self.input_adapter = None

        if task_type == "cr":
            nodes = [self.ninp] + list(head_num_nodes)
            # Cause-specific heads (now directly on transformer ninp)
            # All MLP should be 2-layer and no BatchNorm
            self.cs_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.ninp, nodes[1]),
                    nn.ReLU(),
                    nn.Linear(nodes[1], n_out // num_events)
                ) for _ in range(num_events)
            ])
            self.survival_head = None # Not used directly
        else:
            # Survival Head (MLP) mapping transformer output (ninp) -> Risk Score or Bins
            # All MLP should be 2-layer and no BatchNorm
            hidden_dim = head_num_nodes[0] if len(head_num_nodes) > 0 else 128
            self.survival_head = nn.Sequential(
                nn.Linear(self.ninp, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, n_out, bias=False)
            )
        
        self.to(device)

        # Capturing embeddings using a forward hook
        self._transformer_output = None
        def hook_fn(module, input, output):
            self._transformer_output = output
        self.tabpfn.transformer_encoder.register_forward_hook(hook_fn)

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        y_pfn: torch.Tensor = None,
        eval_pos: int = None,
        return_pfn: bool = False,
        **kwargs
    ):
        if isinstance(input, tuple):
            x, y_pfn_arg = input
            y_pfn = y_pfn_arg if y_pfn is None else y_pfn
        else:
            x = input

        x = x.to(self.dtype)
        if y_pfn is not None:
             y_pfn = y_pfn.to(self.dtype)
        else:
             y_pfn = torch.zeros(x.shape[0], device=x.device, dtype=self.dtype)

        # Apply Input Adapter if enabled
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Match feature dimension for TabPFN
        num_features = x.shape[-1]
        if num_features < self.num_expected_features:
            padding = torch.zeros(*x.shape[:-1], self.num_expected_features - num_features, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        elif num_features > self.num_expected_features:
            x = x[..., :self.num_expected_features]

        if eval_pos is None:
            eval_pos = 0

        # TabPFN forward pass
        self.to(x.device)
        device_type = "cuda" if "cuda" in str(x.device) else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
            enabled=True
        ):
            logits_pfn = self.tabpfn((None, x, y_pfn), single_eval_pos=eval_pos)
        
        query_embs = self._transformer_output[eval_pos:]

        # Survival forward
        # If output dim is 1 (Cox), we want (BatchSize,)
        # If output dim > 1 (Discrete), we want (BatchSize, n_out)
        
        if self.task_type == "cr":
            query_flat = query_embs.squeeze(0) if query_embs.dim()==3 else query_embs
            cs_outs = [head(query_flat) for head in self.cs_heads]
            head_out = torch.stack(cs_outs, dim=1) # (batch, num_events, num_bins)
            # Softmax across all events and bins
            head_out = F.softmax(head_out.view(head_out.size(0), -1), dim=1).view(head_out.size(0), self.num_events, -1)
        else:
            if query_embs.dim() == 3:
                # SeqLen * BatchSize
                T, B, H = query_embs.shape
                head_out = self.survival_head(query_embs.reshape(T * B, H)).view(T, B, -1)
                # Take only the first time step if T > 1? 
                # Or if seq_len is 1, just squeeze.
                if T == 1:
                    head_out = head_out.squeeze(0)
                else:
                    # If T > 1, the pycox loss will fail unless bdur matches. 
                    # For now, let's assume we want one output per input sample. 
                    # If each sample was processed independently, T=1.
                    head_out = head_out.squeeze(0) 
            else:
                head_out = self.survival_head(query_embs)
                
            if head_out.size(-1) == 1:
                head_out = head_out.squeeze(-1)

        if return_pfn:
            logits_pfn = logits_pfn[:, :self.num_classes] if logits_pfn.dim() == 2 else logits_pfn[:, :, :self.num_classes]
            return head_out, logits_pfn

        return head_out


class TabPFNSurvPH:
    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 100,
        head_num_nodes: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        dropout: float = 0.2,
        freeze_tabpfn: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0",
        n_out: Optional[int] = None,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        cr_loss_type: str = "deephit",
        **kwargs
    ):
        self.task_type = task_type
        self.num_events = num_events
        self.head_type = head_type.lower()
        self.dtype = dtype
        self.device = device
        self.alpha = alpha
        self.num_durations = num_durations
        self.cr_loss_type = cr_loss_type
        if n_out is None:
            if self.head_type == "cox":
                n_out = 1
            elif self.task_type == "cr":
                n_out = num_durations * num_events
            else:
                n_out = num_durations
        
        self.net = TabPFNSurvModel(
            n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            freeze_tabpfn=freeze_tabpfn, dtype=dtype, device=device,
            task_type=task_type, num_events=num_events,
            use_adapter=use_adapter, input_dim=input_dim,
        ).to(device, dtype)
        
        if self.head_type == "cox":
            self.model = CoxPH(self.net, tt.optim.Adam(lr=learning_rate))
            self.labtrans = None
        elif self.head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            # Quantile binning is essential for skewed medical data
            self.labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
            self.model = DeepHitSingle(self.net, tt.optim.Adam(lr=learning_rate), duration_index=self.labtrans.cuts)
        elif self.head_type == "pchazard":
            try:
                self.labtrans = PCHazard.label_transform(num_durations, scheme='quantiles')
            except TypeError:
                self.labtrans = PCHazard.label_transform(num_durations)
            self.model = PCHazard(self.net, tt.optim.Adam(lr=learning_rate), duration_index=self.labtrans.cuts)
        elif self.head_type == "mtlr":
            try:
                self.labtrans = MTLR.label_transform(num_durations, scheme='quantiles')
            except TypeError:
                self.labtrans = MTLR.label_transform(num_durations)
            self.model = MTLR(self.net, tt.optim.Adam(lr=learning_rate), duration_index=self.labtrans.cuts)
        elif self.head_type == "deephit_cr":
            self.labtrans = None
            self.model = None # Using custom training loop for CR
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        from pycox.models.loss import CoxPHLoss, DeepHitSingleLoss, NLLPCHazardLoss, NLLMTLRLoss
        from pycox.models.data import pair_rank_mat
        from survpfn.models.shared.heads import (
            compute_deephit_cr_loss, 
            compute_deephit_cr_loss_v2,
            compute_cox_cr_loss
        )
        
        if self.task_type == "cr":
            criterion_surv = None # handled by shared helper
        elif self.head_type == "cox":
            criterion_surv = CoxPHLoss()
        elif self.head_type == "deephit":
            criterion_surv = DeepHitSingleLoss(alpha=0.2, sigma=0.1)
        elif self.head_type == "pchazard":
            criterion_surv = NLLPCHazardLoss()
        elif self.head_type == "mtlr":
            criterion_surv = NLLMTLRLoss()
            
        criterion_pfn = nn.CrossEntropyLoss()
        optimizer = self.model.optimizer if self.model else torch.optim.Adam(self.net.parameters(), lr=0.001)

        x_pt = torch.from_numpy(x.copy()).to(self.device, self.dtype)
        
        # Label transforms for discrete models
        if self.task_type == "cr":
            from survpfn.models.shared.heads import discretize_competing_times
            self._bin_times, t_disc = discretize_competing_times(durations, events, self.num_durations)
            self.num_durations = len(self._bin_times)
            dur_pt = torch.from_numpy(t_disc).to(self.device, torch.long)
            dur_pt_cont = torch.from_numpy(durations.copy()).to(self.device, self.dtype)
            ev_pt = torch.from_numpy(events.copy()).to(self.device, torch.long)
            frac_pt = None
        elif self.labtrans is not None:
            targets = self.labtrans.fit_transform(durations, events)
            dur_pt = torch.from_numpy(targets[0]).to(self.device)
            ev_pt = torch.from_numpy(targets[1]).to(self.device)
            # Some transforms like PCHazard have a third element: interval_frac
            frac_pt = None
            if len(targets) > 2:
                frac_pt = torch.from_numpy(targets[2]).to(self.device, self.dtype)
        else:
            dur_pt = torch.from_numpy(durations.copy()).to(self.device, self.dtype)
            dur_pt_cont = dur_pt
            ev_pt = torch.from_numpy(events.copy()).to(self.device, self.dtype)
            frac_pt = None
            
        # y_pfn_pt = torch.from_numpy(y_pfn.copy()).to(self.device) if y_pfn is not None else None
        y_pfn_pt = torch.from_numpy((events > 0).astype(np.int64)).to(self.device)

        for epoch in range(epochs):
            self.net.train()
            indices = torch.randperm(x_pt.size(0))
            dur_pt_cont = dur_pt_cont if 'dur_pt_cont' in locals() else dur_pt
            epoch_loss = 0
            for i in range(0, x_pt.size(0), batch_size):
                idx = indices[i:i + batch_size]
                bx, bdur, bev = x_pt[idx], dur_pt[idx], ev_pt[idx]
                bdur_cont = dur_pt_cont[idx]
                bfrac = frac_pt[idx] if frac_pt is not None else None
                by_pfn = y_pfn_pt[idx] if y_pfn_pt is not None else None
                optimizer.zero_grad()

                head_out, pfn_logits = self.net(bx, y_pfn=by_pfn, return_pfn=True)

                if self.task_type == "cr":
                    if self.cr_loss_type == "deephit":
                         loss_surv = compute_deephit_cr_loss(head_out, bdur_cont, bev, bdur, # dur_pt was discretized for CR
                                                           self.num_events, self.num_durations, self.device)
                    elif self.cr_loss_type == "deephit_v2":
                         loss_surv = compute_deephit_cr_loss_v2(head_out, bdur_cont, bev, bdur, 
                                                              self.num_events, self.num_durations, self.device)
                    elif self.cr_loss_type == "cox":
                         loss_surv = compute_cox_cr_loss(head_out, bdur_cont, bev, self.num_events, self.device)
                    else:
                         raise ValueError(f"Unknown cr_loss_type: {self.cr_loss_type}")
                elif self.head_type == "cox":
                    loss_surv = criterion_surv(head_out, bdur, bev)
                elif self.head_type == "deephit":
                    # DeepHit requires a rank_mat
                    # We might need to ensure it's a tensor on the correct device
                    _bdur = bdur.cpu().numpy() if hasattr(bdur, 'cpu') else bdur
                    _bev = bev.cpu().numpy() if hasattr(bev, 'cpu') else bev
                    _rank_mat = pair_rank_mat(_bdur, _bev)
                    rank_mat = torch.from_numpy(_rank_mat).to(self.device, self.dtype)
                    loss_surv = criterion_surv(head_out, bdur, bev, rank_mat)
                elif self.head_type == "pchazard":
                    # PCHazard requires interval_frac
                    loss_surv = criterion_surv(head_out, bdur, bev, bfrac)
                else:
                    # Discrete heads
                    loss_surv = criterion_surv(head_out, bdur, bev)

                loss_pfn = 0
                if pfn_logits is not None and by_pfn is not None:
                     loss_pfn = criterion_pfn(pfn_logits.view(-1, pfn_logits.size(-1)), by_pfn.view(-1).long())
                total_loss = loss_surv + self.alpha * loss_pfn
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()

                # check nan
                if torch.isnan(total_loss):
                    breakpoint()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss {epoch_loss / (max(1, x_pt.size(0) // batch_size)):.4f}")
        
        # For Cox, compute baseline hazards
        if self.head_type == "cox":
            self.model.compute_baseline_hazards(input=x_pt, target=(dur_pt, ev_pt))
            
        return self

    def predict_survival_df(self, x: np.ndarray):
        self.net.eval()
        x_pt = torch.from_numpy(x).to(self.device, self.dtype)
        with torch.no_grad():
            if self.task_type == "cr":
                out = self.net(x_pt) # (batch, num_events, num_bins)
                # Convert to CIFs
                cifs = []
                for k in range(self.num_events):
                    cif_k = torch.cumsum(out[:, k, :], dim=1).cpu().numpy()
                    df = pd.DataFrame(cif_k.T, index=self._bin_times)
                    cifs.append(df)
                return cifs
            elif self.head_type == "cox":
                return self.model.predict_surv_df(x_pt)
            else:
                # Discrete models: Check if interpolate is available (e.g., LogisticHazard, MTLR)
                if hasattr(self.model, 'interpolate'):
                    return self.model.interpolate(10).predict_surv_df(x_pt)
                return self.model.predict_surv_df(x_pt)