"""
survpfn.models.tabpfn.model — TabPFN backbone + survival head (jointly-trained / frozen).

Classes / functions
-------------------
* TabPFNSurvModel     — PyTorch nn.Module with TabPFN backbone + survival head
                        (jointly-trained variant; supports cox/deephit/pchazard/mtlr)
* TabPFNSurvPH        — High-level wrapper with fit / predict_survival
* get_tabpfn_embeddings — Extraction utility for frozen embeddings.
"""

import os
import pathlib
from typing import Union, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from survpfn.models.tabpfn.backbone_v1 import load_model_workflow
# from survpfn.models.tabpfn.backbone_v2 import load_model_workflow
from survpfn.models.shared.finetune import BaseJointSurvFinetune, BaseBackboneSurvModel

# ---------------------------------------------------------------------------
# TabPFN Survival Model (Backbone + Survival Head)
# ---------------------------------------------------------------------------

class TabPFNSurvModel(BaseBackboneSurvModel):
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
        batch_norm: bool = False,
    ):
        """
        TabPFNSurvModel: Uses a pre-trained TabPFN model and adds a survival head.
        Uses a forward hook to capture transformer embeddings for the survival task.
        """
       
        # Load TabPFN
        loaded_model = load_model_workflow(device=device, only_inference=True)
        
        # Determine actual embedding dimension (Wrapper.ninp might be incorrect)
        self.num_expected_features = 100
        with torch.no_grad():
            dummy_x = torch.randn(1, self.num_expected_features).to(device)
            dummy_y = torch.zeros(1).to(device)
            dummy_out = loaded_model(dummy_x, dummy_y, dummy_x)
            actual_ninp = dummy_out['test_embeddings'].shape[-1]


        super().__init__(
            ninp=actual_ninp, n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            task_type=task_type, num_events=num_events, use_adapter=use_adapter,
            input_dim=input_dim, batch_norm=batch_norm, 
            adapter_output_dim=self.num_expected_features
        )

        self.model = loaded_model
        
        # Freezing logic
        for param in self.model.parameters():
            param.requires_grad = not freeze_tabpfn

        self.to(dtype)
        self.device = device
        self.dtype = dtype
        self._move_head_to_last()
        self.to(device)

    def _apply_pca_and_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, n_features) -> (N, num_expected_features)."""
        x = self._apply_pca_and_adapter(x)
        num_features = x.shape[-1]
        if num_features < self.num_expected_features:
            padding = torch.zeros(*x.shape[:-1], self.num_expected_features - num_features, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        elif num_features > self.num_expected_features:
            x = x[..., :self.num_expected_features]
        return x

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs
    ):
        """Jointly-trained forward pass."""
        if x_context is None:
            if self._ctx_x is None:
                raise RuntimeError("No context supplied and none stored. Call set_context() or pass x_context.")
            x_context = self._ctx_x
            y_context = self._ctx_y

        device = x_query.device
        self.to(device)
        x_query = x_query.to(device).to(self.dtype)
        x_context = x_context.to(device).to(self.dtype)
        y_context = y_context.to(device).to(self.dtype)

        x_full = torch.cat([x_context, x_query], dim=0)
        x_full = self._apply_pca_and_pad(x_full)
        K = x_context.shape[0]

        output = self.model(x_full[:K], y_context, x_full[K:])
        query_embs = output['test_embeddings'].to(torch.float32).squeeze(1)
        query_flat = self.norm(query_embs)

        if self.task_type == "cr":
            cs_outs = [head(query_flat) for head in self.cs_heads]
            head_out = torch.stack(cs_outs, dim=1)
            head_out = F.softmax(head_out.view(head_out.size(0), -1), dim=1).view(
                head_out.size(0), self.num_events, -1
            )
        else:
            head_out = self.survival_head(query_flat)
            if head_out.size(-1) == 1:
                head_out = head_out.squeeze(-1)

        if return_logits:
            cls_logits = output["logits"].squeeze(1).to(torch.float32)
            return head_out, cls_logits

        return head_out


# ---------------------------------------------------------------------------
# TabPFN Survival Wrapper
# ---------------------------------------------------------------------------

class TabPFNSurvPH(BaseJointSurvFinetune):
    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 100,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        freeze_tabpfn: bool = True,
        dtype: torch.dtype = torch.float32,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        context_size: int = 1024,
        deephit_alpha: float = 0.2,
        deephit_sigma: float = 0.1,
        **kwargs
    ):
        super().__init__(num_durations=num_durations, **kwargs)
        # 1. Store shared attributes (no redundancy)
        self.task_type = task_type
        self.num_events = num_events
        self.head_type = head_type.lower()
        self.dtype = dtype
        self.context_size = context_size
        self.deephit_alpha = deephit_alpha
        self.deephit_sigma = deephit_sigma
        self.backbone_name = "tabpfn"
        
        # 2. Determine output dimension
        if self.head_type in ("cox", "deepsurv"):
            n_out = 1
        elif self.task_type == "cr":
            n_out = num_durations * num_events
        else:
            n_out = num_durations

        # 3. Initialize Backbone + Head
        self.net = TabPFNSurvModel(
            n_out=n_out, 
            head_num_nodes=head_num_nodes, 
            dropout=dropout,
            freeze_tabpfn=freeze_tabpfn, 
            dtype=dtype, 
            device=self.device,
            task_type=task_type, 
            num_events=num_events,
            use_adapter=use_adapter, 
            input_dim=input_dim,
            batch_norm=(self.head_type == "deepsurv"),
        )
        
        # 4. Initialize Pycox Wrapper
        self.model, self.labtrans = self._init_pycox_model(
            self.head_type, num_durations, self.learning_rate, self.net
        )

