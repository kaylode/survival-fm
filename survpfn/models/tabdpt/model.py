"""
survpfn.models.tabdpt.model — TabDPT backbone + survival head (jointly-trained / frozen).

Classes / functions
-------------------
* TabDPTSurvModel     — PyTorch nn.Module with TabDPT backbone + survival head.
* TabDPTSurvPH        — High-level wrapper with fit / predict_survival.
* get_tabdpt_embeddings — Extraction utility for frozen embeddings.
"""

import os
import pathlib
import copy
from typing import Union, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchtuples as tt

from survpfn.models.tabdpt.tabdpt.model import TabDPTModel, TABDPT_CHECKPOINT_PATH
from survpfn.models.tabdpt.tabdpt.tabdpt import TabDPTClassifier
from survpfn.models.tabdpt.tabdpt.models_utils import pad_x
from survpfn.models.shared.finetune import BaseJointSurvFinetune, BaseBackboneSurvModel

_DEFAULT_CONTEXT_SIZE = 128

# ---------------------------------------------------------------------------
# TabDPT Survival Model
# ---------------------------------------------------------------------------

class TabDPTSurvModel(BaseBackboneSurvModel):
    def __init__(
        self,
        device: str,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        batch_norm: bool = False,
        freeze_tabdpt: bool = False,
    ):
        checkpoint = torch.load(TABDPT_CHECKPOINT_PATH, map_location=device, weights_only=False)
        loaded_model = TabDPTModel.load(
            model_state=checkpoint["model"],
            config=checkpoint["cfg"],
        )
        ninp = loaded_model.ninp
        max_features = loaded_model.num_features
        
        super().__init__(
            ninp=ninp, n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            task_type=task_type, num_events=num_events, use_adapter=use_adapter,
            input_dim=input_dim, batch_norm=batch_norm, 
            adapter_output_dim=max_features
        )
        
        self.tabdpt = loaded_model
        self.max_features = max_features

        self.to(device)
        self.device = device
        for param in self.tabdpt.parameters():
            param.requires_grad = not freeze_tabdpt
        self._move_head_to_last()

    def _to_padded(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, n_features) → (N, 1, max_features)."""
        x = self._apply_pca_and_adapter(x)
        x = x.unsqueeze(1)
        n_feat = x.shape[-1]
        if n_feat < self.max_features:
            x = pad_x(x, self.max_features)
        elif n_feat > self.max_features:
            x = x[..., : self.max_features]
        return x

    def forward(
        self,
        x_query: torch.Tensor,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs
    ) -> torch.Tensor:
        if x_context is None:
            x_context = self._ctx_x
            y_context = self._ctx_y
        # Padded forward
        out, emb = self.tabdpt(
            torch.cat([x_context, self._to_padded(x_query)], dim=0),
            y=y_context,
            eval_pos=x_context.shape[0],
            return_embeddings=True
        )
        query_embs = emb[x_context.shape[0]:].squeeze(1)
        query_flat = self.norm(query_embs.to(torch.float32))

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
            cls_logits = out[:, 0, :2].to(torch.float32)
            return head_out, cls_logits
        return head_out


# ---------------------------------------------------------------------------
# TabDPT Survival Wrapper
# ---------------------------------------------------------------------------

class TabDPTSurvPH(BaseJointSurvFinetune):
    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 100,
        head_num_nodes: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        n_out: Optional[int] = None,
        num_events: int = 1,
        device: str = "cuda:0",
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        cr_loss_type: str = "deephit",
        task_type: str = "sr",
        alpha: float = 1.0,
        freeze_backbone: bool = True,
        deephit_alpha: float = 0.2,
        deephit_sigma: float = 0.1,
        **kwargs
    ):
        self.task_type = task_type
        self.num_events = num_events
        self.head_type = head_type.lower()
        self.device = device
        self.context_size = context_size
        self.num_durations = num_durations
        self.cr_loss_type = cr_loss_type
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.deephit_alpha = deephit_alpha
        self.deephit_sigma = deephit_sigma
        self.backbone_name = "tabdpt"

        if n_out is None:
            if self.head_type == "cox":
                n_out = 1
            elif self.task_type == "cr":
                n_out = num_durations * num_events
            else:
                n_out = num_durations

        self.net = TabDPTSurvModel(
            device=device, n_out=n_out,
            head_num_nodes=head_num_nodes, dropout=dropout,
            task_type=task_type, num_events=num_events,
            use_adapter=use_adapter, input_dim=input_dim,
            batch_norm=(self.head_type == "deepsurv"),
            freeze_tabdpt=freeze_backbone,
        ).to(device)

        self.model, self.labtrans = self._init_pycox_model(self.head_type, num_durations, learning_rate, self.net)


