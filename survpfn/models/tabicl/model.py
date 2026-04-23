"""
survpfn.models.tabicl.model — TabICL backbone + survival head (jointly-trained / frozen).

Classes / functions
-------------------
* TabICLSurvModel     — PyTorch nn.Module with TabICL backbone + survival head.
* TabICLSurvPH        — High-level wrapper with fit / predict_survival
* get_tabicl_embeddings — Extraction utility for frozen embeddings.
"""

from __future__ import annotations
import os
import pathlib
from typing import Union, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchtuples as tt

from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier
from survpfn.models.shared.finetune import BaseJointSurvFinetune, BaseBackboneSurvModel

# ---------------------------------------------------------------------------
# TabICL Survival Model
# ---------------------------------------------------------------------------

class TabICLSurvModel(BaseBackboneSurvModel):
    def __init__(
        self,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        batch_norm: bool = False,
        freeze_backbone: bool = True,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        model_path: Optional[str] = None,
        random_state: int = 42,
    ):
        """Standard Joint TabICL-Survival Module."""
        from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier

        clf = TabICLClassifier(
            n_estimators=1,
            norm_methods="none",
            feat_shuffle_method="none",
            use_amp=False,
            allow_auto_download=True,
            checkpoint_version=checkpoint_version,
            model_path=model_path,
            device=device,
            random_state=random_state,
            verbose=False,
        )
        # Warm-up to trigger checkpoint load
        _X_d = np.random.randn(10, 4).astype(np.float32)
        _y_d = np.array([0, 1] * 5)
        try:
            clf.fit(_X_d, _y_d)
            clf.predict_proba(_X_d[:2])
        except Exception:
            pass

        tabicl_model: nn.Module = clf.model_
        tabicl_model.eval()

        # Freezing logic
        for param in tabicl_model.parameters():
            param.requires_grad = not freeze_backbone
        
        ninp = tabicl_model.col_embedder.embed_dim * tabicl_model.row_interactor.num_cls
        adapter_output_dim = 100#tabicl_model.col_embedder.num_features

        super().__init__(
            ninp=ninp, n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            task_type=task_type, num_events=num_events, use_adapter=use_adapter,
            input_dim=input_dim, batch_norm=batch_norm, 
            adapter_output_dim=adapter_output_dim
        )
        
        self._clf = clf
        self.tabicl = tabicl_model
        
        self.to(device)
        self.device = device
        self.dtype = dtype
        self._move_head_to_last()

    def train(self, mode: bool = True) -> "TabICLSurvModel":
        """TabICL backbone stays in eval mode at all times."""
        super().train(mode)
        self.tabicl.eval()
        return self

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

        device = x_query.device
        x_all = torch.cat([x_context, x_query], dim=0).unsqueeze(0).to(device)
        y_ctx = y_context.unsqueeze(0).to(device)

        _, embs = self.tabicl(x_all, y_train=y_ctx, return_embeddings=True)
        query_embs = embs[0, x_context.shape[0]:].to(self.device)
        query_flat = query_embs.to(torch.float32)

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
            # Dummy logits for TabICL classification output
            cls_logits = torch.zeros(x_query.size(0), 2, device=device)
            return head_out, cls_logits
        return head_out


# ---------------------------------------------------------------------------
# TabICL Survival Wrapper
# ---------------------------------------------------------------------------

class TabICLSurvPH(BaseJointSurvFinetune):
    def __init__(
        self,
        head_type: str = "cox",
        num_durations: int = 100,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        context_size: int = 512,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        alpha: float = 1.0,
        freeze_backbone: bool = True,
        deephit_alpha: float = 0.2,
        deephit_sigma: float = 0.1,
        **kwargs
    ):
        super().__init__(num_durations=num_durations, **kwargs)
        self.task_type = task_type
        self.num_events = num_events
        self.head_type = head_type.lower()
        self.backbone_name = "tabicl"
        self.context_size = context_size
        self.alpha = alpha
        self.deephit_alpha = deephit_alpha
        self.deephit_sigma = deephit_sigma

        n_out = 1 if self.head_type == "cox" else (num_durations * num_events if task_type == "cr" else num_durations)

        self.net = TabICLSurvModel(
            n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            device=self.device, task_type=task_type, num_events=num_events,
            use_adapter=use_adapter, input_dim=input_dim,
            batch_norm=(self.head_type == "deepsurv"),
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        self.model, self.labtrans = self._init_pycox_model(self.head_type, num_durations, self.learning_rate, self.net)
