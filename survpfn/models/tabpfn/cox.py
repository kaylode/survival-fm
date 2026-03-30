"""
survpfn.models.tabpfn.cox — TabPFN-aware and embedding-based survival models.

Merged from: aware_cox.py + embedding_cox.py

Classes / functions
-------------------
* TabPFNSurvModel     — PyTorch nn.Module with TabPFN backbone + survival head
                        (jointly-trained variant; supports cox/deephit/pchazard/mtlr)
* TabPFNSurvPH        — High-level wrapper with fit / predict_survival
* MLPVanilla          — Re-exported from survpfn.models.heads (backward compat)
* EmbeddingCoxPH      — Re-exported from survpfn.models.heads (backward compat)
* train_tabpfn_embedding_cox — Thin wrapper over train_fm_embedding_surv (head_type="cox")
* train_embedding_surv — Full wrapper supporting all four head types
"""

import os
import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from typing import Union, List, Optional
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

from survpfn.models.tabpfn.backbone.utils import load_model_workflow

# Re-export shared head utilities so existing imports continue to work.
from survpfn.models.heads import (  # noqa: F401
    MLPVanilla,
    EmbeddingCoxPH,
    EmbeddingSurvHead,
    train_fm_embedding_surv,
)


# ---------------------------------------------------------------------------
# TabPFN Survival model
# ---------------------------------------------------------------------------

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
        n_pfn_classes: int = 10,  # Default to TabPFN's usual 10-class capacity
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
        self.num_classes = n_pfn_classes
        self.num_expected_features = self.config.get('num_features', 100)

        # Survival Head (MLP) mapping transformer output (ninp) -> Risk Score or Bins
        nodes = [self.ninp] + list(head_num_nodes)
        layers = []
        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        # Output layer dimension depends on the head (Cox: 1, Discrete: num_durations)
        layers.append(nn.Linear(nodes[-1], n_out, bias=False))
        self.survival_head = nn.Sequential(*layers)

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

        # Match feature dimension
        num_features = x.shape[-1]
        if num_features < self.num_expected_features:
            padding = torch.zeros(*x.shape[:-1], self.num_expected_features - num_features, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        elif num_features > self.num_expected_features:
            x = x[..., :self.num_expected_features]

        if eval_pos is None:
            eval_pos = 0

        # TabPFN forward pass
        with autocast('cuda', enabled=(self.dtype == torch.float16 and 'cuda' in str(self.device))):
            logits_pfn = self.tabpfn((None, x, y_pfn), single_eval_pos=eval_pos)
        
        query_embs = self._transformer_output[eval_pos:]

        # Survival forward
        # If output dim is 1 (Cox), we want (BatchSize,)
        # If output dim > 1 (Discrete), we want (BatchSize, n_out)
        
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
        n_pfn_classes: int = 2
    ):
        self.head_type = head_type.lower()
        self.dtype = dtype
        self.device = device
        self.alpha = alpha
        self.num_durations = num_durations
        
        if n_out is None:
            n_out = 1 if self.head_type == "cox" else num_durations
        
        self.net = TabPFNSurvModel(
            n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            freeze_tabpfn=freeze_tabpfn, dtype=dtype, device=device,
            n_pfn_classes=n_pfn_classes
        )
        
        from pycox.models import CoxPH, PCHazard, MTLR, DeepHitSingle
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
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        y_pfn: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        from pycox.models.loss import CoxPHLoss, DeepHitSingleLoss, NLLPCHazardLoss, NLLMTLRLoss
        from pycox.models.data import pair_rank_mat
        
        if self.head_type == "cox":
            criterion_surv = CoxPHLoss()
        elif self.head_type == "deephit":
            criterion_surv = DeepHitSingleLoss(alpha=0.2, sigma=0.1)
        elif self.head_type == "pchazard":
            criterion_surv = NLLPCHazardLoss()
        elif self.head_type == "mtlr":
            criterion_surv = NLLMTLRLoss()
            
        criterion_pfn = nn.CrossEntropyLoss()
        optimizer = self.model.optimizer

        x_pt = torch.from_numpy(x.copy()).to(self.device, self.dtype)
        
        # Label transforms for discrete models
        if self.labtrans is not None:
            targets = self.labtrans.fit_transform(durations, events)
            dur_pt = torch.from_numpy(targets[0]).to(self.device)
            ev_pt = torch.from_numpy(targets[1]).to(self.device)
            # Some transforms like PCHazard have a third element: interval_frac
            frac_pt = None
            if len(targets) > 2:
                frac_pt = torch.from_numpy(targets[2]).to(self.device, self.dtype)
        else:
            dur_pt = torch.from_numpy(durations.copy()).to(self.device, self.dtype)
            ev_pt = torch.from_numpy(events.copy()).to(self.device, self.dtype)
            frac_pt = None
            
        y_pfn_pt = torch.from_numpy(y_pfn.copy()).to(self.device) if y_pfn is not None else None

        for epoch in range(epochs):
            self.net.train()
            indices = torch.randperm(x_pt.size(0))
            epoch_loss = 0
            for i in range(0, x_pt.size(0), batch_size):
                idx = indices[i:i + batch_size]
                bx, bdur, bev = x_pt[idx], dur_pt[idx], ev_pt[idx]
                bfrac = frac_pt[idx] if frac_pt is not None else None
                by_pfn = y_pfn_pt[idx] if y_pfn_pt is not None else None
                optimizer.zero_grad()

                head_out, pfn_logits = self.net(bx, y_pfn=by_pfn, return_pfn=True)

                if self.head_type == "cox":
                    loss_surv = criterion_surv(head_out, bdur, bev)
                elif self.head_type == "deephit":
                    # DeepHit requires a rank_mat
                    # We might need to ensure it's a tensor on the correct device
                    import numpy as np
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
                optimizer.step()
                epoch_loss += total_loss.item()

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
            if self.head_type == "cox":
                return self.model.predict_surv_df(x_pt)
            else:
                # Discrete models: Check if interpolate is available (e.g., LogisticHazard, MTLR)
                if hasattr(self.model, 'interpolate'):
                    return self.model.interpolate(10).predict_surv_df(x_pt)
                return self.model.predict_surv_df(x_pt)

# ---------------------------------------------------------------------------
# Embedding-based survival heads — thin wrappers over survpfn.models.heads
# ---------------------------------------------------------------------------
# MLPVanilla, EmbeddingCoxPH, and EmbeddingSurvHead are imported at the top
# of this file from survpfn.models.heads and re-exported for backward compat.


def train_embedding_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    head_type: str = "cox",
    tune: bool = False,
    n_trials: int = 10,
    save_dir: str = "results",
    study_id: Optional[str] = None,
) -> tuple:
    """Frozen TabPFN embedding + any survival head.

    Parameters
    ----------
    head_type : one of "cox", "deephit", "pchazard", "mtlr"

    Returns
    -------
    (model, risk_scores, surv_probs, surv_times)
    """
    from survpfn.models.tabpfn.embedding import get_tabpfn_embeddings

    def _embedding_fn(X_train: np.ndarray, y_bin: np.ndarray,
                      X_test: np.ndarray) -> tuple:
        # TabPFN's extractor also accepts test labels (used as a dummy class
        # for the in-context classifier); pass zeros for the test set.
        y_test_zeros = np.zeros(len(X_test), dtype=np.float32)
        return get_tabpfn_embeddings(
            pd.DataFrame(X_train),
            pd.Series(y_bin.astype(int)),
            pd.DataFrame(X_test),
            pd.Series(y_test_zeros.astype(int)),
        )

    return train_fm_embedding_surv(
        df_train, df_test, duration_col, event_col,
        embedding_fn=_embedding_fn,
        head_type=head_type,
        tune=tune,
        n_trials=n_trials,
        save_dir=save_dir,
        study_id=study_id,
        fm_name="tabpfn",
    )


def train_tabpfn_embedding_cox(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    tune: bool = False,
    n_trials: int = 10,
    save_dir: str = "results",
    study_id: Optional[str] = None,
) -> tuple:
    """Backward-compatible alias for train_embedding_surv(head_type='cox')."""
    return train_embedding_surv(
        df_train, df_test, duration_col, event_col,
        head_type="cox",
        tune=tune,
        n_trials=n_trials,
        save_dir=save_dir,
        study_id=study_id,
    )
