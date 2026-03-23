import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
import torchtuples as tt
from pycox.models import CoxPH
from typing import Union, List, Optional
import pathlib

# Import the loader function from the relative path
from survpfn.tabpfn_modeling.utils import load_model_workflow

class TabPFNCoxModel(nn.Module):
    def __init__(
        self,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        device: str = "cuda:0",
        freeze_tabpfn: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        TabPFNCoxModel: Uses a pre-trained TabPFN model and adds a survival head.
        Uses a forward hook to capture transformer embeddings for the survival task.
        """
        super().__init__()
        
        # Load TabPFN
        base_path = pathlib.Path(__file__).parent
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
        self.num_classes = n_out
        self.num_expected_features = self.config.get('num_features', 100)

        # Survival Head (MLP) mapping transformer output (ninp) -> Risk Score (1)
        nodes = [self.ninp] + list(head_num_nodes)
        layers = []
        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(nodes[-1], 1, bias=False))
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
             # Provide dummy y_pfn to avoid AttributeError in TabPFN encoder
             y_pfn = torch.zeros(x.shape[0], device=x.device, dtype=self.dtype)

        # Match feature dimension
        num_features = x.shape[-1]
        if num_features < self.num_expected_features:
            padding = torch.zeros(*x.shape[:-1], self.num_expected_features - num_features, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=-1)
        elif num_features > self.num_expected_features:
            x = x[..., :self.num_expected_features]

        if eval_pos is None:
            # For survival tasks, we typically want risk scores for the entire batch.
            # Setting eval_pos to 0 treats the entire block as the query/target set.
            eval_pos = 0

        # TabPFN forward pass (PFN classification + logic)
        # Replaces manual sequence construction
        with autocast('cuda', enabled=(self.dtype == torch.float16)):
            logits_pfn = self.tabpfn((None, x, y_pfn), single_eval_pos=eval_pos)
        # Capture the query embeddings from the transformer (stored by the hook)
        # output is (Seq, Batch, Hidden). Query is at [eval_pos:]
        query_embs = self._transformer_output[eval_pos:]
        
        # Survival forward
        if query_embs.dim() == 3:
            T, B, H = query_embs.shape
            risk_scores = self.survival_head(query_embs.reshape(T*B, H)).view(T, B)
        else:
            risk_scores = self.survival_head(query_embs).squeeze(-1)

        if return_pfn:
            # Squeeze PFN logits if 3D (Seq, Batch, Classes)
            if logits_pfn.dim() == 3:
                logits_pfn = logits_pfn[:, :, :self.num_classes]
            else:
                logits_pfn = logits_pfn[:, :self.num_classes]
            return risk_scores, logits_pfn
        
        return risk_scores

class TabPFNCoxPH:
    def __init__(
        self,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        dropout: float = 0.2,
        freeze_tabpfn: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0"
    ):
        self.dtype = dtype
        self.device = device
        self.alpha = alpha
        self.net = TabPFNCoxModel(
            n_out=n_out, head_num_nodes=head_num_nodes, dropout=dropout,
            freeze_tabpfn=freeze_tabpfn, dtype=dtype, device=device
        )
        self.model = CoxPH(self.net, tt.optim.Adam(lr=learning_rate))

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        y_pfn: np.ndarray = None, 
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        from pycox.models.loss import CoxPHLoss
        criterion_cox = CoxPHLoss()
        criterion_pfn = nn.CrossEntropyLoss()
        optimizer = self.model.optimizer
        
        x_pt = torch.from_numpy(x).to(self.device, self.dtype)
        durations_pt = torch.from_numpy(durations).to(self.device, self.dtype)
        events_pt = torch.from_numpy(events).to(self.device, self.dtype)
        y_pfn_pt = torch.from_numpy(y_pfn).to(self.device) if y_pfn is not None else None

        for epoch in range(epochs):
            self.net.train()
            indices = torch.randperm(x_pt.size(0))
            epoch_loss = 0
            for i in range(0, x_pt.size(0), batch_size):
                idx = indices[i:i+batch_size]
                bx, bdur, bev = x_pt[idx], durations_pt[idx], events_pt[idx]
                by_pfn = y_pfn_pt[idx] if y_pfn_pt is not None else None
                optimizer.zero_grad()
                
                risk_scores, pfn_logits = self.net(bx, y_pfn=by_pfn, return_pfn=True)
                
                if not torch.isfinite(risk_scores).all():
                     risk_scores = torch.nan_to_num(risk_scores, nan=0.0, posinf=10.0, neginf=-10.0)

                if bev.sum() == 0:
                     loss_cox = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                     loss_cox = criterion_cox(risk_scores, bdur, bev)
                
                loss_pfn = 0
                if pfn_logits is not None and by_pfn is not None:
                     loss_pfn = criterion_pfn(pfn_logits.view(-1, pfn_logits.size(-1)), by_pfn.view(-1).long())
                
                total_loss = loss_cox + self.alpha * loss_pfn
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss {epoch_loss / (max(1, x_pt.size(0)//batch_size)):.4f}")
        return self

    def predict_survival(self, x: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            risk_scores = self.net(torch.from_numpy(x).to(self.device, self.dtype))
        return risk_scores
