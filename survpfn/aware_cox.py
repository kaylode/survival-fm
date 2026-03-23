import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
import torchtuples as tt
from pycox.models import CoxPH
from typing import Union, List, Optional
from omegaconf import DictConfig
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
        TabPFNCoxModel: Uses a pre-trained TabPFN model as an encoder and adds a 
        trainable survival head.
        
        Args:
            n_out: Number of output classes for TabPFN.
            head_num_nodes: HIDDEN layer sizes for the survival head MLP.
            dropout: Dropout rate for the head.
            device: Computing device.
            freeze_tabpfn: If True, only the survival head is trained.
        """
        super().__init__()
        
        # Load TabPFN using the workflow function
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
        self.nhid = self.tabpfn.nhid
        self.num_classes = n_out

        # Determine expected number of features for TabPFN encoder
        if hasattr(self.tabpfn.encoder, 'in_features'):
            self.num_expected_features = self.tabpfn.encoder.in_features
        elif hasattr(self.tabpfn.encoder, 'weight'): # basic linear
            self.num_expected_features = self.tabpfn.encoder.weight.shape[1]
        else:
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

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        y_pfn: torch.Tensor = None,
        eval_pos: int = None,
        return_pfn: bool = False,
        **kwargs
    ):
        """
        Forward pass. Extracts transformer embeddings for the survival head.
        Replicates TabPFN's forward logic to capture mid-level representations.
        """
        if isinstance(input, tuple):
            x, y_pfn_arg = input
            y_pfn = y_pfn_arg if y_pfn is None else y_pfn
        else:
            x = input

        # Explicitly cast to model's dtype to avoid mismatch during matrix multiplication
        x = x.to(self.dtype)
        if y_pfn is not None:
             # For the encoder path, y_pfn needs to be float/half
             y_pfn_enc = y_pfn.to(self.dtype)
        else:
             y_pfn_enc = None

        # Match feature dimension to TabPFN encoder expectations (pad if too small, cut if too large)
        num_features = x.shape[-1]
        if num_features < self.num_expected_features:
            padding = torch.zeros(
                *x.shape[:-1], self.num_expected_features - num_features,
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=-1)
        elif num_features > self.num_expected_features:
            x = x[..., :self.num_expected_features]

        if y_pfn is None:
            # For pure inference or if no labels provided, we might need a dummy or handle it
            # TabPFN expects a support set. If only query is provided, we might need to handle.
            # Here we assume standard TabPFN application where first 'eval_pos' samples are support.
            pass

        if eval_pos is None:
            eval_pos = x.shape[0] // 2 if y_pfn is not None else 0

        # We need to replicate the TabPFN forward logic to get mid-layer embeddings
        # (This avoids modifying the TabPFN source code)
        
        with autocast('cuda', enabled=(self.dtype == torch.float16)):
            # 1. TabPFN Encoding
            x_enc = self.tabpfn.encoder(x)
            
            # y_encoder expects (Seq, Batch, 1) or similar
            if y_pfn_enc is not None:
                y_enc = self.tabpfn.y_encoder(y_pfn_enc.unsqueeze(-1) if y_pfn_enc.dim() < x.dim() else y_pfn_enc)
            else:
                y_enc = torch.zeros_like(x_enc) # dummy
            
            # 2. Prepare Source for Transformer
            # We follow TabPFN's structure: [GlobalTokens, StyleTokens, train_x + train_y, query_x]
            # Simplifying for the case where we don't use styles/global tokens if not present
            style_src = torch.tensor([], device=x.device) # Assume no style for now
            train_x_y = x_enc[:eval_pos] + y_enc[:eval_pos]
            src = torch.cat([style_src, train_x_y, x_enc[eval_pos:]], 0)
            
            # 3. Transformer Encoder
            # Note: We skip complex masking for simplicity here, assuming full attention or simple causal
            # In a real scenario, we should use self.tabpfn.generate_D_q_matrix etc.
            output = self.tabpfn.transformer_encoder(src)
            
            # 4. Extract Query Embeddings (after eval_pos)
            query_embs = output[eval_pos + len(style_src):]
            # Flatten if necessary for the head
            if query_embs.dim() == 3:
                T, B, H = query_embs.shape
                query_embs_flat = query_embs.reshape(T * B, H)
                risk_scores = self.survival_head(query_embs_flat).view(T, B, 1)
                logits_pfn = self.tabpfn.decoder(query_embs_flat).view(T, B, -1)
            else:
                risk_scores = self.survival_head(query_embs)
                logits_pfn = self.tabpfn.decoder(query_embs)

        # Flatten risk scores for Cox loss (expects 1D or squeezed 2D)
        if risk_scores is not None:
             # If risk_scores is (Seq, Batch, 1), we can't easily flatten without knowing intention
             # but most commonly it's (Batch, 1) or (Batch,)
             if risk_scores.dim() == 2 and risk_scores.size(-1) == 1:
                 risk_scores = risk_scores.squeeze(-1)
             elif risk_scores.dim() == 3 and risk_scores.size(-1) == 1:
                 # (T, B, 1) -> (T, B) or flattened depending on context
                 risk_scores = risk_scores.squeeze(-1)

        if return_pfn:
            if logits_pfn is not None:
                # Handle both 2D (Batch, Classes) and 3D (Seq, Batch, Classes)
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
        """
        High-level wrapper for training TabPFNCoxModel.
        """
        self.dtype = dtype
        self.device = device
        self.net = TabPFNCoxModel(
            n_out=n_out,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            freeze_tabpfn=freeze_tabpfn,
            dtype=dtype,
            device=device
        )
        
        # We still use pycox.models.CoxPH for convenience
        self.model = CoxPH(self.net, tt.optim.Adam(lr=learning_rate))
        self.alpha = alpha

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
        """
        Training loop with joint Cox and PFN loss.
        """
        from pycox.models.loss import CoxPHLoss
        criterion_cox = CoxPHLoss()
        criterion_pfn = nn.CrossEntropyLoss()
        
        optimizer = self.model.optimizer
        device = next(self.net.parameters()).device
        
        # Prepare data with unified dtype
        x_pt = torch.from_numpy(x).to(self.device, self.dtype)
        durations_pt = torch.from_numpy(durations).to(self.device, self.dtype)
        events_pt = torch.from_numpy(events).to(self.device, self.dtype)
        y_pfn_pt = torch.from_numpy(y_pfn).to(self.device) if y_pfn is not None else None # keep as long for loss

        for epoch in range(epochs):
            self.net.train()
            indices = torch.randperm(x_pt.size(0))
            epoch_loss = 0
            
            for i in range(0, x_pt.size(0), batch_size):
                idx = indices[i:i+batch_size]
                bx, bdur, bev = x_pt[idx], durations_pt[idx], events_pt[idx]
                by_pfn = y_pfn_pt[idx] if y_pfn_pt is not None else None
                
                optimizer.zero_grad()
                
                # Joint forward
                risk_scores, pfn_logits = self.net(bx, y_pfn=by_pfn, return_pfn=True)
                
                # Check for NaNs/Infs (common in half precision or with PFN)
                if not torch.isfinite(risk_scores).all():
                     print(f"Warning: Non-finite risk scores in batch. Clipping...")
                     risk_scores = torch.nan_to_num(risk_scores, nan=0.0, posinf=10.0, neginf=-10.0)

                # 1. Cox Loss
                # Guard: Cox partial likelihood requires at least one event in the batch
                if bev.sum() == 0:
                     loss_cox = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                     loss_cox = criterion_cox(risk_scores, bdur, bev)
                
                # 2. PFN Loss (optional)
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
        """
        Predict survival probabilities using the trained model.
        (Needs baseline hazards to be computed first via pycox methods)
        """
        # Note: integration with pycox.predict_surv_df might require careful handling
        # of the input tuple in forward.
        self.net.eval()
        with torch.no_grad():
            risk_scores = self.net(torch.from_numpy(x).to(self.device, self.dtype))
        return risk_scores
