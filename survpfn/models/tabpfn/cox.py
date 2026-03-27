"""
survpfn.models.tabpfn.cox — TabPFN-aware and embedding-based Cox PH models.

Merged from: aware_cox.py + embedding_cox.py

Classes / functions
-------------------
* TabPFNSurvModel   — PyTorch nn.Module with TabPFN backbone + survival head
* TabPFNSurvPH      — High-level wrapper with fit / predict_survival
* TabPFNCoxPH       — Alias for TabPFNSurvPH(head_type='cox')
* MLPVanilla        — Custom MLP with Kaiming init (from embedding_cox.py)
* EmbeddingCoxPH    — Cox PH fitted on TabPFN embeddings
* train_embedding_cox — end-to-end helper (extract embeddings + fit EmbeddingCoxPH)
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
            self.labtrans = LabTransDiscreteTime(num_durations)
            self.model = DeepHitSingle(self.net, tt.optim.Adam(lr=learning_rate), duration_index=self.labtrans.cuts)
        elif self.head_type == "pchazard":
            self.labtrans = PCHazard.label_transform(num_durations)
            self.model = PCHazard(self.net, tt.optim.Adam(lr=learning_rate), duration_index=self.labtrans.cuts)
        elif self.head_type == "mtlr":
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

        x_pt = torch.from_numpy(x).to(self.device, self.dtype)
        
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
            dur_pt = torch.from_numpy(durations).to(self.device, self.dtype)
            ev_pt = torch.from_numpy(events).to(self.device, self.dtype)
            frac_pt = None
            
        y_pfn_pt = torch.from_numpy(y_pfn).to(self.device) if y_pfn is not None else None

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

# Alias for backward compatibility
class TabPFNCoxModel(TabPFNSurvModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TabPFNCoxPH(TabPFNSurvPH):
    def __init__(self, **kwargs):
        super().__init__(head_type="cox", **kwargs)


# ---------------------------------------------------------------------------
# Embedding-based Cox PH (from embedding_cox.py)
# ---------------------------------------------------------------------------

class MLPVanilla(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_nodes: list,
        out_features: int,
        batch_norm: bool = True,
        dropout: float = None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias: bool = True,
    ):
        super().__init__()

        nodes = [in_features] + list(num_nodes)
        layers = []

        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes[i + 1]))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

        # Output layer — NO batch norm, NO dropout
        out_layer = nn.Linear(nodes[-1], out_features, bias=output_bias)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity='relu')
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

        # Init hidden layer weights
        self._init_weights()

    def _init_weights(self):
        layers = list(self.net.children())
        output_layer = layers[-1] if not isinstance(layers[-1], nn.Module.__class__) else None

        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear) and m is not list(self.net.children())[-1]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingCoxPH:
    def __init__(
        self,
        embedding_dim: int,
        num_nodes: list = [64, 64],
        out_features: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        net = MLPVanilla(
            in_features=embedding_dim,
            num_nodes=num_nodes,
            out_features=out_features,
            batch_norm=batch_norm,
            dropout=dropout,
            output_activation=None,
        )

        self.model = CoxPH(net, tt.optim.Adam(lr=learning_rate))

        # Store training data for compute_baseline_hazards
        self._x_train = None
        self._y_train = None

    def fit(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: tuple = None,
        epochs: int = 100,
        batch_size: int = 256,
        callbacks=None,
        verbose: bool = True,
    ):
        x = embeddings.astype(np.float32)
        y = (durations.astype(np.float32), events.astype(np.float32))

        self._x_train = x
        self._y_train = y

        val = None
        if val_data is not None:
            x_val, dur_val, ev_val = val_data
            val = (
                x_val.astype(np.float32),
                (dur_val.astype(np.float32), ev_val.astype(np.float32)),
            )

        self.log = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            val_data=val,
        )
        return self

    def compute_baseline(self):
        if self._x_train is None:
            raise RuntimeError("Call fit() before compute_baseline().")

        self.model.compute_baseline_hazards(
            input=self._x_train,
            target=self._y_train,
        )
        return self

    def predict_survival(self, embeddings: np.ndarray) -> "pd.DataFrame":
        """
        Returns a DataFrame (timepoints x subjects) with survival probabilities.
        Expected values: float in (0, 1].
        """
        if self.model.baseline_hazards_ is None:
            raise RuntimeError("Call compute_baseline() before predict_survival().")

        x = embeddings.astype(np.float32)
        return self.model.predict_surv_df(x)

    def concordance_index(
        self,
        embeddings: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        method: str = "antolini",
    ) -> float:
        """
        Compute the concordance index on the provided set.

        Parameters
        ----------
        embeddings : np.ndarray
            Feature matrix (n_samples, embedding_dim)
        durations : np.ndarray
            Observation times
        events : np.ndarray
            Event indicator (1 = event, 0 = censored)
        method : str
            'antolini' for time-dependent C-index (default),
            or other methods supported by EvalSurv.

        Returns
        -------
        float
            Concordance index in [0, 1]. 0.5 = random, 1.0 = perfect.
        """
        surv_df = self.predict_survival(embeddings)

        ev = EvalSurv(
            surv=surv_df,
            durations=durations,
            events=events,
            censor_surv="km",
        )

        return ev.concordance_td(method=method)


def train_embedding_cox(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, save_dir="results", study_id=None):
    from survpfn.models.tabpfn.embedding import get_tabpfn_embeddings
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    X_train_raw = df_train.drop(columns=[duration_col, event_col])
    t_train = df_train[duration_col].values.astype(np.float32)
    e_train = df_train[event_col].values.astype(np.float32)

    X_test_raw = df_test.drop(columns=[duration_col, event_col])
    t_test = df_test[duration_col].values.astype(np.float32)
    e_test = df_test[event_col].values.astype(np.float32)

    # Binarize event for TabPFN classifier (competing risks have values > 1)
    y_train_binary = (df_train[event_col] > 0).astype(int)
    y_test_binary = (df_test[event_col] > 0).astype(int)

    print("Generating TabPFN Embeddings...")
    train_emb, test_emb = get_tabpfn_embeddings(X_train_raw, y_train_binary, X_test_raw, y_test_binary)

    params = {'lr': 1e-3, 'nodes': [128, 64], 'dropout': 0.1, 'batch_size': 128}

    if tune:
        X_tr, X_val, T_tr, T_val, E_tr, E_val = train_test_split(
            train_emb, t_train, e_train, test_size=0.2, random_state=42, stratify=e_train
        )

        os.makedirs(save_dir, exist_ok=True)
        log_name = f"optuna_embedding_cox_{study_id}.log" if study_id else "optuna_embedding_cox.log"
        log_file = os.path.join(save_dir, log_name)
        storage = JournalStorage(JournalFileBackend(log_file))
        
        study_name = f"embedding_cox_tuning_{study_id}" if study_id else "embedding_cox_tuning"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        _nodes_map = {"64-64": [64, 64], "128-64": [128, 64], "256-128": [256, 128]}

        def objective(trial):
            lr_trial = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            dropout_trial = trial.suggest_float('dropout', 0.0, 0.5)
            nodes_key = trial.suggest_categorical('nodes', list(_nodes_map.keys()))
            nodes_trial = _nodes_map[nodes_key]

            model = EmbeddingCoxPH(
                embedding_dim=train_emb.shape[1],
                num_nodes=nodes_trial,
                dropout=dropout_trial,
                learning_rate=lr_trial
            )
            model.fit(X_tr, T_tr, E_tr, epochs=50, batch_size=params['batch_size'], verbose=False)
            model.compute_baseline()
            return model.concordance_index(X_val, T_val, E_val)

        study.optimize(objective, n_trials=n_trials)
        params.update(study.best_params)

    model = EmbeddingCoxPH(
        embedding_dim=train_emb.shape[1],
        num_nodes=params['nodes'],
        dropout=params['dropout'],
        learning_rate=params['lr']
    )
    model.fit(train_emb, t_train, e_train, epochs=200, batch_size=params['batch_size'], verbose=True)
    model.compute_baseline()

    surv_df = model.predict_survival(test_emb)
    # Align risk scores: higher hazard = lower survival
    risk_scores = -surv_df.iloc[-1].values

    surv_times = surv_df.index.values
    surv_probs = surv_df.values.T

    return model, risk_scores, surv_probs, surv_times
