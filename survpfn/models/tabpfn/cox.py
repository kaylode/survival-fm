"""
survpfn.models.tabpfn.cox — TabPFN-aware and embedding-based Cox PH models.

Merged from: aware_cox.py + embedding_cox.py

Classes / functions
-------------------
* TabPFNCoxModel    — PyTorch nn.Module with TabPFN backbone + survival head
* TabPFNCoxPH       — High-level wrapper with fit / predict_survival
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
# TabPFN-aware Cox model (from aware_cox.py)
# ---------------------------------------------------------------------------

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
            risk_scores = self.survival_head(query_embs.reshape(T * B, H)).view(T, B)
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
                idx = indices[i:i + batch_size]
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
                print(f"Epoch {epoch}: Average Loss {epoch_loss / (max(1, x_pt.size(0) // batch_size)):.4f}")
        return self

    def predict_survival(self, x: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            risk_scores = self.net(torch.from_numpy(x).to(self.device, self.dtype))
        return risk_scores


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


def train_embedding_cox(df_train, df_test, duration_col, event_col, tune=False, n_trials=10, save_dir="results"):
    from survpfn.models.tabpfn.embedding import get_tabpfn_embeddings
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage

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
        log_file = os.path.join(save_dir, "optuna_embedding_cox.log")
        storage = JournalStorage(JournalFileStorage(log_file))

        study = optuna.create_study(
            study_name="embedding_cox_tuning",
            direction="maximize",
            storage=storage,
            load_if_exists=True
        )

        def objective(trial):
            lr_trial = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            dropout_trial = trial.suggest_float('dropout', 0.0, 0.5)
            nodes_trial = trial.suggest_categorical('nodes', [[64, 64], [128, 64], [256, 128]])

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
