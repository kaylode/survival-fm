import os
import torch
import torch.nn as nn
import numpy as np
from torchtuples import practical
import torchtuples as tt
from pycox.models import CoxPH
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split

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

        # Layer di output — NO batch norm, NO dropout
        out_layer = nn.Linear(nodes[-1], out_features, bias=output_bias)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity='relu')
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

        # Init pesi layer nascosti
        self._init_weights()

    def _init_weights(self):
        layers = list(self.net.children())
        output_layer = layers[-1] if not isinstance(layers[-1], nn.Module.__class__) else None

        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear) and m is not list(self.net.children())[-1]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        #for m in self.net:
        #    if isinstance(m, nn.Linear):
        #        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #        nn.init.zeros_(m.bias)

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

        # Salviamo i dati di training per compute_baseline_hazards
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
            raise RuntimeError("Chiama fit() prima di compute_baseline().")

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
        Calcola il concordance index sul set fornito.

        Parameters
        ----------
        embeddings : np.ndarray
            Feature matrix (n_samples, embedding_dim)
        durations : np.ndarray
            Tempi di osservazione
        events : np.ndarray
            Event indicator (1 = evento, 0 = censurato)
        method : str
            'antolini' per il C-index time-dependent (default),
            oppure altri metodi supportati da EvalSurv.

        Returns
        -------
        float
            Concordance index in [0, 1]. 0.5 = random, 1.0 = perfetto.
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
    from survpfn.tabpfn import get_tabpfn_embeddings
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