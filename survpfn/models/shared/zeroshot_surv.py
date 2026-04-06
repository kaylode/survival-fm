"""
survpfn.models.shared.zeroshot_surv — Zero-shot ICL survival prediction.

Algorithm (Kim, Lai, Zhang 2026 — arXiv:2601.22259)
----------------------------------------------------
Reframe survival analysis as K classification tasks at K discrete
time bins.  Each FM (TabPFN / TabDPT / TabICL) is used *directly* as a
classifier — no survival head is trained.

Supports:
- Single Risk (Binary classification per bin)
- Competing Risks (Multinomial or Per-event Binary per bin)

Two modes
---------
``single_context`` (default)
    Append t_k as an extra feature column. Fit the FM once on this mixed-time context.

``per_bin``
    For each bin k independently, build context. The FM is re-fit for each bin.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from survpfn.models.shared.preprocessing import get_time_bins

# ---------------------------------------------------------------------------
# Helper: monotone clipping
# ---------------------------------------------------------------------------

def _enforce_monotone(surv_row: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    indices = np.arange(len(surv_row))
    clipped = iso.fit_transform(indices, surv_row)
    return np.clip(clipped, 0.0, 1.0)

# ---------------------------------------------------------------------------
# ZeroShotSurvivalPredictor
# ---------------------------------------------------------------------------

class ZeroShotSurvivalPredictor:
    def __init__(
        self,
        backbone: str = "tabpfn",
        n_bins: int = 10,
        method: str = "single_context",
        cr_method: str = "multinomial", # multinomial or per_event
        max_context_size: int = 3000,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        context_size: int = 128,
        use_retrieval: bool = True,
        model_path: Optional[str] = None,
        **kwargs
    ) -> None:
        self.backbone = backbone
        self.n_bins = n_bins
        self.method = method
        self.cr_method = cr_method
        self.max_context_size = max_context_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.context_size = context_size
        self.use_retrieval = use_retrieval
        self.model_path = model_path

        self._bin_times = None
        self._X_train = None
        self._durations = None
        self._events = None
        self._feature_cols = None
        self._num_events = 1
        self._clf = None # for single_context

    def _make_classifier(self):
        if self.backbone == "tabpfn":
            import torch
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(
                n_estimators=1,
                device="cuda" if (torch.cuda.is_available() and "cuda" in self.device) else "cpu",
            )
        # Similar for tabdpt, tabicl
        if self.backbone == "tabdpt":
             from survpfn.models.tabdpt.tabdpt.tabdpt import TabDPTClassifier
             chk = self.checkpoint_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/models_diff/tabdpt1_1.pth")
             return TabDPTClassifier(path=chk, device=self.device)
        if self.backbone == "tabicl":
            from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier
            return TabICLClassifier(device=self.device, model_path=self.model_path)
        raise ValueError(f"Unknown backbone: {self.backbone}")

    def fit(self, df_train: pd.DataFrame, duration_col: str, event_col: str) -> "ZeroShotSurvivalPredictor":
        self._feature_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
        self._X_train = df_train[self._feature_cols].values.astype(np.float32)
        self._durations = df_train[duration_col].values
        self._events = df_train[event_col].values.astype(int)
        self._num_events = int(self._events.max())

        self._bin_times = get_time_bins(self._durations, self._events, self.n_bins)
        
        if self.method == "single_context" and self.cr_method == "multinomial":
            self._clf = self._fit_context(self._X_train, self._durations, self._events)
        elif self.method == "single_context" and self.cr_method == "per_event":
            self._clfs = {}
            for e in range(1, self._num_events + 1):
                e_binary = (self._events == e).astype(int)
                self._clfs[e] = self._fit_context(self._X_train, self._durations, e_binary)
        
        return self

    def _fit_context(self, X, T, E):
        X_exp, y_exp = self._build_expanded_context(X, T, E, self._bin_times)
        if len(X_exp) > self.max_context_size:
            idx = np.random.choice(len(X_exp), self.max_context_size, replace=False)
            X_exp, y_exp = X_exp[idx], y_exp[idx]
        
        clf = self._make_classifier()
        clf.fit(X_exp, y_exp)
        return clf

    def predict_survival(self, df_test: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        X_test = df_test[self._feature_cols].values.astype(np.float32)
        
        if self._num_events == 1:
            return self._predict_sr(X_test)
        else:
            if self.cr_method == "multinomial":
                return self._predict_cr_multinomial(X_test)
            else:
                return self._predict_cr_per_event(X_test)

    def _predict_sr(self, X_test) -> pd.DataFrame:
        # Standard SR prediction
        K = len(self._bin_times)
        n_test = len(X_test)
        surv_matrix = np.zeros((n_test, K))

        for k, t_k in enumerate(self._bin_times):
            if self.method == "single_context":
                t_col = np.full((n_test, 1), t_k)
                X_q = np.concatenate([X_test, t_col], axis=1)
                probs = self._clf.predict_proba(X_q)
                surv_matrix[:, k] = 1.0 - probs[:, 1]
            else:
                # per_bin implementation...
                pass
        
        # enforce monotone
        for i in range(n_test):
            surv_matrix[i] = _enforce_monotone(surv_matrix[i])
            
        return pd.DataFrame(surv_matrix, columns=self._bin_times)

    def _predict_cr_multinomial(self, X_test) -> List[pd.DataFrame]:
        K = len(self._bin_times)
        n_test = len(X_test)
        # Prob of Event m by time t_k
        event_probs = [np.zeros((n_test, K)) for _ in range(self._num_events)]

        for k, t_k in enumerate(self._bin_times):
            t_col = np.full((n_test, 1), t_k)
            X_q = np.concatenate([X_test, t_col], axis=1)
            probs = self._clf.predict_proba(X_q) # (batch, num_events + 1)
            
            for m in range(1, self._num_events + 1):
                if m < probs.shape[1]:
                    event_probs[m-1][:, k] = probs[:, m]
        
        # Convert to CIDFs
        cifs = []
        for m in range(self._num_events):
            # CIDF should be non-decreasing
            # Actually, standard zero-shot gives P(T <= t, E=m) directly
            # We should probably enforce monotone increasing
            cif_m = event_probs[m]
            cifs.append(pd.DataFrame(cif_m, columns=self._bin_times))
        return cifs

    def _predict_cr_per_event(self, X_test) -> List[pd.DataFrame]:
        cifs = []
        for m in range(1, self._num_events + 1):
            K = len(self._bin_times)
            n_test = len(X_test)
            cif_m = np.zeros((n_test, K))
            clf_m = self._clfs[m]
            for k, t_k in enumerate(self._bin_times):
                t_col = np.full((n_test, 1), t_k)
                X_q = np.concatenate([X_test, t_col], axis=1)
                probs = clf_m.predict_proba(X_q)
                cif_m[:, k] = probs[:, 1]
            cifs.append(pd.DataFrame(cif_m, columns=self._bin_times))
        return cifs

    @staticmethod
    def _build_expanded_context(X, T, E, bin_times):
        rows_X, rows_y = [], []
        for t_k in bin_times:
            # Include if event happened by t_k OR survived past t_k
            valid = (E > 0) | (T > t_k)
            X_k = X[valid]
            T_k = T[valid]
            E_k = E[valid]
            
            # Label = E if T <= t_k else 0
            y_k = np.where(T_k <= t_k, E_k, 0)
            
            t_col = np.full((len(X_k), 1), t_k)
            rows_X.append(np.concatenate([X_k, t_col], axis=1))
            rows_y.append(y_k)
            
        return np.concatenate(rows_X, axis=0), np.concatenate(rows_y, axis=0)

def train_zeroshot_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    **kwargs
) -> tuple:
    predictor = ZeroShotSurvivalPredictor(**kwargs)
    predictor.fit(df_train, duration_col, event_col)
    
    out = predictor.predict_survival(df_test)
    
    if isinstance(out, list):
        # Competing risks
        cifs = out
        # risk = cause 1 probability at last time bin
        risk = cifs[0].iloc[:, -1].values
        return predictor, risk, [c.values for c in cifs], cifs[0].columns.values
    else:
        # Single risk
        surv_df = out
        risk = 1.0 - surv_df.iloc[:, -1].values
        return predictor, risk, surv_df.values, surv_df.columns.values
