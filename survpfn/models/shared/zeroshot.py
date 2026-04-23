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
import sys
import warnings
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from survpfn.models.shared.binning import get_cuts, resolve_num_durations
from survpfn.models.shared.preprocessing import SurvivalTimeBinEncoder

# ---------------------------------------------------------------------------
# Helper: monotone clipping
# ---------------------------------------------------------------------------

def _enforce_monotone(row: np.ndarray, increasing: bool = False) -> np.ndarray:
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    indices = np.arange(len(row))
    clipped = iso.fit_transform(indices, row)
    return np.clip(clipped, 0.0, 1.0)

# ---------------------------------------------------------------------------
# ZeroShotSurvivalPredictor
# ---------------------------------------------------------------------------

class ZeroShotSurvivalPredictor:
    def __init__(
        self,
        backbone: str = "tabpfn",
        n_bins: int = -1,
        method: str = "single_context",
        cr_method: str = "multinomial", # multinomial or per_event
        max_context_size: int = 512,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        context_size: int = 512,
        use_retrieval: bool = True,
        model_path: Optional[str] = None,
        use_time_bin_encoder: bool = False,
        n_estimators: int = 1,
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
        self.use_time_bin_encoder = use_time_bin_encoder
        self.n_estimators = n_estimators
        self.verbose = kwargs.get("verbose", True)
        self.binning_scheme = kwargs.get("binning_scheme", "quantile")
        self.early_event_quantile = kwargs.get("early_event_quantile", 0.5)
        self.early_bin_frac = kwargs.get("early_bin_frac", 0.7)

        self._bin_times = None
        self._X_train = None
        self._durations = None
        self._events = None
        self._feature_cols = None
        self._num_events = 1
        self._clf = None          # for single_context SR / multinomial CR
        self._clfs = None         # for per_event single_context CR
        self._clfs_per_bin = None # for per_bin mode
        self._time_encoder = None  # SurvivalTimeBinEncoder instance
        # PCA fitted ONCE globally in fit(); never overwritten after that.
        # All _fit_context calls and all _predict_* calls use the same object.
        self._pca = None
        self._pca_n_components = None  # stored for diagnostics
        self._shared_clf = None

    def _make_classifier(self):
        if self.backbone == "tabpfn":
            import torch
            from survpfn.models.tabpfn.backbone_v2 import get_classifier

            # Respect self.device — fall back to CPU when CUDA is unavailable or
            # the caller requested CPU (BUG-H3: was hardcoded to "cuda").
            _tabpfn_device = (
                "cuda"
                if ("cuda" in self.device and torch.cuda.is_available())
                else "cpu"
            )
            return get_classifier(device=_tabpfn_device)

        if self.backbone == "tabdpt":
             from survpfn.models.tabdpt.tabdpt.tabdpt import TabDPTClassifier
             chk = self.checkpoint_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/models_diff/tabdpt1_1.pth")
             return TabDPTClassifier(path=chk, device=self.device)
        if self.backbone == "tabicl":
            from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier
            return TabICLClassifier(device=self.device, model_path=self.model_path)
        raise ValueError(f"Unknown backbone: {self.backbone}")

    def _get_clf_instance(self):
        if self._shared_clf is None:
            self._shared_clf = self._make_classifier()
        return self._shared_clf

    def fit(self, df_train: pd.DataFrame, duration_col: str, event_col: str) -> "ZeroShotSurvivalPredictor":
        self._feature_cols = [c for c in df_train.columns if c not in {duration_col, event_col}]
        self._X_train = df_train[self._feature_cols].values.astype(np.float32)
        self._durations = df_train[duration_col].values
        self._events = df_train[event_col].values.astype(int)
        self._num_events = int(self._events.max())

        n_bins = resolve_num_durations(self._durations, self._events, self.n_bins)
        if self.use_time_bin_encoder:
            self._time_encoder = SurvivalTimeBinEncoder(
                n_bins=n_bins, scheme=self.binning_scheme,
                early_event_quantile=self.early_event_quantile,
                early_bin_frac=self.early_bin_frac,
            )
            self._time_encoder.fit(self._durations, self._events)
            self._bin_times = self._time_encoder.bin_times
        else:
            self._time_encoder = None
            # Right edges of quantile bins (prediction time points)
            self._bin_times = get_cuts(
                self._durations, self._events, n_bins, scheme=self.binning_scheme,
                early_event_quantile=self.early_event_quantile,
                early_bin_frac=self.early_bin_frac,
            )#[1:]

        if self.verbose:
            print(f"  [{self.backbone}/ZeroShot] Method: {self.method}, Bins: {len(self._bin_times)}, Estimators: {self.n_estimators}", flush=True)

        # PCA Feature Reduction: Fit once globally if features exceed TabPFN limit (2000)
        # This prevents state overwriting during multi-bin or multi-event fitting.
        n_temp_features = 6 if self.use_time_bin_encoder else 1
        if len(self._feature_cols) + n_temp_features > 2000:
            print(f"Number of features {len(self._feature_cols) + n_temp_features} exceeds TabPFN limit. Fitting global PCA...")
            # We build a global expanded context to capture feature distribution across time
            X_pca_sample, _ = self._build_expanded_context(self._X_train, self._durations, self._events, self._bin_times, encoder=self._time_encoder)
            n_samples, n_features = X_pca_sample.shape
            n_components = min(n_samples, n_features, 2000)
            self._pca = PCA(n_components=n_components, random_state=42)
            self._pca.fit(X_pca_sample)
            print(f"Global PCA fitted: {n_features} -> {n_components} components.")

        if self.method == "single_context":
            if self.verbose: print(f"  [{self.backbone}/ZeroShot] Building context (single-context mode)...", flush=True)
            if self._num_events == 1 or self.cr_method == "multinomial":
                self._clf = self._fit_context(self._X_train, self._durations, self._events)
            else: # per_event
                self._clfs = {}
                for e in range(1, self._num_events + 1):
                    e_binary = (self._events == e).astype(int)
                    self._clfs[e] = self._fit_context(self._X_train, self._durations, e_binary)
        else: # per_bin
            self._clfs_per_bin = {}
            pbar = tqdm(enumerate(self._bin_times), total=len(self._bin_times), desc=f"  [{self.backbone}/ZeroShot] Fitting per-bin context", file=sys.stdout, leave=False, disable=not self.verbose)
            for k, t_k in pbar:
                if self._num_events == 1 or self.cr_method == "multinomial":
                    self._clfs_per_bin[k] = self._fit_context(self._X_train, self._durations, self._events, bin_times=[t_k])
                else: # per_event
                    self._clfs_per_bin[k] = {}
                    for e in range(1, self._num_events + 1):
                        e_binary = (self._events == e).astype(int)
                        self._clfs_per_bin[k][e] = self._fit_context(self._X_train, self._durations, e_binary, bin_times=[t_k])
        
        return self

    def _fit_context(self, X, T, E, bin_times=None):
        if bin_times is None:
            bin_times = self._bin_times
        X_exp, y_exp = self._build_expanded_context(X, T, E, bin_times, encoder=self._time_encoder)
        
        # Ensure class diversity: if a bin context is missing labels, add 'anchors' from global context
        unique_y = np.unique(y_exp)
        target_classes = self._num_events + 1
        if len(bin_times) == 1 and len(unique_y) < target_classes:
            X_global, y_global = self._build_expanded_context(X, T, E, self._bin_times, encoder=self._time_encoder)
            missing = [c for c in range(target_classes) if c not in unique_y]
            anchors_X, anchors_y = [], []
            for c in missing:
                idx_c = np.where(y_global == c)[0]
                if len(idx_c) > 0:
                    # Pull up to 25 anchor points from global multi-bin context for each missing class
                    sel = np.random.choice(idx_c, min(len(idx_c), 25), replace=False)
                    anchors_X.append(X_global[sel])
                    anchors_y.append(y_global[sel])
            if anchors_X:
                X_exp = np.concatenate([X_exp] + anchors_X, axis=0)
                y_exp = np.concatenate([y_exp] + anchors_y, axis=0)

        if self._pca is not None:
            X_exp = self._pca.transform(X_exp)

        def _get_context(Xe, ye, random_state=None):
            if len(Xe) > self.max_context_size:
                idx = self._get_stratified_indices(ye, self.max_context_size, random_state=random_state)
                Xe, ye = Xe[idx], ye[idx]
            return (Xe, ye)

        if self.n_estimators <= 1:
            return [_get_context(X_exp, y_exp)]
        else:
            contexts = []
            for i in range(self.n_estimators):
                contexts.append(_get_context(X_exp, y_exp, random_state=i))
            return contexts

    def _get_probs(self, contexts, X, class_idx):
        """Safely extract probability of class_idx. Handles ensemble by sharing one model instance and fitting just-in-time."""
        clf = self._get_clf_instance()
        probs_list = []
        
        for Xe, ye in contexts:
            clf.fit(Xe, ye)
            probs = clf.predict_proba(X)
            
            if hasattr(clf, "classes_"):
                # Find column index for class_idx
                loc = np.where(clf.classes_ == class_idx)[0]
                if len(loc) > 0:
                    # loc[0] is the column index
                    p = probs[:, loc[0]]
                else:
                    p = np.zeros(len(X))
            else:
                # Fallback to direct indexing if classes_ is missing
                if probs.shape[1] > class_idx:
                    p = probs[:, class_idx]
                else:
                    p = np.zeros(len(X))
            probs_list.append(p)
            
        return np.mean(probs_list, axis=0)

    def _get_stratified_indices(self, y, n_samples, random_state=None):
        rng = np.random.RandomState(random_state)
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) <= 1:
            return rng.choice(len(y), min(n_samples, len(y)), replace=False)
        
        # Proportional sampling with at least 1 element per class
        n_per_class = (counts / len(y)) * n_samples
        n_per_class = np.floor(n_per_class).astype(int)
        n_per_class = np.maximum(n_per_class, 1)
        
        while n_per_class.sum() > n_samples:
            idx_max = np.argmax(n_per_class)
            n_per_class[idx_max] -= 1
            
        all_indices = []
        for i, cls in enumerate(unique_classes):
            cls_indices = np.where(y == cls)[0]
            sampled = rng.choice(cls_indices, min(n_per_class[i], len(cls_indices)), replace=False)
            all_indices.append(sampled)
            
        current_indices = np.concatenate(all_indices)
        if len(current_indices) < n_samples:
            remaining = np.setdiff1d(np.arange(len(y)), current_indices)
            if len(remaining) > 0:
                fill = rng.choice(remaining, min(n_samples - len(current_indices), len(remaining)), replace=False)
                all_indices.append(fill)
            
        return np.concatenate(all_indices)

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

        pbar = tqdm(enumerate(self._bin_times), total=K, desc=f"  [{self.backbone}/ZeroShot] Predicting", file=sys.stdout, leave=False, disable=not self.verbose)
        for k, t_k in pbar:
            if self.use_time_bin_encoder:
                feat = self._time_encoder.bin_feats[k]
                t_f_col = np.tile(feat, (n_test, 1))
                X_q = np.concatenate([X_test, t_f_col], axis=1)
            else:
                t_col = np.full((n_test, 1), t_k)
                X_q = np.concatenate([X_test, t_col], axis=1)
            
            if self._pca is not None:
                X_q = self._pca.transform(X_q)

            if self.method == "single_context":
                clf = self._clf
            else:
                clf = self._clfs_per_bin[k]

            p1 = self._get_probs(clf, X_q, 1)
            surv_matrix[:, k] = 1.0 - p1

        # ── Interpolation (Unique times only) ──
        times = np.concatenate([[0], self._bin_times])
        surv = np.concatenate([np.ones((n_test, 1)), surv_matrix], axis=1)
        
        # Remove duplicates (e.g., if bin_times already started at 0)
        times, unique_idx = np.unique(times, return_index=True)
        surv = surv[:, unique_idx]

        grid_times = np.linspace(0, self._bin_times.max(), 1000)
        from scipy.interpolate import interp1d
        
        f = interp1d(times, surv, kind='linear', axis=1, fill_value="extrapolate")
        surv_matrix_interp = f(grid_times)
        
        # Final clip
        surv_matrix_interp = np.clip(surv_matrix_interp, 0.0, 1.0)

        # pycox convention: rows = times, cols = patients
        return pd.DataFrame(
            surv_matrix_interp.T,
            index=grid_times,
            columns=np.arange(n_test),
        )

    def _predict_cr_multinomial(self, X_test) -> List[pd.DataFrame]:
        K = len(self._bin_times)
        n_test = len(X_test)
        # Prob of Event m by time t_k
        event_probs = [np.zeros((n_test, K)) for _ in range(self._num_events)]

        pbar = tqdm(enumerate(self._bin_times), total=K, desc=f"  [{self.backbone}/ZeroShot] Predicting (Multinomial CR)", file=sys.stdout, leave=False, disable=not self.verbose)
        for k, t_k in pbar:
            if self.use_time_bin_encoder:
                feat = self._time_encoder.bin_feats[k]
                t_f_col = np.tile(feat, (n_test, 1))
                X_q = np.concatenate([X_test, t_f_col], axis=1)
            else:
                t_col = np.full((n_test, 1), t_k)
                X_q = np.concatenate([X_test, t_col], axis=1)
            
            if self._pca is not None:
                X_q = self._pca.transform(X_q)

            if self.method == "single_context":
                clf = self._clf
            else:
                clf = self._clfs_per_bin[k]
            
            for m in range(1, self._num_events + 1):
                event_probs[m-1][:, k] = self._get_probs(clf, X_q, m)
        
        # ── Interpolation ──
        times = np.concatenate([[0], self._bin_times])
        grid_times = np.linspace(0, self._bin_times.max(), 1000)
        from scipy.interpolate import interp1d
        
        cifs = []
        for m in range(self._num_events):
            cif_m = event_probs[m]
            for i in range(n_test):
                cif_m[i] = _enforce_monotone(cif_m[i], increasing=True)
            
            # Prepend start point and filter unique
            cif_m_ext = np.concatenate([np.zeros((n_test, 1)), cif_m], axis=1)
            t_unique, idx_unique = np.unique(times, return_index=True)
            cif_m_unique = cif_m_ext[:, idx_unique]
            
            # Interpolate CIF starting at 0
            f = interp1d(t_unique, cif_m_unique, kind='linear', axis=1, fill_value="extrapolate")
            cif_m_interp = np.clip(f(grid_times), 0.0, 1.0)
            # pycox convention: rows = times, cols = patients
            cifs.append(pd.DataFrame(cif_m_interp.T, index=grid_times, columns=np.arange(n_test)))
        return cifs

    def _predict_cr_per_event(self, X_test) -> List[pd.DataFrame]:
        cifs = []
        for m in range(1, self._num_events + 1):
            K = len(self._bin_times)
            n_test = len(X_test)
            cif_m = np.zeros((n_test, K))
            
            pbar = tqdm(enumerate(self._bin_times), total=K, desc=f"  [{self.backbone}/ZeroShot] Predicting (Event {m})", file=sys.stdout, leave=False, disable=not self.verbose)
            for k, t_k in pbar:
                if self.use_time_bin_encoder:
                    feat = self._time_encoder.bin_feats[k]
                    t_f_col = np.tile(feat, (n_test, 1))
                    X_q = np.concatenate([X_test, t_f_col], axis=1)
                else:
                    t_col = np.full((n_test, 1), t_k)
                    X_q = np.concatenate([X_test, t_col], axis=1)
                
                if self._pca is not None:
                    X_q = self._pca.transform(X_q)

                if self.method == "single_context":
                    clf = self._clfs[m]
                else:
                    clf = self._clfs_per_bin[k][m]

                cif_m[:, k] = self._get_probs(clf, X_q, 1)
            
            # ── Monotone and Interpolation ──
            for i in range(n_test):
                cif_m[i] = _enforce_monotone(cif_m[i], increasing=True)
            
            # Prepend t=0 with CIF(0)=0.0
            times = np.concatenate([[0], self._bin_times])
            cif_m_ext = np.concatenate([np.zeros((n_test, 1)), cif_m], axis=1)
            
            t_unique, idx_unique = np.unique(times, return_index=True)
            cif_m_unique = cif_m_ext[:, idx_unique]

            grid_times = np.linspace(0, self._bin_times.max(), 1000)
            from scipy.interpolate import interp1d
            f = interp1d(t_unique, cif_m_unique, kind='linear', axis=1, fill_value="extrapolate")
            cif_m_interp = np.clip(f(grid_times), 0.0, 1.0)
            # pycox convention: rows = times, cols = patients
            cifs.append(pd.DataFrame(cif_m_interp.T, index=grid_times, columns=np.arange(n_test)))
        return cifs

    @staticmethod
    def _build_expanded_context(X, T, E, bin_times, encoder=None):
        rows_X, rows_y = [], []
        
        if encoder is not None:
            bin_times_enc = encoder.bin_times
            bin_feats_enc = encoder.bin_feats
            time_to_feat = {float(t): f for t, f in zip(bin_times_enc, bin_feats_enc)}

        for t_k in bin_times:
            # Include if event happened by t_k OR survived past t_k
            valid = (E > 0) | (T > t_k)
            X_k = X[valid]
            T_k = T[valid]
            E_k = E[valid]
            
            # Label = E if T <= t_k else 0
            y_k = np.where(T_k <= t_k, E_k, 0)
            
            if encoder is not None:
                feat = time_to_feat.get(float(t_k))
                if feat is None:
                    # nearest neighbor
                    idx = np.abs(bin_times_enc - t_k).argmin()
                    feat = bin_feats_enc[idx]
                t_f_col = np.tile(feat, (len(X_k), 1))
                rows_X.append(np.concatenate([X_k, t_f_col], axis=1))
            else:
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
    """Fit and predict with ZeroShotSurvivalPredictor.

    Returns
    -------
    (predictor, risk, surv_arrays, grid_times)
        ``risk``        — ``(n_test,)`` risk scores.
        ``surv_arrays`` — SR: ``(n_test, n_times)`` survival matrix.
                          CR: ``List[np.ndarray]``, each ``(n_test, n_times)``.
        ``grid_times``  — ``(n_times,)`` shared time grid.

    DataFrame convention (pycox): rows = times, cols = patients.
    """
    predictor = ZeroShotSurvivalPredictor(**kwargs)
    predictor.fit(df_train, duration_col, event_col)

    out = predictor.predict_survival(df_test)

    if isinstance(out, list):
        # Competing risks — rows = times, cols = patients
        cifs = out
        # CIF_1 at last time = max cumulative incidence = risk score
        risk = cifs[0].iloc[-1].values                          # (n_test,)
        return predictor, risk, [c.values.T for c in cifs], cifs[0].index.values
    else:
        # Single risk — rows = times, cols = patients
        surv_df = out
        risk = 1.0 - surv_df.iloc[-1].values                   # (n_test,)
        return predictor, risk, surv_df.values.T, surv_df.index.values
