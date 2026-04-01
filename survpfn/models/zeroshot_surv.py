"""
survpfn.models.zeroshot_surv — Zero-shot ICL survival prediction.

Algorithm (Kim, Lai, Zhang 2026 — arXiv:2601.22259)
----------------------------------------------------
Reframe survival analysis as K binary classification tasks at K discrete
time bins.  Each FM (TabPFN / TabDPT / TabICL) is used *directly* as a
binary classifier — no survival head is trained.

Two modes
---------
``single_context`` (default)
    Append t_k as an extra feature column.  Build one expanded training
    context: for patient i and bin k, include row (X_i ⊕ t_k, Y_{ik})
    whenever t_k < C_i (censoring time), where Y_{ik} = 1(T_i ≤ t_k).
    Fit the FM once on this mixed-time context, then query with (X_test ⊕ t_k)
    for each bin.

``per_bin``
    For each bin k independently, build context = {(X_i, Y_{ik}) : t_k < C_i}.
    No time feature is appended.  The FM is re-fit for each bin.  More accurate
    but ~K× slower.

Public API
----------
    predictor = ZeroShotSurvivalPredictor(backbone="tabpfn")
    predictor.fit(df_train, duration_col="duration", event_col="event")
    surv_df   = predictor.predict_survival(df_test)       # (n_test × K)
    risk      = predictor.predict_risk(df_test)            # (n_test,)
    ci        = predictor.concordance_index(df_test, "duration", "event")

Benchmark-compatible wrapper
----------------------------
    model, risk, surv_probs, surv_times = train_zeroshot_surv(
        df_train, df_test, duration_col, event_col, backbone="tabpfn"
    )
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Helper: monotone clipping of survival curve
# ---------------------------------------------------------------------------

def _enforce_monotone(surv_row: np.ndarray) -> np.ndarray:
    """Clip survival probabilities to be non-increasing via isotonic regression.

    Parameters
    ----------
    surv_row : (K,) array of S(t_1), …, S(t_K)

    Returns
    -------
    clipped : (K,) non-increasing probabilities ∈ [0, 1]
    """
    # IsotonicRegression minimises MSE subject to monotone constraint.
    # We want non-increasing, so negate, fit increasing, then negate back.
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    indices = np.arange(len(surv_row))
    clipped = iso.fit_transform(indices, surv_row)
    return np.clip(clipped, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ZeroShotSurvivalPredictor
# ---------------------------------------------------------------------------

class ZeroShotSurvivalPredictor:
    """Zero-shot ICL survival predictor based on any FM binary classifier.

    Parameters
    ----------
    backbone : "tabpfn" | "tabdpt" | "tabicl"
    n_bins : int
        Number of discrete time bins.  Bin boundaries are placed at
        equally-spaced quantiles of the observed (uncensored) event times.
    method : "single_context" | "per_bin"
        ICL strategy; see module docstring.
    max_context_size : int
        Maximum number of rows passed as context to the FM.  When the
        expanded dataset exceeds this limit the rows are randomly
        subsampled (seed = 0 for reproducibility).
    device : str
        Torch device (e.g. ``"cpu"`` or ``"cuda:0"``).
    checkpoint_path : str | None
        TabDPT checkpoint path (ignored for tabpfn / tabicl).
    context_size : int
        TabDPT retrieval context window (ignored for tabpfn / tabicl).
    use_retrieval : bool
        Whether to use k-NN retrieval in TabDPT (ignored otherwise).
    model_path : str | None
        TabICL local checkpoint path; ``None`` = auto-download (ignored
        for tabpfn / tabdpt).
    """

    def __init__(
        self,
        backbone: str = "tabpfn",
        n_bins: int = 10,
        method: str = "single_context",
        max_context_size: int = 3000,
        device: str = "cpu",
        # TabDPT options
        checkpoint_path: Optional[str] = None,
        context_size: int = 128,
        use_retrieval: bool = True,
        # TabICL options
        model_path: Optional[str] = None,
    ) -> None:
        if backbone not in ("tabpfn", "tabdpt", "tabicl"):
            raise ValueError(f"backbone must be 'tabpfn', 'tabdpt', or 'tabicl', got {backbone!r}")
        if method not in ("single_context", "per_bin"):
            raise ValueError(f"method must be 'single_context' or 'per_bin', got {method!r}")

        self.backbone = backbone
        self.n_bins = n_bins
        self.method = method
        self.max_context_size = max_context_size
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.context_size = context_size
        self.use_retrieval = use_retrieval
        self.model_path = model_path

        # Set after fit()
        self._bin_times: Optional[np.ndarray] = None   # (K,) quantile boundaries
        self._X_train: Optional[np.ndarray] = None
        self._durations: Optional[np.ndarray] = None
        self._events: Optional[np.ndarray] = None
        self._feature_cols: Optional[list] = None

    # ------------------------------------------------------------------
    # FM classifier factory
    # ------------------------------------------------------------------

    def _make_classifier(self):
        """Return a fresh scikit-learn-compatible FM classifier."""
        if self.backbone == "tabpfn":
            import torch
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(
                n_estimators=1,
                device="cuda" if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu",
            )
        if self.backbone == "tabdpt":
            ckpt = self.checkpoint_path or os.environ.get("TABDPT_CHECKPOINT", "")
            if not ckpt:
                # Try known default paths
                _here = os.path.dirname(__file__)
                candidates = [
                    os.path.join(_here, "models_diff", "tabdpt1_1.pth"),
                    "/home/mpham/workspace/source/ehrfm/survpfn/survpfn/models/models_diff/tabdpt1_1.pth",
                ]
                for c in candidates:
                    if os.path.exists(c):
                        ckpt = c
                        break
            if not ckpt:
                raise ValueError(
                    "TabDPT checkpoint not found.  Pass checkpoint_path= or set "
                    "TABDPT_CHECKPOINT env var."
                )
            from survpfn.models.tabdpt.tabdpt.tabdpt import TabDPTClassifier
            return TabDPTClassifier(path=ckpt, device=self.device)
        if self.backbone == "tabicl":
            from survpfn.models.tabicl.tabicl.sklearn.classifier import TabICLClassifier
            return TabICLClassifier(
                n_estimators=1,
                norm_methods="none",
                feat_shuffle_method="none",
                use_amp=True,
                allow_auto_download=True,
                device=self.device,
                random_state=42,
                verbose=False,
                model_path=self.model_path,
            )
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "ZeroShotSurvivalPredictor":
        """Store training context and compute bin boundaries.

        Parameters
        ----------
        df_train : DataFrame with feature columns + duration + event
        duration_col : name of the duration column
        event_col : name of the event column (0=censored, >0=event)
        """
        self._feature_cols = [c for c in df_train.columns
                               if c not in {duration_col, event_col}]
        X = df_train[self._feature_cols].values.astype(np.float32)
        T = df_train[duration_col].values.astype(np.float64)
        E = (df_train[event_col].values > 0).astype(np.int32)

        self._X_train = X
        self._durations = T
        self._events = E

        # Bin boundaries: K quantiles of observed event times
        event_times = T[E == 1]
        if len(event_times) < self.n_bins:
            warnings.warn(
                f"Only {len(event_times)} uncensored events; using all as bin boundaries.",
                stacklevel=2,
            )
        quantile_probs = np.linspace(0, 100, self.n_bins + 2)[1:-1]  # exclude 0% and 100%
        self._bin_times = np.unique(np.percentile(event_times, quantile_probs))

        if self.method == "single_context":
            self._fit_single_context()
        # For per_bin, context is rebuilt at predict time

        return self

    def _fit_single_context(self) -> None:
        """Fit FM once on the time-appended expanded training context."""
        X_exp, y_exp = self._build_expanded_context(
            self._X_train, self._durations, self._events, self._bin_times
        )

        # Subsample if too large
        if len(X_exp) > self.max_context_size:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_exp), self.max_context_size, replace=False)
            X_exp, y_exp = X_exp[idx], y_exp[idx]

        # Guard: need both classes present
        if len(np.unique(y_exp)) < 2:
            warnings.warn("Expanded context has only one class; predictions will be degenerate.")

        self._clf = self._make_classifier()
        self._clf.fit(X_exp, y_exp)

    # ------------------------------------------------------------------
    # predict_survival
    # ------------------------------------------------------------------

    def predict_survival(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Compute survival probabilities S(t_k | X) for each test row.

        Returns
        -------
        pd.DataFrame, shape (n_test, K)
            Columns are bin time values; rows are test samples.
            Each entry S(t_k | X_i) ∈ [0, 1] is non-increasing across columns.
        """
        if self._bin_times is None:
            raise RuntimeError("Call fit() before predict_survival().")

        X_test = df_test[self._feature_cols].values.astype(np.float32)

        if self.method == "single_context":
            surv_matrix = self._predict_single_context(X_test)
        else:
            surv_matrix = self._predict_per_bin(X_test)

        # Enforce monotone non-increasing S(t)
        for i in range(len(X_test)):
            surv_matrix[i] = _enforce_monotone(surv_matrix[i])

        return pd.DataFrame(
            surv_matrix,
            columns=self._bin_times,
        )

    def _predict_single_context(self, X_test: np.ndarray) -> np.ndarray:
        """Predict S(t_k) for all test samples using the pre-fit single-context FM."""
        K = len(self._bin_times)
        n_test = len(X_test)
        surv_matrix = np.zeros((n_test, K), dtype=np.float32)

        for k, t_k in enumerate(self._bin_times):
            # Append normalised time as last feature
            t_col = np.full((n_test, 1), t_k, dtype=np.float32)
            X_q = np.concatenate([X_test, t_col], axis=1)

            try:
                proba = self._clf.predict_proba(X_q)
                # proba[:, 1] = P(event by t_k)
                p_event = proba[:, 1]
            except Exception as exc:
                warnings.warn(f"predict_proba failed at bin {k} (t={t_k:.3f}): {exc}")
                p_event = np.full(n_test, 0.5)

            surv_matrix[:, k] = 1.0 - p_event

        return surv_matrix

    def _predict_per_bin(self, X_test: np.ndarray) -> np.ndarray:
        """Fit one FM per bin and predict S(t_k) independently."""
        K = len(self._bin_times)
        n_test = len(X_test)
        surv_matrix = np.zeros((n_test, K), dtype=np.float32)

        for k, t_k in enumerate(self._bin_times):
            # Context: patients where we have definitive information at t_k
            # i.e., T_i < t_k (event occurred before t_k) OR C_i > t_k (survived past t_k)
            valid_mask = (self._events == 1) | (self._durations > t_k)
            X_ctx = self._X_train[valid_mask]
            T_ctx = self._durations[valid_mask]
            E_ctx = self._events[valid_mask]
            y_ctx = ((E_ctx == 1) & (T_ctx <= t_k)).astype(np.int32)

            if len(np.unique(y_ctx)) < 2 or len(X_ctx) < 2:
                # Degenerate bin — estimate from event rate
                p_event = y_ctx.mean() if len(y_ctx) > 0 else 0.5
                surv_matrix[:, k] = 1.0 - p_event
                continue

            # Subsample
            if len(X_ctx) > self.max_context_size:
                rng = np.random.default_rng(k)
                idx = rng.choice(len(X_ctx), self.max_context_size, replace=False)
                X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]
                if len(np.unique(y_ctx)) < 2:
                    p_event = y_ctx.mean()
                    surv_matrix[:, k] = 1.0 - p_event
                    continue

            clf = self._make_classifier()
            clf.fit(X_ctx, y_ctx)

            try:
                proba = clf.predict_proba(X_test)
                surv_matrix[:, k] = 1.0 - proba[:, 1]
            except Exception as exc:
                warnings.warn(f"predict_proba failed at bin {k} (t={t_k:.3f}): {exc}")
                surv_matrix[:, k] = 1.0 - y_ctx.mean()

        return surv_matrix

    # ------------------------------------------------------------------
    # predict_risk
    # ------------------------------------------------------------------

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Compute a scalar risk score for each test sample.

        Risk = mean(1 − S(t_k)) = 1 − mean(S(t_k)).
        Higher risk → earlier expected event.

        Returns
        -------
        risk : (n_test,) float array — higher values indicate higher risk
        """
        surv_df = self.predict_survival(df_test)
        return 1.0 - surv_df.values.mean(axis=1)

    # ------------------------------------------------------------------
    # concordance_index
    # ------------------------------------------------------------------

    def concordance_index(
        self,
        df_test: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> float:
        """Compute Harrell's C-index on the test set.

        Parameters
        ----------
        df_test : DataFrame with the same feature columns as df_train,
                  plus duration and event columns.

        Returns
        -------
        c_index : float
        """
        from lifelines.utils import concordance_index as ci_fn

        risk = self.predict_risk(df_test)
        durations = df_test[duration_col].values
        events = (df_test[event_col].values > 0).astype(bool)

        try:
            return float(ci_fn(durations, -risk, events))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # compute_baseline — API compatibility
    # ------------------------------------------------------------------

    def compute_baseline(self):
        """No-op.  Retained for API compatibility with pycox-based models."""
        return None

    # ------------------------------------------------------------------
    # Internal helper: expand training data
    # ------------------------------------------------------------------

    @staticmethod
    def _build_expanded_context(
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        bin_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the time-appended expanded context for single_context mode.

        For each patient i and each bin time t_k where t_k < C_i (the
        censoring / event time), we add row (X_i ⊕ t_k, Y_{ik}) where
        Y_{ik} = 1(T_i ≤ t_k AND E_i = 1).

        Parameters
        ----------
        X : (n, d) feature matrix
        T : (n,) times (event or censoring)
        E : (n,) event indicators (1=event, 0=censored)
        bin_times : (K,) sorted time boundaries

        Returns
        -------
        X_exp : (M, d+1)  — time appended as last column
        y_exp : (M,)      — binary labels
        """
        rows_X, rows_y = [], []

        for k, t_k in enumerate(bin_times):
            # Include patient i only if we have definitive information at t_k:
            #   - event patients: always included (T_i < t_k → event; T_i ≥ t_k → no-event-yet)
            #   - censored patients: only if C_i > t_k (we KNOW they survived past t_k)
            valid = (E == 1) | (T > t_k)
            X_k = X[valid]
            T_k = T[valid]
            E_k = E[valid]
            y_k = ((E_k == 1) & (T_k <= t_k)).astype(np.int32)

            t_col = np.full((len(X_k), 1), t_k, dtype=np.float32)
            rows_X.append(np.concatenate([X_k, t_col], axis=1))
            rows_y.append(y_k)

        if not rows_X:
            return np.empty((0, X.shape[1] + 1), dtype=np.float32), np.empty(0, dtype=np.int32)

        return np.concatenate(rows_X, axis=0), np.concatenate(rows_y, axis=0)


# ---------------------------------------------------------------------------
# Benchmark-compatible training function
# ---------------------------------------------------------------------------

def train_zeroshot_surv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    backbone: str = "tabpfn",
    n_bins: int = 10,
    method: str = "single_context",
    max_context_size: int = 3000,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    context_size: int = 128,
    use_retrieval: bool = True,
    model_path: Optional[str] = None,
    **_,
) -> tuple:
    """Fit a ZeroShotSurvivalPredictor and return benchmark-compatible outputs.

    Returns
    -------
    (predictor, risk_scores, surv_probs, surv_times)
        risk_scores : (n_test,)
        surv_probs  : (n_test, K)  — survival probabilities at bin times
        surv_times  : (K,)         — bin time boundaries
    """
    predictor = ZeroShotSurvivalPredictor(
        backbone=backbone,
        n_bins=n_bins,
        method=method,
        max_context_size=max_context_size,
        device=device,
        checkpoint_path=checkpoint_path,
        context_size=context_size,
        use_retrieval=use_retrieval,
        model_path=model_path,
    )
    predictor.fit(df_train, duration_col, event_col)

    surv_df = predictor.predict_survival(df_test)
    risk = predictor.predict_risk(df_test)

    surv_probs = surv_df.values            # (n_test, K)
    surv_times = surv_df.columns.values    # (K,)

    return predictor, risk, surv_probs, surv_times
