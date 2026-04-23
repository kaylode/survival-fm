"""survpfn.models.shared.cr.zeroshot — Zero-shot ICL for competing risks.

Class
-----
CRZeroShotPredictor
    Extends ``ZeroShotSurvivalPredictor`` for competing-risks CIF estimation.

    Two CR modes
    ~~~~~~~~~~~~
    ``multinomial`` (default)
        A single FM instance is trained with *K+1* class labels
        (0 = censored, 1..K = causes).  At each time bin the FM outputs
        ``P(cause k by t_k | x)`` directly as class probabilities.

    ``per_event``
        One binary FM per cause k, trained with cause-k events as positives
        and all other events treated as censored.  CIF_k(t) = P(event k by t).

    Output convention
    ~~~~~~~~~~~~~~~~~
    All ``predict_survival`` calls return ``List[pd.DataFrame]`` where each
    DataFrame has shape ``(n_times, n_patients)`` — rows = times, cols =
    patients — matching pycox's ``EvalSurv`` convention.

Public API
----------
``train_cr_zeroshot`` — convenience wrapper around ``CRZeroShotPredictor``.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from survpfn.models.shared.zeroshot import ZeroShotSurvivalPredictor, _enforce_monotone
from survpfn.models.shared.binning import get_cuts, resolve_num_durations


class CRZeroShotPredictor(ZeroShotSurvivalPredictor):
    """Zero-shot ICL predictor for competing risks.

    Inherits all backbone/context/PCA infrastructure from
    ``ZeroShotSurvivalPredictor`` and overrides ``fit`` / ``predict_survival``
    to handle multiple causes correctly.

    Parameters
    ----------
    cr_method : str
        ``"multinomial"`` or ``"per_event"`` (default ``"per_event"``).
    All other parameters are passed through to ``ZeroShotSurvivalPredictor``.
    """

    def __init__(
        self,
        cr_method: str = "per_event",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cr_method = cr_method
        # Per-event context stores: {cause_idx: contexts_list}
        self._clfs_cr: Dict[int, Any] = {}
        # Per-event per-bin: {bin_idx: {cause_idx: contexts_list}}
        self._clfs_cr_per_bin: Dict[int, Dict[int, Any]] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "CRZeroShotPredictor":
        """Fit CR zero-shot predictor from training data."""
        from survpfn.models.shared.preprocessing import SurvivalTimeBinEncoder

        self._feature_cols = [
            c for c in df_train.columns if c not in {duration_col, event_col}
        ]
        self._X_train  = df_train[self._feature_cols].values.astype(np.float32)
        self._durations = df_train[duration_col].values
        self._events   = df_train[event_col].values.astype(int)
        self._num_events = int(self._events.max())

        n_bins = resolve_num_durations(self._durations, self._events, self.n_bins)

        # Respect use_time_bin_encoder — the parent class does this; the CR
        # override must mirror it so context features and query features always
        # have the same dimensionality (1-dim vs 6-dim time feature).
        if self.use_time_bin_encoder:
            self._time_encoder = SurvivalTimeBinEncoder(
                n_bins=n_bins,
                scheme=self.binning_scheme,
                early_event_quantile=self.early_event_quantile,
                early_bin_frac=self.early_bin_frac,
            )
            self._time_encoder.fit(self._durations, self._events)
            self._bin_times = self._time_encoder.bin_times
        else:
            self._time_encoder = None
            self._bin_times = get_cuts(
                self._durations, self._events, n_bins, scheme=self.binning_scheme,
                early_event_quantile=self.early_event_quantile,
                early_bin_frac=self.early_bin_frac,
            )

        if self.verbose:
            print(
                f"  [{self.backbone}/ZeroShot-CR] "
                f"method={self.method}, cr_method={self.cr_method}, "
                f"bins={len(self._bin_times)}, estimators={self.n_estimators}",
                flush=True,
            )

        # Global PCA if features are too many
        n_temp = SurvivalTimeBinEncoder.N_TIME_FEATURES if self.use_time_bin_encoder else 1
        if len(self._feature_cols) + n_temp > 2000:
            X_pca, _ = self._build_expanded_context(
                self._X_train, self._durations, self._events, self._bin_times,
                encoder=self._time_encoder,
            )
            n_comp = min(X_pca.shape[0], X_pca.shape[1], 2000)
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(X_pca)

        # ── Fit contexts based on method × cr_method ──
        if self.method == "single_context":
            if self.cr_method == "multinomial":
                # Single FM sees all cause codes as class labels
                self._clf = self._fit_context(
                    self._X_train, self._durations, self._events
                )
            else:  # per_event
                for cause in range(1, self._num_events + 1):
                    e_binary = (self._events == cause).astype(int)
                    self._clfs_cr[cause] = self._fit_context(
                        self._X_train, self._durations, e_binary
                    )

        else:  # per_bin
            pbar = tqdm(
                enumerate(self._bin_times),
                total=len(self._bin_times),
                desc=f"  [{self.backbone}/ZeroShot-CR] Fitting per-bin",
                file=sys.stdout,
                leave=False,
                disable=not self.verbose,
            )
            for k, t_k in pbar:
                if self.cr_method == "multinomial":
                    self._clfs_per_bin[k] = self._fit_context(
                        self._X_train, self._durations,
                        self._events, bin_times=[t_k],
                    )
                else:  # per_event
                    self._clfs_cr_per_bin[k] = {}
                    for cause in range(1, self._num_events + 1):
                        e_binary = (self._events == cause).astype(int)
                        self._clfs_cr_per_bin[k][cause] = self._fit_context(
                            self._X_train, self._durations,
                            e_binary, bin_times=[t_k],
                        )

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_survival(self, df_test: pd.DataFrame) -> List[pd.DataFrame]:
        """Return per-cause CIF DataFrames in pycox convention (n_times × n_patients)."""
        X_test = df_test[self._feature_cols].values.astype(np.float32)
        if self.cr_method == "multinomial":
            return self._predict_cr_multinomial(X_test)
        return self._predict_cr_per_event(X_test)

    # ------------------------------------------------------------------
    # internal predictors
    # ------------------------------------------------------------------

    def _predict_cr_multinomial(self, X_test: np.ndarray) -> List[pd.DataFrame]:
        """CIF via a single multinomial FM: P(cause k by t_k | x)."""
        K_bins = len(self._bin_times)
        n_test = len(X_test)
        # event_probs[m][i, k] = P(cause m+1 by time bin k | patient i)
        event_probs = [np.zeros((n_test, K_bins)) for _ in range(self._num_events)]

        pbar = tqdm(
            enumerate(self._bin_times),
            total=K_bins,
            desc=f"  [{self.backbone}/ZeroShot-CR] Predicting (multinomial)",
            file=sys.stdout, leave=False, disable=not self.verbose,
        )
        for k, t_k in pbar:
            if self._time_encoder is not None:
                feat    = self._time_encoder.bin_feats[k]
                t_f_col = np.tile(feat, (n_test, 1))
                X_q     = np.concatenate([X_test, t_f_col], axis=1)
            else:
                t_col = np.full((n_test, 1), t_k)
                X_q   = np.concatenate([X_test, t_col], axis=1)
            if self._pca is not None:
                X_q = self._pca.transform(X_q)

            clf = (
                self._clf if self.method == "single_context"
                else self._clfs_per_bin[k]
            )
            for m in range(1, self._num_events + 1):
                event_probs[m - 1][:, k] = self._get_probs(clf, X_q, m)

        return self._build_cif_dfs(event_probs, n_test)

    def _predict_cr_per_event(self, X_test: np.ndarray) -> List[pd.DataFrame]:
        """CIF via per-cause binary FMs: CIF_k(t) = P(event k by t)."""
        K_bins = len(self._bin_times)
        n_test = len(X_test)
        event_probs = [np.zeros((n_test, K_bins)) for _ in range(self._num_events)]

        for m in range(1, self._num_events + 1):
            pbar = tqdm(
                enumerate(self._bin_times),
                total=K_bins,
                desc=f"  [{self.backbone}/ZeroShot-CR] Predicting (cause {m})",
                file=sys.stdout, leave=False, disable=not self.verbose,
            )
            for k, t_k in pbar:
                if self._time_encoder is not None:
                    feat    = self._time_encoder.bin_feats[k]
                    t_f_col = np.tile(feat, (n_test, 1))
                    X_q     = np.concatenate([X_test, t_f_col], axis=1)
                else:
                    t_col = np.full((n_test, 1), t_k)
                    X_q   = np.concatenate([X_test, t_col], axis=1)
                if self._pca is not None:
                    X_q = self._pca.transform(X_q)

                if self.method == "single_context":
                    clf = self._clfs_cr[m]
                else:
                    clf = self._clfs_cr_per_bin[k][m]

                # For binary cause-k FM: class 1 = cause-k event
                event_probs[m - 1][:, k] = self._get_probs(clf, X_q, 1)

        return self._build_cif_dfs(event_probs, n_test)

    # ------------------------------------------------------------------
    # shared helper
    # ------------------------------------------------------------------

    def _build_cif_dfs(
        self,
        event_probs: List[np.ndarray],
        n_test: int,
    ) -> List[pd.DataFrame]:
        """Convert raw bin-level incidence probabilities to pycox DataFrames.

        Parameters
        ----------
        event_probs : List[np.ndarray]
            Length ``num_events``.  Each element is ``(n_test, n_bins)``:
            estimated P(cause m+1 by time bin k | patient i).
        n_test : int
            Number of test patients.

        Returns
        -------
        List[pd.DataFrame]
            One DataFrame per cause, shape ``(n_times, n_patients)``.
        """
        from scipy.interpolate import interp1d

        times = np.concatenate([[0.0], self._bin_times])
        t_unique, idx_unique = np.unique(times, return_index=True)

        t_max      = float(self._bin_times.max())
        grid_times = np.linspace(0.0, t_max, 1000)

        cifs: List[pd.DataFrame] = []
        for m in range(self._num_events):
            cif_m = event_probs[m].copy()  # (n_test, n_bins)

            # Enforce isotonic monotonicity (non-decreasing CIF)
            for i in range(n_test):
                cif_m[i] = _enforce_monotone(cif_m[i], increasing=True)

            # Prepend t=0 with CIF(0)=0
            cif_ext    = np.concatenate([np.zeros((n_test, 1)), cif_m], axis=1)
            cif_unique = cif_ext[:, idx_unique]   # remove duplicate time points

            # Interpolate to dense grid
            f        = interp1d(t_unique, cif_unique, kind="linear",
                                axis=1, fill_value="extrapolate")
            cif_grid = np.clip(f(grid_times), 0.0, 1.0)  # (n_test, 1000)

            # pycox convention: rows = times, cols = patients
            cifs.append(
                pd.DataFrame(
                    cif_grid.T,
                    index=grid_times,
                    columns=np.arange(n_test),
                )
            )
        return cifs


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def train_cr_zeroshot(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    **kwargs,
) -> tuple:
    """Fit a ``CRZeroShotPredictor`` and return benchmark-ready outputs.

    Returns
    -------
    (predictor, risk, cif_arrays, grid_times)
        ``risk``      — ``(n_test,)`` — CIF_1 at last time (cause-1 risk score).
        ``cif_arrays``— ``List[np.ndarray]`` — each ``(n_times, n_test)``.
        ``grid_times``— ``(n_times,)`` — shared time grid.
    """
    predictor = CRZeroShotPredictor(**kwargs)
    predictor.fit(df_train, duration_col, event_col)
    cifs = predictor.predict_survival(df_test)

    # Risk score: CIF_1 at last time point (rows=times, take last row)
    risk = cifs[0].iloc[-1].values  # (n_test,)

    return (
        predictor,
        risk,
        [c.values.T for c in cifs],   # (n_test, n_times) per cause
        cifs[0].index.values,          # grid_times
    )
