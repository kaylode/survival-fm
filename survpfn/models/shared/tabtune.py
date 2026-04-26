"""
TabTune — survival finetuning via `tabtune.TabularPipeline`.

Supported backbones
-------------------
* TabPFNTune  — uses model_name="TabPFN"  with tuning_strategy="peft"
* TabDPTTune  — uses model_name="TabDPT"  with tuning_strategy="peft"
* TabICLTune  — uses model_name="TabICL"  with tuning_strategy="peft"

All three share the same survival-specific logic from the base class `TabTune`:
1. Temporal expansion  (survival → binary classification per time-bin)
2. Monotone survival-curve reconstruction
3. Standard evaluation

Usage example
-------------
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 10,
        "learning_rate": 5e-5,
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05
        }
    }
)
pipeline.fit(X_train, y_train)

# Save entire pipeline
pipeline.save("my_pipeline.joblib")

# Load and use
loaded_pipeline = TabularPipeline.load("my_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from survpfn.models.shared.preprocessing import (
    FMDataPrep,
    SurvivalTimeBinEncoder,
    expand_survival_data,
)


# ---------------------------------------------------------------------------
# TabTune base class
# ---------------------------------------------------------------------------

class TabTune:
    """Base class for survival finetuning via temporal expansion using tabtune.

    Shared logic for:
    1. Dataset expansion (survival -> binary classification)
    2. Monotone survival path reconstruction
    3. Standard evaluation

    Concrete subclasses must implement:
    - ``_init_pipeline()``  — create and store ``self.pipeline`` (a tabtune TabularPipeline)
    - ``_fit_pipeline(X_exp, y_exp)`` — call ``self.pipeline.fit(X_exp, y_exp)``
    - ``_predict_proba_internal(x_bin)`` — return (N, 2) probability array
    """

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.pop("learning_rate", 1e-4)
        self.epochs = kwargs.pop("epochs", 30)
        self.batch_size = kwargs.pop("batch_size", 512)
        self.device = kwargs.pop("device", "cuda:0")
        self.random_state = kwargs.pop("random_state", 42)
        self.sampling_ratio = kwargs.pop("sampling_ratio", None)
        self.binning_scheme = kwargs.pop("binning_scheme", "quantile")
        self.patience = kwargs.pop("patience", None)
        self.early_event_quantile = kwargs.pop("early_event_quantile", 0.5)
        self.early_bin_frac = kwargs.pop("early_bin_frac", 0.7)
        self.task_type = kwargs.pop("task_type", "sr")

        self.use_isotonic_calibration = kwargs.pop("use_isotonic_calibration", False)
        self.calibration_quantiles = kwargs.pop("calibration_quantiles", (0.25, 0.50, 0.75))
        self.calibration_horizons = kwargs.pop("calibration_horizons", None)
        self._calibrators = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        """Create and store ``self.pipeline``.  Called once inside ``fit()``."""
        raise NotImplementedError("Subclass must implement _init_pipeline()")

    def _fit_pipeline(self, X_exp: np.ndarray, y_exp: np.ndarray) -> None:
        """Fit ``self.pipeline`` on the temporally-expanded dataset."""
        raise NotImplementedError("Subclass must implement _fit_pipeline()")

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array for a batch of (patient+bin) rows."""
        raise NotImplementedError("Subclass must implement _predict_proba_internal()")

    # ------------------------------------------------------------------
    # Shared predict_survival_df (identical to BaseSurvExpandedFinetune)
    # ------------------------------------------------------------------

    def predict_survival_df(self, x: np.ndarray, n_ensemble: int = 0) -> pd.DataFrame:
        """Return survival probability DataFrame (rows=times, cols=subjects)."""
        x_scaled = self._prep.transform(x)
        n_test = len(x_scaled)
        K = len(self.bin_times)

        def _get_surv_matrix():
            f_0 = np.repeat(self.bin_feats[0:1], n_test, axis=0)
            x_bin_0 = np.concatenate([x_scaled, f_0], axis=1)
            probs_0 = self._predict_proba_internal(x_bin_0)
            num_classes = probs_0.shape[1]

            surv_m_list = [np.zeros((n_test, K), dtype=np.float32) for _ in range(num_classes)]
            for k in range(K):
                f_k = np.repeat(self.bin_feats[k:k + 1], n_test, axis=0)
                x_bin = np.concatenate([x_scaled, f_k], axis=1)
                probs = self._predict_proba_internal(x_bin)
                for c in range(num_classes):
                    surv_m_list[c][:, k] = probs[:, c]
            return surv_m_list

        surv_matrix_list = _get_surv_matrix()
        num_classes = len(surv_matrix_list)

        # ── Interpolation (Unique times only) ──
        times = np.concatenate([[0], self.bin_times])
        times, unique_idx = np.unique(times, return_index=True)
        grid_times = np.linspace(0, self.bin_times.max(), 1000)
        from scipy.interpolate import interp1d

        result_dfs = []
        for c in range(num_classes):
            surv_m = surv_matrix_list[c]
            if c == 0:
                surv = np.concatenate([np.ones((n_test, 1)), surv_m], axis=1)
            else:
                surv = np.concatenate([np.zeros((n_test, 1)), surv_m], axis=1)

            surv = surv[:, unique_idx]
            f = interp1d(times, surv, kind="linear", axis=1, fill_value="extrapolate")
            surv_matrix_interp = np.clip(f(grid_times), 0.0, 1.0)

            surv_df = pd.DataFrame(
                surv_matrix_interp.T,
                index=grid_times,
                columns=np.arange(n_test),
            )
            result_dfs.append(surv_df)

        if self.task_type == "sr":
            return result_dfs[0]
        else:
            return result_dfs[1:]

    # ------------------------------------------------------------------
    # Shared fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        val_split: float = 0.2,
        verbose: bool = True,
        **kwargs,
    ) -> "TabTune":
        """Generic training loop for temporal expansion survival finetuning."""

        # ── 0. Validation Split (kept for API compatibility) ──────────────
        if val_data is None and val_split > 0:
            x, x_val, durations, durations_val, events, events_val = train_test_split(
                x, durations, events, test_size=val_split,
                random_state=self.random_state, stratify=events,
            )
            val_data = (x_val, durations_val, events_val)

        # ── 1. Time bins ──────────────────────────────────────────────────
        self.encoder = SurvivalTimeBinEncoder(
            n_bins=self.num_durations,
            scheme=self.binning_scheme,
            early_event_quantile=self.early_event_quantile,
            early_bin_frac=self.early_bin_frac,
        )
        self.encoder.fit(durations, events)
        self.bin_times = self.encoder.bin_times
        self.bin_feats = self.encoder.bin_feats  # (K, N_TIME_FEATURES)

        # ── 2. Preprocessing ──────────────────────────────────────────────
        self._prep = FMDataPrep()
        x_scaled = self._prep.fit_transform(x, max_features=None)

        # ── 3. Temporal expansion ─────────────────────────────────────────
        #  No sampling — pass sampling_ratio=None so every (patient, bin) pair
        #  with definitive information is included.
        X_exp, y_exp = expand_survival_data(
            x_scaled, durations, events,
            self.bin_times, self.bin_feats,
            sampling_ratio=None,          # no sampling, per user request
            random_state=self.random_state,
        )

        if verbose:
            print(f"Expanded dataset: {X_exp.shape[0]:,} rows, {X_exp.shape[1]} features", flush=True)
            unique, counts = np.unique(y_exp, return_counts=True)
            print(f"y_exp distribution: {dict(zip(unique, counts))}", flush=True)

        # ── 4. Build pipeline (subclass) + fit ────────────────────────────
        self._init_pipeline()
        self._fit_pipeline(X_exp, y_exp)

        return self

    # ------------------------------------------------------------------
    # Evaluation helper (static, mirrors BaseSurvExpandedFinetune)
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        surv_df: pd.DataFrame,
        durations_test: np.ndarray,
        events_test: np.ndarray,
        durations_train: np.ndarray,
        events_train: np.ndarray,
    ) -> dict:
        """Compute survival evaluation metrics via pycox EvalSurv."""
        from pycox.evaluation import EvalSurv

        if isinstance(surv_df, list):
            events_test = (events_test == 1).astype(int)
            surv_df = 1.0 - surv_df[0]

        ev = EvalSurv(surv_df, durations_test, events_test, censor_surv="km")
        c_index = ev.concordance_td("antolini")
        return {"c_index": c_index}


# ---------------------------------------------------------------------------
# Helper: build a tabtune TabularPipeline
# ---------------------------------------------------------------------------

# Default LoRA config used when tuning_strategy="peft"
_DEFAULT_PEFT_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}


def _make_tabtune_pipeline(
    model_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    random_state: int,
    finetune_mode: str = "meta-learning",
    tuning_strategy: str = "peft",
    peft_config: Optional[dict] = None,
    extra_tuning_params: Optional[dict] = None,
    extra_model_params: Optional[dict] = None,
) -> "TabularPipeline":  # type: ignore[name-defined]
    """Construct and return a tabtune ``TabularPipeline`` using PEFT (LoRA)."""
    from tabtune.TabularPipeline.pipeline import TabularPipeline  # type: ignore[import]

    resolved_peft = dict(_DEFAULT_PEFT_CONFIG)
    if peft_config:
        resolved_peft.update(peft_config)

    tuning_params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": device,
        "show_progress": False,
        "peft_config": resolved_peft,
    }
    if extra_tuning_params:
        tuning_params.update(extra_tuning_params)

    model_params: dict = {"device": device} #, "random_state": random_state}
    if extra_model_params:
        model_params.update(extra_model_params)

    pipeline = TabularPipeline(
        model_name=model_name,
        task_type="classification",
        tuning_strategy=tuning_strategy,
        finetune_mode=finetune_mode,
        tuning_params=tuning_params,
        model_params=model_params,
    )
    return pipeline


def _pipeline_predict_proba(pipeline, x_bin: np.ndarray) -> np.ndarray:
    """Call ``pipeline.predict_proba`` and normalise to (N, 2) float32."""
    X_df = pd.DataFrame(x_bin)
    proba = pipeline.predict_proba(X_df)
    # tabtune returns (N, n_classes); ensure 2-class output
    if proba.ndim == 1:
        proba = np.stack([1 - proba, proba], axis=1)
    return proba.astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete subclass — TabPFNTune
# ---------------------------------------------------------------------------

class TabPFNTune(TabTune):
    """Survival finetuning with TabPFN via tabtune PEFT (LoRA).

    Parameters
    ----------
    num_durations : int
        Number of time bins K.
    finetune_mode : str
        ``"meta-learning"`` (default) or ``"sft"`` or ``"native"``.
    peft_config : dict, optional
        LoRA config overrides.  Merged with ``_DEFAULT_PEFT_CONFIG``.
        Keys: ``r``, ``lora_alpha``, ``lora_dropout``.
    **kwargs
        Forwarded to :class:`TabTune` base constructor.
    """

    def __init__(
        self,
        num_durations: int = 100,
        finetune_mode: str = "meta-learning",
        tuning_strategy: str = "peft",
        peft_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        self.finetune_mode = finetune_mode
        self.peft_config = peft_config
        self.backbone_name = "tabpfn"
        self.tuning_strategy = tuning_strategy

    def _init_pipeline(self) -> None:
        self.pipeline = _make_tabtune_pipeline(
            model_name="TabPFN",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            finetune_mode=self.finetune_mode,
            peft_config=self.peft_config,
            tuning_strategy=self.tuning_strategy,
        )
        print(
            f"TabPFNTune initialised  (device={self.device}, bins={self.num_durations}, "
            f"strategy=peft, finetune_mode={self.finetune_mode})",
            flush=True,
        )

    def _fit_pipeline(self, X_exp: np.ndarray, y_exp: np.ndarray) -> None:
        X_df = pd.DataFrame(X_exp)
        y_s = pd.Series(y_exp.astype(int))
        self.pipeline.fit(X_df, y_s)

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(x_bin)
        proba = self.pipeline.predict_proba(X_df)
        if proba.ndim == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba.astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete subclass — TabDPTTune
# ---------------------------------------------------------------------------

class TabDPTTune(TabTune):
    """Survival finetuning with TabDPT via tabtune PEFT (LoRA).

    Parameters
    ----------
    num_durations : int
        Number of time bins K.
    finetune_mode : str
        ``"meta-learning"`` (default) or ``"sft"``.
    peft_config : dict, optional
        LoRA config overrides.  Merged with ``_DEFAULT_PEFT_CONFIG``.
    **kwargs
        Forwarded to :class:`TabTune` base constructor.
    """

    def __init__(
        self,
        num_durations: int = 50,
        finetune_mode: str = "meta-learning",
        tuning_strategy: str = "peft",
        peft_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        self.finetune_mode = finetune_mode
        self.peft_config = peft_config
        self.backbone_name = "tabdpt"
        self.tuning_strategy = tuning_strategy

    def _init_pipeline(self) -> None:
        extra_model_params = {
            "compile": False,
            "use_flash": False,
            "normalizer": "standard",
            "missing_indicators": False,
        }
        self.pipeline = _make_tabtune_pipeline(
            model_name="TabDPT",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            finetune_mode=self.finetune_mode,
            peft_config=self.peft_config,
            extra_model_params=extra_model_params,
            tuning_strategy=self.tuning_strategy,
        )
        print(
            f"TabDPTTune initialised  (device={self.device}, bins={self.num_durations}, "
            f"strategy=peft, finetune_mode={self.finetune_mode})",
            flush=True,
        )

    def _fit_pipeline(self, X_exp: np.ndarray, y_exp: np.ndarray) -> None:
        X_df = pd.DataFrame(X_exp)
        y_s = pd.Series(y_exp.astype(int))
        self.pipeline.fit(X_df, y_s)

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(x_bin)
        proba = self.pipeline.predict_proba(X_df)
        if proba.ndim == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba.astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete subclass — TabICLTune
# ---------------------------------------------------------------------------

class TabICLTune(TabTune):
    """Survival finetuning with TabICL via tabtune PEFT (LoRA).

    Parameters
    ----------
    num_durations : int
        Number of time bins K.
    finetune_mode : str
        ``"meta-learning"`` (default, recommended for ICL models) or ``"sft"``.
    peft_config : dict, optional
        LoRA config overrides.  Merged with ``_DEFAULT_PEFT_CONFIG``.
    **kwargs
        Forwarded to :class:`TabTune` base constructor.
    """

    def __init__(
        self,
        num_durations: int = 50,
        finetune_mode: str = "meta-learning",
        tuning_strategy: str = "peft",
        peft_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        self.finetune_mode = finetune_mode
        self.peft_config = peft_config
        self.backbone_name = "tabicl"
        self.tuning_strategy = tuning_strategy

    def _init_pipeline(self) -> None:
        self.pipeline = _make_tabtune_pipeline(
            model_name="TabICL",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            finetune_mode=self.finetune_mode,
            peft_config=self.peft_config,
            tuning_strategy=self.tuning_strategy,
        )
        print(
            f"TabICLTune initialised  (device={self.device}, bins={self.num_durations}, "
            f"strategy=peft, finetune_mode={self.finetune_mode})",
            flush=True,
        )

    def _fit_pipeline(self, X_exp: np.ndarray, y_exp: np.ndarray) -> None:
        X_df = pd.DataFrame(X_exp)
        y_s = pd.Series(y_exp.astype(int))
        self.pipeline.fit(X_df, y_s)

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(x_bin)
        proba = self.pipeline.predict_proba(X_df)
        if proba.ndim == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba.astype(np.float32)