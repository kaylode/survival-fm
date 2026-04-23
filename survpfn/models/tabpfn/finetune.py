from __future__ import annotations

import warnings
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
import pandas as pd

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from survpfn.models.shared.preprocessing import expand_survival_data, FMDataPrep, prepare_targets, SurvivalTimeBinEncoder
from survpfn.models.shared.finetune import BaseSurvExpandedFinetune


# ---------------------------------------------------------------------------
# TabPFN Survival Model (Backbone + Survival Head)
# ---------------------------------------------------------------------------

# TabPFNSurvModelFinetune was removed as FinetunedTabPFNClassifier handles the model.


class TabPFNSurvPHFinetune(BaseSurvExpandedFinetune):
    def __init__(
        self,
        num_durations: int = 100,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 2,
        output_dir: str = None,
        **kwargs
    ):
        super().__init__(num_durations=num_durations, **kwargs)
        self.n_estimators_finetune = n_estimators_finetune
        self.n_estimators_validation = n_estimators_validation
        self.n_estimators_final_inference = n_estimators_final_inference
        self.output_dir = output_dir
        
        # We'll use a standard classifier wrapper for finetuning
        self.clf = FinetunedTabPFNClassifier(
            device=self.device,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            n_estimators_finetune=self.n_estimators_finetune,
            n_estimators_validation=self.n_estimators_validation,
            n_estimators_final_inference=self.n_estimators_final_inference,
            random_state=self.random_state,
        )
        
        self.num_expected_features = 2038 # Common TabPFN capacity
        print(f"TabPFNSurvPH Initialized with FinetunedTabPFNClassifier (device={self.device})", flush=True)
        
        # Print parameter summary
        # breakpoint()
        # total_params = sum(p.numel() for p in self.clf.finetuned_estimator_.model_.parameters())
        # trainable_params = sum(p.numel() for p in self.clf.finetuned_estimator_.model_.parameters() if p.requires_grad)
        # print(f"TabDPTSurvPHFinetune Initialized: {trainable_params:,} trainable / {total_params:,} total parameters "
        #       f"({trainable_params/total_params:.2%})", flush=True)

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        val_split: float = 0.2,
        verbose: bool = True,
        **kwargs
    ):
        # ── 0. Validation Split ──
        if val_data is None and val_split > 0:
            from sklearn.model_selection import train_test_split
            x, x_val, durations, durations_val, events, events_val = train_test_split(
                x, durations, events, test_size=val_split, random_state=self.random_state,
                stratify=events
            )
            val_data = (x_val, durations_val, events_val)
            
        # Time Bins & Encoder
        self.encoder = SurvivalTimeBinEncoder(n_bins=self.num_durations, scheme=self.binning_scheme)
        self.encoder.fit(durations, events)
        
        self.bin_times = self.encoder.bin_times
        self.bin_feats = self.encoder.bin_feats # (K, 6)

        # Feature Preprocessing
        pca_capacity = self.num_expected_features
        self._prep = FMDataPrep()
        x_scaled = self._prep.fit_transform(x, max_features=pca_capacity - SurvivalTimeBinEncoder.N_TIME_FEATURES)
        
        # Dataset Expansion (Survival -> Classification)
        if verbose:
            print("Expanding survival dataset for FinetunedTabPFNClassifier...", flush=True)

        x_exp, y_exp = expand_survival_data(
            x_scaled, durations, events, self.bin_times, self.bin_feats,
            sampling_ratio=self.sampling_ratio,
            random_state=self.random_state
        )
        
        if verbose:
            print(f"Expanded dataset: {x_exp.shape[0]:,} rows, {x_exp.shape[1]} features", flush=True)
            # Label distribution
            unique, counts = np.unique(y_exp, return_counts=True)
            print(f"y_exp distribution: {dict(zip(unique, counts))}", flush=True)
            
        # Fit Wrapper
        self.clf.fit(x_exp, y_exp, output_dir=self.output_dir)
        
        return self

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        # Get probabilities (Class 0 = Survived, Class 1 = Event)
        return self.clf.predict_proba(x_bin)

