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
        input_dim: Optional[int] = None,
        num_durations: int = 100,
        learning_rate: float = 2e-5,
        epochs: int = 30,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 2,
        device: str = "cuda:0",
        random_state: int = 0,
        output_dir:str = None,
        **kwargs
    ):
        self.num_durations = num_durations
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.random_state = random_state
        self.n_estimators_finetune = n_estimators_finetune
        self.n_estimators_validation = n_estimators_validation
        self.n_estimators_final_inference = n_estimators_final_inference
        self.output_dir=output_dir
        # We'll use a standard classifier wrapper for finetuning
        self.clf = FinetunedTabPFNClassifier(
            device=device,
            epochs=15,
            learning_rate=learning_rate,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            random_state=random_state,
        )
        
        self.num_expected_features = 2038 # Common TabPFN capacity
        print(f"TabPFNSurvPH Initialized with FinetunedTabPFNClassifier (device={device})", flush=True)
        
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
        verbose: bool = True,
        sampling_ratio: Optional[float] = None,
        **kwargs
    ):
        # Time Bins & Encoder
        self.encoder = SurvivalTimeBinEncoder(n_bins=self.num_durations)
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
            
        sampling_ratio = sampling_ratio if sampling_ratio is not None else getattr(self, "sampling_ratio", None)
        random_state = kwargs.get("random_state", self.random_state if hasattr(self, "random_state") else 42)

        x_exp, y_exp = expand_survival_data(
            x_scaled, durations, events, self.bin_times, self.bin_feats,
            sampling_ratio=sampling_ratio,
            random_state=random_state
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

