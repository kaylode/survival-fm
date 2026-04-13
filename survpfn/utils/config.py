
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, List, Literal
import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# FMConfig — single source of truth for all FM hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class FMConfig:
    """Unified configuration for all Foundation Model survival strategies.

    Pass one of these to any FM wrapper instead of scattered **kwargs.
    Every parameter has a sensible default so callers only need to override
    what they actually want to change.

    Strategies
    ----------
    frozen_transfer  : freeze backbone, train survival head only
    joint_adaptation : frozen backbone + survival head + classification loss
    zero_shot        : ICL direct prediction, no head training
    surv_adapter     : KM-augmented BCE adapter on frozen embeddings

    Examples
    --------
    >>> cfg = FMConfig(backbone="tabicl", head_type="deephit", device="cuda:0")
    >>> cfg = FMConfig.from_kwargs(epochs=50, lr=1e-4, device="cuda:0")
    """

    # ── Head ─────────────────────────────────────────────────────────────────
    head_type: str = "cox"
    """Survival head: cox | deephit | pchazard | mtlr | deephit_cr"""
    num_durations: int = -1
    """Number of discrete time bins for DeepHit / MTLR / PCHazard heads.
    -1 (default) = auto-detect from training data via adaptive_num_durations()."""
    head_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    """Hidden layer widths of the survival head MLP."""
    dropout: float = 0.1

    # ── FM Backbone ───────────────────────────────────────────────────────────
    backbone: str = "tabpfn"
    """Foundation model: tabpfn | tabdpt | tabicl"""
    freeze_backbone: bool = True
    """If True only the head trains; if False gradients flow through backbone."""
    context_size: int = 256
    """Max ICL context samples per forward pass (joint/zero-shot strategies)."""
    use_adapter: bool = False
    """Prepend a 2-layer MLP adapter before the FM to project raw features."""

    # ── Training ─────────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    """Learning rate for the survival head (and backbone if freeze_backbone=False)."""
    alpha: float = 0.5
    """Weight for the FM classification loss added to the survival loss."""
    n_ensemble: int = 0
    """Number of random contexts sampled for ensembling during inference (0 = disabled)."""
    sampling_ratio: Optional[float] = None
    """Ratio of samples to use for training (None = use all)."""

    # ── Task ─────────────────────────────────────────────────────────────────
    task_type: str = "sr"
    """Task type: sr (single-risk) | cr (competing-risks)."""
    cr_loss_type: str = "deephit"
    """Competing-risks loss: deephit | deephit_v2 | cox."""

    # ── Zero-shot specific ───────────────────────────────────────────────────
    zeroshot_method: str = "single_context"
    """Zero-shot mode: single_context (1 FM fit) | per_bin (K FM fits)."""
    zeroshot_n_bins: int = -1
    """Number of quantile time bins for zero-shot ICL. -1 = auto-detect."""
    zeroshot_cr_method: str = "multinomial"
    """CR labelling for zero-shot: multinomial | per_event."""

    # ── Runtime ──────────────────────────────────────────────────────────────
    device: str = "cpu"
    random_state: int = 42
    verbose: bool = True

    # ── HPO ──────────────────────────────────────────────────────────────────
    tune: bool = False
    n_trials: int = 10

    # -------------------------------------------------------------------------

    @classmethod
    def from_kwargs(cls, **kw) -> "FMConfig":
        """Build an FMConfig from a flat kwargs dict, silently ignoring unknown keys.

        Alias mapping: ``lr`` → ``learning_rate``.
        """
        alias = {"lr": "learning_rate"}
        kw = {alias.get(k, k): v for k, v in kw.items()}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kw.items() if k in valid})

    def to_dict(self) -> dict:
        """Return config as a plain dict (for JSON serialisation)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

def load_model_config(model_name: str) -> dict[str, Any]:
    """Load model-specific HPO and default configurations from JSON.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'deephit_single', 'rsf', 'embedding_head').
        
    Returns
    -------
    A dictionary containing 'tuning' and 'default' parameters.
    """
    # Find the absolute path to the configs directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "configs", f"{model_name}.json")
    
    if not os.path.exists(config_path):
        # Fallback for embedding heads which share a config
        if "embedding" in model_name:
            config_path = os.path.join(base_dir, "configs", "embedding_head.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config for {model_name} not found at {config_path}")
            
    with open(config_path, "r") as f:
        return json.load(f)

def apply_tuning_params(trial, tuning_config: dict[str, Any]) -> dict[str, Any]:
    """Apply trial.suggest_* for each parameter in the tuning config.
    
    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object.
    tuning_config : dict
        A dictionary mapping parameter names to their suggestion ranges/choices.
        Example: {"lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True}}
        
    Returns
    -------
    A dictionary of suggested parameters.
    """
    params = {}
    for name, cfg in tuning_config.items():
        type_ = cfg.get("type")
        if type_ == "float":
            params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=cfg.get("log", False))
        elif type_ == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"], step=cfg.get("step", 1), log=cfg.get("log", False))
        elif type_ == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
    return params

def seed_everything(seed: int = 1702):
    """
    https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False