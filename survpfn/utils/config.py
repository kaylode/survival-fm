
import json
import os
from typing import Any
import random
import numpy as np
import torch

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