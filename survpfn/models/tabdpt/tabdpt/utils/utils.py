import datetime
import functools
import os
import random
import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.using_dist:
                dist.destroy_process_group()

    return wrapper

def seed_everything(seed: int):
    """Set the random seed for reproducibility.

    Args:
        seed (int): seed value to set for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cleanup():
    """Cleanup the distributed training environment."""
    dist.destroy_process_group()


def signal_handler(*_):
    """Handle Ctrl+C signal to gracefully exit the program."""
    print("Received Ctrl+C, exiting...")
    cleanup()
    sys.exit(0)


def get_module(model: nn.Module) -> nn.Module:
    """Get the underlying module from a DistributedDataParallel model.
    Args:
        model (torch.nn.Module): The model, possibly wrapped in DistributedDataParallel.
    Returns:
        torch.nn.Module: The underlying module if model is DistributedDataParallel, else the model itself.
    """
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def log_param_norms_wandb(
    model: nn.Module, step: int, task: str, global_step: int
):
    """Log the norms of model parameters to WandB.

    Args:
        model (nn.Module): model to log parameter norms for.
        step (int): current training step, used for logging.
        task (str): regression or classification task, used for logging.
        global_step (int): current global step, used for logging.
    """
    import wandb
    model = get_module(model)
    # Log encoder weight and bias norms (if bias exists)
    wandb.log({"norms/encoder_weight": torch.norm(model.encoder.weight).item()}, step=step)
    if model.encoder.bias is not None:
        wandb.log({"norms/encoder_bias": torch.norm(model.encoder.bias).item()}, step=step)

    # Log y_encoder weight and bias norms (if bias exists)
    wandb.log({"norms/y_encoder_weight": torch.norm(model.y_encoder.weight).item()}, step=step)
    if model.y_encoder.bias is not None:
        wandb.log({"norms/y_encoder_bias": torch.norm(model.y_encoder.bias).item()}, step=step)

    # Log transformer encoder total weights and biases (if biases exist)
    transformer_weights_norm = sum(
        torch.norm(layer.kv_proj.weight).item() for layer in model.transformer_encoder
    )

    wandb.log({"norms/transformer_weights": transformer_weights_norm}, step=step)

    # Log classifier head weight and bias norms
    head_weights_norm = sum(torch.norm(param).item() for param in model.head.parameters())

    wandb.log({"norms/head_weights_biases": head_weights_norm}, step=step)

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
        ),
        2,
    )
    total_norm1 = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 1) for p in model.parameters() if p.grad is not None]
        ),
        1,
    )
    param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in model.parameters()]), 2)
    wandb.log({"gradient/norm": total_norm}, step=global_step)
    wandb.log({"gradient/norm_L1": total_norm1}, step=global_step)
    wandb.log({"parameter/norm": param_norm}, step=global_step)
    wandb.log({"prior/reg_ratio": task.float().mean().item()}, step=global_step)


def print_on_master_only(is_master: bool):
    """Override the built-in print function to only print on the master process.
    Args:
        is_master (bool): Whether the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(config):
    """Initialize distributed training.

    Args:
        device (str): The device to use for training (e.g., "cuda:0" or "cpu").

    Returns:
        bool: Whether distributed training is being used.
        int: The rank of the current process.
        str: The device to use for training.
    """
    ddp = int(os.environ.get("RANK", -1)) != -1
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        ddp_rank = int(os.environ["RANK"])
    else:
        ddp_rank = 0
        rank = 0

    seed_offset = ddp_rank if ddp else 0
    np.random.seed(config.seed + seed_offset)
    torch.manual_seed(config.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if "LOCAL_RANK" in os.environ:
        return True, rank, f"cuda:{rank}"
    else:
        return False, 0, config.env.device


class DataPreprocessor:
    def __init__(self, impute_method: str = "mean", encode_y: bool = False):
        """Initialize the DataPreprocessor.
        Parameters:
        - impute_method: str, method for imputing missing values (default is "mean").
        - encode_y: bool, whether to encode the target variable y (default is False).
        """
        self.impute_method = impute_method
        self.encode_y = encode_y
        # Initialize the imputer and scaler
        self.imputer = SimpleImputer(strategy=impute_method)
        self.scaler = StandardScaler()

    def convert_cat2num(
        self, X: pd.DataFrame, y: pd.Series | None = None, les: Optional[Dict[str, LabelEncoder]] = None
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """Convert categorical columns to numerical values.

        Parameters:
        - X: DataFrame, input features.
        - y: Series or None, target variable.

        Returns:
        - X: numpy array, features with categorical columns converted to numerical.
        - y: numpy array, target variable with categorical values converted to numerical.
        - cat_vals: list of bool, indicating which columns were categorical.
        """
        cat_vals = []
        if les is None:
            les = {}
        # Convert categorical columns to numerical values using dataframe information
        for col in X.columns:
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                if col not in les:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    les[col] = le
                else:
                    le = les.get(col, None)
                    X[col] = le.transform(X[col])
    
                cat_vals.append(True)
            else:
                cat_vals.append(False)
        X = X.to_numpy().astype(np.float32)

        # Convert categorical target to numerical values
        if y is not None:
            if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
                if "target" not in les:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    les["target"] = le
                else:
                    le = les.get("target", None)
                    y = le.transform(y)

                cat_vals.append(True)
            else:
                y = y.to_numpy().astype(np.float32)
                cat_vals.append(False)
        else:
            y = X[:, -1]
            X = np.delete(X, -1, axis=1)
        return X, y, cat_vals, les

    def fit_transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Fit the imputer and scaler on the training data and transform it.

        Parameters:
            - X: numpy array, training features.

        Returns:
            - X: numpy array, transformed training features.
        """
        # Impute missing values
        X = self.imputer.fit_transform(X)

        # Scale the features
        X = self.scaler.fit_transform(X)

        return X

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Transform the test data using the fitted imputer and scaler.

        Parameters:
            - X: numpy array, testing features.

        Returns:
            - X: numpy array, transformed testing features.
        """
        # Impute missing values
        X = self.imputer.transform(X)

        # Scale the features
        X = self.scaler.transform(X)

        return X


def standardize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize the input tensor by subtracting the mean and dividing by the standard deviation.
    Args:
        tensor (torch.Tensor): Input tensor to standardize.
    Returns:
        torch.Tensor: Standardized tensor.
        torch.Tensor: Mean of the input tensor.
        torch.Tensor: Standard deviation of the input tensor.
    """
    y_means = tensor.mean(dim=0)
    y_stds = tensor.std(dim=0) + 1e-6
    return (tensor - y_means) / y_stds, y_means, y_stds



def get_eval_pos(min_eval_pos: int, max_eval_pos: int, rank:int=0, using_dist: bool = False, device: torch.device = torch.device("cpu")) -> int:
    # Set eval_pos
    if using_dist and rank == 0:
        eval_pos_t = torch.randint(
            min_eval_pos,
            max_eval_pos + 1,
            (1,),
            device=device,
            dtype=torch.long,
        )
    elif using_dist:
        eval_pos_t = torch.empty(1, dtype=torch.long, device=device)
    else:
        eval_pos = torch.randint(
            min_eval_pos,
            max_eval_pos + 1,
            (1,),
            device=device,
            dtype=torch.long,
        ).item()

    if using_dist:
        torch.distributed.broadcast(eval_pos_t, src=0)
        eval_pos = int(eval_pos_t.item())

    return eval_pos