from enum import Enum
import numpy as np
import torch
from functools import wraps
from torch.nn.attention import SDPBackend, sdpa_kernel

class Task(Enum):
    """Enum representing the type of task for the model.
    REG: Regression task.
    CLS: Classification task.
    """

    REG = 1
    CLS = 2


def pad_x(X: torch.Tensor, num_features: int) -> torch.Tensor:
    """Pad the input tensor X with zeros to match the specified number of features.
    Args:
        X (torch.Tensor): Input tensor of shape (seq_len, batch_size, n_features) or (seq_len, n_features).
        num_features (int): Desired number of features after padding.
    Returns:
        torch.Tensor: Padded tensor of shape (seq_len, batch_size, num_features).
    """

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if len(X.shape) != 3:
        seq_len, n_features = X.shape
        zero_feature_padding = torch.zeros(
            (seq_len, num_features - n_features), device=X.device
        )
    else:
        seq_len, batch_size, n_features = X.shape
        zero_feature_padding = torch.zeros(
            (seq_len, batch_size, num_features - n_features), device=X.device
        )
    return torch.cat([X, zero_feature_padding], -1)


def maskmean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the mean of x along the specified dimension, ignoring masked values.
    Args:
        x (torch.Tensor): Input tensor of shape (time, batch, hidden dimension).
        mask (torch.Tensor): Mask tensor of the same shape as x, where True indicates valid values.
        dim (int): Dimension along which to compute the mean.
    Returns:
        torch.Tensor: Mean of x along the specified dimension, with masked values ignored.
    """
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x: torch.Tensor, mask: torch.Tensor, dim: int = 0):
    """Compute the standard deviation of x along the specified dimension, ignoring masked values.
    Args:
        x (torch.Tensor): Input tensor of shape (time, batch, hidden dimension).
        mask (torch.Tensor): Mask tensor of the same shape as x, where True indicates valid values.
        dim (int): Dimension along which to compute the standard deviation.
    Returns:
        torch.Tensor: Standard deviation of x along the specified dimension, with masked values ignored.
    """
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data: torch.Tensor, eval_pos: int) -> torch.Tensor:
    """Normalize the input data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (torch.Tensor): input data of shape (time, batch, hidden dimension).
        eval_pos (int): Evaluation position used for slicing attention keys and values.

    Returns:
        torch.Tensor: normalized data of shape (time, batch, hidden dimension).
    """
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data


def clip_outliers(data: torch.Tensor, eval_pos: int, n_sigma: int = 4):
    """
    clip outliers in data based on a given number of standard deviations.

    Args:
        data (torch.Tensor): Input data of shape (time, batch, hidden dimension).
        eval_pos (int): Evaluation position used for slicing attention keys and values.
        n_sigma (int): Number of standard deviations to use for clipping outliers.
    Returns:
        torch.Tensor: Data with outliers clipped, of shape (time, batch, hidden dimension)."""
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask = mask & (cutoff >= torch.abs(X - mean))
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a NumPy array or a PyTorch tensor to a PyTorch tensor.
    Args:
        input (np.ndarray | torch.Tensor): Input data to be converted.
    Returns:
        torch.Tensor: Converted PyTorch tensor.
    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")



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



def flash_context(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "use_flash", False):
            assert torch.cuda.is_available(), "FlashAttention requires CUDA support"
            bf_support = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if bf_support else torch.float16
            with torch.autocast(device_type='cuda', dtype=dtype), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper
