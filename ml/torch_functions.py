"""
A module containing useful PyTorch functions.

Author: Gemma McLean
Date: April 2026
"""

import random
import numpy as np
import torch


def set_random_seed(seed, deterministic=True):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed (int): Random seed to apply.
        deterministic (bool, optional): Whether to require deterministic PyTorch
            algorithms. Defaults to True.
    """

    # Set the random seed for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If CUDA is available, set the random seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # If deterministic is True, set PyTorch to use deterministic algorithms
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def get_device():
    """
    Get the available device (GPU or CPU) for PyTorch.

    Returns:
        torch.device: The device to be used.
    """

    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for MPS (Apple Silicon GPU) availability
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    # If neither CUDA nor MPS is available, return CPU
    else:
        return torch.device("cpu")
