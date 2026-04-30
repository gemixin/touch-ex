"""
A collection of useful functions for PyTorch.

Author: Gemma McLean
Date: April 2026
"""

import torch


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
