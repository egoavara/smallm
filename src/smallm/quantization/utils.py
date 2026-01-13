"""Quantization utility functions."""

from typing import Set, List
import torch
import torch.nn as nn


def get_model_size(model: nn.Module) -> int:
    """Get model size in bytes.

    Args:
        model: PyTorch model

    Returns:
        Total size in bytes
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size


def get_model_size_str(model: nn.Module) -> str:
    """Get human-readable model size string.

    Args:
        model: PyTorch model

    Returns:
        Size string (e.g., "12.3 MB")
    """
    size_bytes = get_model_size(model)
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    from pathlib import Path
    return Path(path).stat().st_size


def get_file_size_str(path: str) -> str:
    """Get human-readable file size string."""
    size_bytes = get_file_size(path)
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def get_linear_modules(model: nn.Module, skip_patterns: List[str] = None) -> Set[str]:
    """Get names of Linear modules, excluding those matching skip patterns.

    Args:
        model: PyTorch model
        skip_patterns: List of patterns to exclude (e.g., ["tok_emb", "lm_head"])

    Returns:
        Set of module names that are Linear and not skipped
    """
    skip_patterns = skip_patterns or []
    linear_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = False
            for pattern in skip_patterns:
                if pattern in name:
                    should_skip = True
                    break
            if not should_skip:
                linear_modules.add(name)

    return linear_modules


def count_quantized_params(model: nn.Module) -> dict:
    """Count parameters by dtype.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping dtype string to parameter count
    """
    dtype_counts = {}

    for param in model.parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in dtype_counts:
            dtype_counts[dtype_str] = 0
        dtype_counts[dtype_str] += param.numel()

    return dtype_counts


# Default modules to skip during quantization
# These are excluded for stability (weight tying, normalization)
DEFAULT_SKIP_MODULES = [
    "tok_emb",       # Embedding (weight tied with lm_head)
    "lm_head",       # Output projection (weight tied with tok_emb)
    "norm",          # RMSNorm layers
    "attention_norm",
    "ffn_norm",
]
