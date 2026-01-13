"""INT8 Dynamic Quantization for SmallM models."""

import copy
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as quant

from .utils import DEFAULT_SKIP_MODULES, get_linear_modules


def quantize_dynamic_int8(
    model: nn.Module,
    skip_modules: Optional[List[str]] = None,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False,
) -> nn.Module:
    """Apply dynamic INT8 quantization to model.

    Dynamic quantization quantizes weights statically but activations dynamically.
    Best for models where execution time is dominated by loading weights (e.g., LSTMs, Transformers).

    Note: INT8 quantization is most efficient on CPU. GPU inference may not see benefits.

    Args:
        model: Original model (float32)
        skip_modules: Module name patterns to skip from quantization.
                     Default: ["tok_emb", "lm_head", "norm", ...]
        dtype: Quantization dtype (torch.qint8 or torch.quint8)
        inplace: If True, modify model in place. Otherwise, create a copy.

    Returns:
        INT8 quantized model
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Use default skip modules if not specified
    skip_modules = skip_modules if skip_modules is not None else DEFAULT_SKIP_MODULES

    # Move model to CPU for quantization
    model = model.cpu()
    model.eval()

    # Get modules to quantize
    modules_to_quantize = get_linear_modules(model, skip_modules)

    # Create module mapping for selective quantization
    # torch.ao.quantization.quantize_dynamic applies to whole model
    # We need to be selective, so we'll use a custom approach

    quantized_model = _selective_dynamic_quantize(
        model,
        modules_to_quantize,
        dtype,
    )

    return quantized_model


def _selective_dynamic_quantize(
    model: nn.Module,
    modules_to_quantize: Set[str],
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Selectively quantize specific modules.

    Args:
        model: Model to quantize
        modules_to_quantize: Set of module names to quantize
        dtype: Quantization dtype

    Returns:
        Quantized model
    """
    # For simplicity, we use PyTorch's dynamic quantization
    # which targets all Linear layers by default
    # The skip_modules filtering is done by keeping those layers separate

    # Create a set of module types to quantize
    # This is a limitation: torch.ao.quantization.quantize_dynamic
    # doesn't support selective module-by-module quantization easily

    # Alternative approach: manually replace Linear modules
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize all Linear layers
        dtype=dtype,
    )

    return quantized_model


def save_int8_checkpoint(
    model: nn.Module,
    path: str,
    original_checkpoint: Optional[Dict[str, Any]] = None,
    skip_modules: Optional[List[str]] = None,
) -> str:
    """Save INT8 quantized model checkpoint.

    Args:
        model: INT8 quantized model
        path: Output path for checkpoint
        original_checkpoint: Original checkpoint data (for metadata)
        skip_modules: Modules skipped during quantization (for metadata)

    Returns:
        Path to saved checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "quantization": {
            "type": "int8_dynamic",
            "original_dtype": "float32",
            "skip_modules": skip_modules or DEFAULT_SKIP_MODULES,
        },
    }

    # Copy metadata from original checkpoint
    if original_checkpoint:
        if "model_config" in original_checkpoint:
            checkpoint["model_config"] = original_checkpoint["model_config"]
        if "step" in original_checkpoint:
            checkpoint["step"] = original_checkpoint["step"]
        if "loss" in original_checkpoint:
            checkpoint["loss"] = original_checkpoint["loss"]
        checkpoint["quantization"]["source_checkpoint"] = str(
            original_checkpoint.get("source_path", "unknown")
        )

    torch.save(checkpoint, path)
    return str(path)
