"""FP16 (Half Precision) conversion for SmallM models."""

import copy
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn


def convert_to_fp16(
    model: nn.Module,
    keep_fp32_modules: Optional[List[str]] = None,
    inplace: bool = False,
) -> nn.Module:
    """Convert model to FP16 (half precision).

    Args:
        model: Original model (float32)
        keep_fp32_modules: Module name patterns to keep in FP32.
                          Default: None (convert everything to FP16)
                          Note: RMSNorm already does FP32 computation internally.
        inplace: If True, modify model in place. Otherwise, create a copy.

    Returns:
        FP16 converted model
    """
    if not inplace:
        model = copy.deepcopy(model)

    keep_fp32_modules = keep_fp32_modules or []

    # Convert entire model to FP16 first
    model = model.half()

    # Optionally convert specific modules back to FP32
    if keep_fp32_modules:
        for name, module in model.named_modules():
            should_keep_fp32 = False
            for pattern in keep_fp32_modules:
                if pattern in name:
                    should_keep_fp32 = True
                    break

            if should_keep_fp32:
                module.float()

    return model


def save_fp16_checkpoint(
    model: nn.Module,
    path: str,
    original_checkpoint: Optional[Dict[str, Any]] = None,
    keep_fp32_modules: Optional[List[str]] = None,
) -> str:
    """Save FP16 model checkpoint.

    Args:
        model: FP16 converted model
        path: Output path for checkpoint
        original_checkpoint: Original checkpoint data (for metadata)
        keep_fp32_modules: Modules kept in FP32 (for metadata)

    Returns:
        Path to saved checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "quantization": {
            "type": "fp16",
            "original_dtype": "float32",
            "keep_fp32_modules": keep_fp32_modules or [],
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
