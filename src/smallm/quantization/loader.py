"""Unified loader for quantized SmallM models."""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn


def load_quantized_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a quantized model checkpoint.

    Automatically detects quantization type and loads appropriately.

    Args:
        checkpoint_path: Path to quantized checkpoint
        device: Device to load model to.
               Note: INT8 models are most efficient on CPU.

    Returns:
        Tuple of (model, quantization_info)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint format is invalid
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get quantization info
    quant_info = checkpoint.get("quantization", {})
    quant_type = quant_info.get("type", "fp32")

    # Get model config
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint missing 'model_config'")

    # Load based on quantization type
    if quant_type == "fp16":
        model = _load_fp16_model(checkpoint, model_config, device)
    elif quant_type == "int8_dynamic":
        model = _load_int8_model(checkpoint, model_config, device)
        if device != "cpu":
            print(
                f"Warning: INT8 model loaded to {device}. "
                "INT8 quantization is most efficient on CPU."
            )
    else:
        # Assume FP32
        model = _load_fp32_model(checkpoint, model_config, device)

    return model, quant_info


def _load_fp32_model(
    checkpoint: Dict[str, Any],
    model_config: Dict[str, Any],
    device: str,
) -> nn.Module:
    """Load FP32 model."""
    from smallm.model import LLaMA
    from smallm.model.config import ModelConfig

    config = ModelConfig(**model_config)
    model = LLaMA(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def _load_fp16_model(
    checkpoint: Dict[str, Any],
    model_config: Dict[str, Any],
    device: str,
) -> nn.Module:
    """Load FP16 model."""
    from smallm.model import LLaMA
    from smallm.model.config import ModelConfig

    config = ModelConfig(**model_config)
    model = LLaMA(config)

    # Load state dict (already in FP16)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Ensure model is in half precision
    model = model.half()

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def _load_int8_model(
    checkpoint: Dict[str, Any],
    model_config: Dict[str, Any],
    device: str,
) -> nn.Module:
    """Load INT8 dynamically quantized model.

    INT8 models should stay on CPU for efficiency.
    """
    from smallm.model import LLaMA
    from smallm.model.config import ModelConfig
    import torch.ao.quantization as quant

    # Create FP32 model first
    config = ModelConfig(**model_config)
    model = LLaMA(config)
    model.eval()

    # Apply dynamic quantization structure
    model = quant.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Load quantized state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # INT8 models work best on CPU
    # If device is not CPU, we can still move but warn the user
    if device != "cpu":
        # Note: Moving quantized model to GPU may dequantize it
        model = model.to(device)

    return model


def get_quantization_info(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Get quantization info from checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Quantization info dict or None if not a quantized checkpoint
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    # Load only metadata (weights_only would be ideal but need full load for quantization info)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return checkpoint.get("quantization")
